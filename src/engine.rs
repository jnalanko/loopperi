//! The looper itself: audio devices, the playback/overdub streams, the loop
//! buffer, the metronome and overdub latency compensation. It never prints or
//! reads input -- a front end drives it through `Engine`'s methods and draws
//! whatever it likes from its read-only accessors.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};

/// Some interfaces (e.g. the Roland QUAD-CAPTURE) expose extra channels
/// beyond the real analog inputs -- such as a hardware loopback pair that
/// echoes back whatever is currently playing. Keeping only the first
/// `used_channels` of each `device_channels`-wide frame drops those extras.
fn select_channels(
    data: impl Iterator<Item = f32>,
    device_channels: usize,
    used_channels: usize,
) -> impl Iterator<Item = f32> {
    data.enumerate()
        .filter(move |(i, _)| i % device_channels < used_channels)
        .map(|(_, s)| s)
}

/// Identity below the threshold -- so mixing in silence never alters existing
/// loop content -- and a smooth knee curving anything above it toward +-1.0
/// instead of hard-clipping.
fn soft_clip(x: f32) -> f32 {
    const THRESHOLD: f32 = 0.8;
    let mag = x.abs();
    if mag <= THRESHOLD {
        x
    } else {
        let over = mag - THRESHOLD;
        let curved = THRESHOLD + (1.0 - THRESHOLD) * (over / (over + (1.0 - THRESHOLD)));
        x.signum() * curved
    }
}

/// Mixes freshly captured audio into the loop buffer starting at `write_frame`,
/// advancing it (wrapping at `loop_frames`) as samples are consumed. The
/// cursor is owned entirely by the overdub input stream -- it is seeded once
/// from the playback position when overdubbing starts and then never
/// re-synced, so a single overdub pass writes each loop frame exactly once
/// even if the input and output streams' callback timings don't line up.
fn mix_into_loop(
    loop_buf: &Mutex<Vec<f32>>,
    write_frame: &mut usize,
    loop_frames: usize,
    channels: usize,
    data: impl Iterator<Item = f32>,
) {
    if loop_frames == 0 {
        return;
    }
    let mut buf = loop_buf.lock().unwrap();
    let mut ch = 0usize;
    for sample in data {
        let idx = *write_frame * channels + ch;
        if let Some(existing) = buf.get_mut(idx) {
            *existing = soft_clip(*existing + sample * 0.8);
        }
        ch += 1;
        if ch == channels {
            ch = 0;
            *write_frame = (*write_frame + 1) % loop_frames;
        }
    }
}

fn build_overdub_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    start_frame: usize,
    loop_frames: usize,
    device_channels: u16,
    used_channels: u16,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream_config = config.config();
    let device_channels = device_channels as usize;
    let used_channels = used_channels as usize;
    let mut write_frame = if loop_frames == 0 { 0 } else { start_frame % loop_frames };

    match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let samples = select_channels(data.iter().copied(), device_channels, used_channels);
                mix_into_loop(&loop_buf, &mut write_frame, loop_frames, used_channels, samples);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let samples = select_channels(
                    data.iter().map(|s| s.to_sample::<f32>()),
                    device_channels,
                    used_channels,
                );
                mix_into_loop(&loop_buf, &mut write_frame, loop_frames, used_channels, samples);
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let samples = select_channels(
                    data.iter().map(|s| s.to_sample::<f32>()),
                    device_channels,
                    used_channels,
                );
                mix_into_loop(&loop_buf, &mut write_frame, loop_frames, used_channels, samples);
            },
            err_fn,
            None,
        ),
        sample_format => panic!("unsupported input sample format: {sample_format}"),
    }
}

/// Click track derived from the tempo grid the loop was built on. It is mixed
/// into the output stream only -- never into `loop_buf` -- so toggling it never
/// alters the recorded material.
struct Metronome {
    enabled: AtomicBool,
    frames_per_beat: usize,
    beats_per_measure: usize,
    /// Length of one click, in frames.
    click_frames: usize,
    sample_rate: f32,
}

impl Metronome {
    fn toggle(&self) -> bool {
        !self.enabled.fetch_xor(true, Ordering::Relaxed)
    }

    /// A decaying sine blip at the start of every beat, pitched higher on the
    /// first beat of each measure so the downbeat is audible.
    fn click_at(&self, frame: usize) -> f32 {
        if !self.enabled.load(Ordering::Relaxed) {
            return 0.0;
        }
        let offset = frame % self.frames_per_beat;
        if offset >= self.click_frames {
            return 0.0;
        }
        let downbeat = (frame / self.frames_per_beat).is_multiple_of(self.beats_per_measure);
        let freq = if downbeat { 1600.0 } else { 1000.0 };
        let decay = 1.0 - offset as f32 / self.click_frames as f32;
        let phase = std::f32::consts::TAU * freq * offset as f32 / self.sample_rate;
        phase.sin() * decay * decay * 0.4
    }
}

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    play_pos: Arc<AtomicUsize>,
    loop_frames: usize,
    recorded_channels: u16,
    metronome: Arc<Metronome>,
) -> Result<Stream, cpal::BuildStreamError> {
    match config.sample_format() {
        SampleFormat::F32 => {
            build_output_stream_of::<f32>(device, config, loop_buf, play_pos, loop_frames, recorded_channels, metronome)
        }
        SampleFormat::I16 => {
            build_output_stream_of::<i16>(device, config, loop_buf, play_pos, loop_frames, recorded_channels, metronome)
        }
        SampleFormat::U16 => {
            build_output_stream_of::<u16>(device, config, loop_buf, play_pos, loop_frames, recorded_channels, metronome)
        }
        sample_format => panic!("unsupported output sample format: {sample_format}"),
    }
}

fn build_output_stream_of<T>(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    play_pos: Arc<AtomicUsize>,
    loop_frames: usize,
    recorded_channels: u16,
    metronome: Arc<Metronome>,
) -> Result<Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    let err_fn = |err| eprintln!("output stream error: {err}");
    let stream_config = config.config();
    let out_channels = stream_config.channels as usize;
    let rec_channels = recorded_channels as usize;

    device.build_output_stream(
        &stream_config,
        move |data: &mut [T], _| {
            let buf = loop_buf.lock().unwrap();
            let mut frame = play_pos.load(Ordering::Relaxed);
            for out_frame in data.chunks_mut(out_channels) {
                let click = metronome.click_at(frame);
                for (ch, sample) in out_frame.iter_mut().enumerate() {
                    let src_channel = if rec_channels == 1 { 0 } else { ch % rec_channels };
                    let value = buf[frame * rec_channels + src_channel] + click;
                    *sample = soft_clip(value).to_sample::<T>();
                }
                frame = (frame + 1) % loop_frames;
            }
            play_pos.store(frame, Ordering::Relaxed);
        },
        err_fn,
        None,
    )
}

/// On Linux/PulseAudio, `default_input_config()` can report fewer channels
/// than the underlying source really has: PulseAudio silently downmixes
/// multi-channel sources to plain stereo for clients that don't ask for
/// more, and that downmix can fold in channels we don't want (e.g. some
/// interfaces, like the Roland QUAD-CAPTURE, expose a hardware loopback pair
/// alongside the real analog inputs). This asks `pactl` for the current
/// default source's true channel count so we can request the full,
/// un-downmixed stream instead. Best-effort: returns `None` if `pactl` isn't
/// present or parsing fails, and callers should fall back to the default.
fn true_input_channel_count() -> Option<u16> {
    let default_source = pactl_run(&["get-default-source"])?.trim().to_string();
    let sources = pactl_run(&["list", "sources"])?;

    let mut in_block = false;
    for line in sources.lines() {
        let line = line.trim();
        if let Some(name) = line.strip_prefix("Name: ") {
            in_block = name == default_source;
        } else if in_block {
            if let Some(spec) = line.strip_prefix("Sample Specification: ") {
                // e.g. "s32le 6ch 44100Hz"
                return spec.split_whitespace().find_map(|part| part.strip_suffix("ch")?.parse().ok());
            }
        }
    }
    None
}

fn pactl_run(args: &[&str]) -> Option<String> {
    String::from_utf8(std::process::Command::new("pactl").args(args).output().ok()?.stdout).ok()
}

/// Finds a supported input config with exactly `channels` channels that also
/// supports `sample_rate` -- matching the device's already-correct default
/// sample rate rather than picking the config's own (possibly synthetic, on
/// a PulseAudio virtual device) maximum rate -- and a sample format we can
/// actually decode (a device may advertise the same channel/rate combo under
/// several sample formats, including ones we don't handle).
fn pick_input_config_with_channels(
    device: &cpal::Device,
    channels: u16,
    sample_rate: cpal::SampleRate,
) -> Option<cpal::SupportedStreamConfig> {
    let format_rank = |c: &cpal::SupportedStreamConfigRange| match c.sample_format() {
        SampleFormat::F32 => Some(0),
        SampleFormat::I16 => Some(1),
        SampleFormat::U16 => Some(2),
        _ => None,
    };
    let mut candidates: Vec<_> = device
        .supported_input_configs()
        .ok()?
        .filter(|c| c.channels() == channels && c.min_sample_rate() <= sample_rate && sample_rate <= c.max_sample_rate())
        .filter_map(|c| Some((format_rank(&c)?, c)))
        .collect();
    candidates.sort_by_key(|(rank, _)| *rank);
    let (_, best) = candidates.into_iter().next()?;
    Some(best.with_sample_rate(sample_rate))
}

/// How many leading channels of the input device we actually record/mix.
/// Capping at 2 keeps things simple (mono/stereo) and, as a side effect,
/// ignores any extra channels (loopback, S/PDIF, etc.) some interfaces tack
/// on after the real analog inputs.
const MAX_USED_CHANNELS: u16 = 2;

/// Loop geometry asked for on the command line. A "beat" is the note value
/// named by the time signature's denominator, which is also what the tempo
/// counts -- so 90 6/8 means 90 eighth notes per minute, six to a measure.
pub struct LoopSpec {
    pub tempo_bpm: f64,
    pub beats_per_measure: usize,
    pub beat_unit: u32,
    pub measures: usize,
}

/// Base round-trip latency (mic + speaker) assumed for overdub timing
/// compensation, adjustable at runtime with `Engine::nudge_latency`.
const BASE_LATENCY_MS: f64 = 90.0;
const DEFAULT_TRIM_MS: f64 = 0.0;

/// Devices/configs/channel counts needed to build streams.
struct Audio {
    input_device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    output_device: cpal::Device,
    output_config: cpal::SupportedStreamConfig,
    device_channels: u16,
    used_channels: u16,
}

/// Static facts about the running engine, for a front end to display.
pub struct EngineInfo {
    pub input_device: String,
    pub output_device: String,
    pub input_rate: u32,
    pub input_device_channels: u16,
    pub used_channels: u16,
    pub output_rate: u32,
    pub output_channels: u16,
    pub beats: usize,
    pub loop_seconds: f64,
}

/// An overdub in progress.
struct Overdub {
    /// Mixes captured audio into `loop_buf` for as long as it is alive.
    in_stream: Stream,
    /// Snapshot of `loop_buf` taken right before overdubbing started, so
    /// a cancelled overdub can be discarded by restoring it verbatim.
    pre_overdub: Vec<f32>,
}

pub struct Engine {
    audio: Audio,
    /// Plays the loop (plus metronome) for as long as it is alive.
    _out_stream: Stream,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    play_pos: Arc<AtomicUsize>,
    loop_frames: usize,
    metronome: Arc<Metronome>,
    trim_ms: f64,
    overdub: Option<Overdub>,
    /// `waveform_peaks` result and the width it was computed for. Cleared
    /// whenever what's displayed can change, so redrawing doesn't rescan (and
    /// lock) the whole loop buffer every frame.
    peaks: Option<(usize, Vec<f32>)>,
    info: EngineInfo,
}

impl Engine {
    /// Opens the default input/output devices and starts playing an empty
    /// loop sized from `spec`.
    pub fn new(spec: &LoopSpec) -> Result<Engine, Box<dyn std::error::Error>> {
        let host = cpal::default_host();

        let input_device = host
            .default_input_device()
            .ok_or("no input audio device available")?;
        let output_device = host
            .default_output_device()
            .ok_or("no output audio device available")?;

        let default_input_config = input_device.default_input_config()?;
        let input_config = true_input_channel_count()
            .filter(|&n| n > default_input_config.channels())
            .and_then(|n| pick_input_config_with_channels(&input_device, n, default_input_config.sample_rate()))
            .unwrap_or(default_input_config);
        let output_config = output_device.default_output_config()?;
        let used_channels = input_config.channels().min(MAX_USED_CHANNELS);

        // The loop grid is measured against the output clock, since that is what
        // paces playback and therefore the tempo you actually hear. Deriving the
        // loop length from whole beats keeps the beat grid aligned across the wrap.
        let playback_rate = output_config.sample_rate().0 as f64;
        let frames_per_beat = (60.0 / spec.tempo_bpm * playback_rate).round() as usize;
        let beats = spec.beats_per_measure * spec.measures;
        let loop_frames = frames_per_beat * beats;

        let metronome = Arc::new(Metronome {
            enabled: AtomicBool::new(false),
            frames_per_beat,
            beats_per_measure: spec.beats_per_measure,
            click_frames: (playback_rate * 0.02) as usize,
            sample_rate: playback_rate as f32,
        });

        let info = EngineInfo {
            input_device: input_device.name()?,
            output_device: output_device.name()?,
            input_rate: input_config.sample_rate().0,
            input_device_channels: input_config.channels(),
            used_channels,
            output_rate: output_config.sample_rate().0,
            output_channels: output_config.channels(),
            beats,
            loop_seconds: loop_frames as f64 / playback_rate,
        };

        let audio = Audio {
            device_channels: input_config.channels(),
            input_device,
            input_config,
            output_device,
            output_config,
            used_channels,
        };

        let loop_buf = Arc::new(Mutex::new(vec![0.0; loop_frames * used_channels as usize]));
        let play_pos = Arc::new(AtomicUsize::new(0));
        let out_stream = build_output_stream(
            &audio.output_device,
            &audio.output_config,
            loop_buf.clone(),
            play_pos.clone(),
            loop_frames,
            audio.used_channels,
            metronome.clone(),
        )?;
        out_stream.play()?;

        Ok(Engine {
            audio,
            _out_stream: out_stream,
            loop_buf,
            play_pos,
            loop_frames,
            metronome,
            trim_ms: DEFAULT_TRIM_MS,
            overdub: None,
            peaks: None,
            info,
        })
    }

    pub fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// Starts an overdub, or stops (and keeps) the one in progress.
    pub fn toggle_overdub(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.overdub.take().is_some() {
            // Dropping the overdub stopped capturing/mixing; the loop keeps
            // playing, now with the kept take in it.
            self.peaks = None;
            return Ok(());
        }
        // Audio captured "now" corresponds to what was heard roughly
        // `latency_frames` ago (mic + speaker round-trip delay), so seed
        // the overdub write cursor that far behind the current playback
        // position instead of exactly on it.
        let sample_rate = self.audio.input_config.sample_rate().0 as f64;
        let latency_frames = (self.latency_ms() / 1000.0 * sample_rate).round() as isize;
        let start_frame = (self.play_pos.load(Ordering::Relaxed) as isize - latency_frames)
            .rem_euclid(self.loop_frames as isize) as usize;
        let pre_overdub = self.loop_buf.lock().unwrap().clone();
        let in_stream = build_overdub_input_stream(
            &self.audio.input_device,
            &self.audio.input_config,
            self.loop_buf.clone(),
            start_frame,
            self.loop_frames,
            self.audio.device_channels,
            self.audio.used_channels,
        )?;
        in_stream.play()?;
        self.overdub = Some(Overdub { in_stream, pre_overdub });
        Ok(())
    }

    /// Aborts an overdub in progress and discards it, rather than keeping it
    /// like `toggle_overdub` would -- `loop_buf` is restored to its
    /// pre-overdub snapshot, undoing whatever was mixed in so far. Returns
    /// whether there was an overdub to cancel.
    pub fn cancel_overdub(&mut self) -> bool {
        let Some(Overdub { in_stream, pre_overdub }) = self.overdub.take() else {
            return false;
        };
        drop(in_stream); // stop capturing/mixing before restoring
        *self.loop_buf.lock().unwrap() = pre_overdub;
        self.peaks = None;
        true
    }

    /// Drops any overdub in progress and empties the loop, keeping its tempo
    /// grid and playback timeline intact -- the loop's length is fixed by the
    /// `LoopSpec`, so there is nothing to re-measure.
    pub fn clear(&mut self) {
        self.overdub = None; // stop capturing/mixing
        self.loop_buf.lock().unwrap().fill(0.0);
        self.peaks = None;
    }

    /// Returns whether the metronome is now on.
    pub fn toggle_metronome(&self) -> bool {
        self.metronome.toggle()
    }

    pub fn metronome_on(&self) -> bool {
        self.metronome.enabled.load(Ordering::Relaxed)
    }

    /// Overdub latency compensation currently in effect, in milliseconds.
    pub fn latency_ms(&self) -> f64 {
        BASE_LATENCY_MS + self.trim_ms
    }

    /// Adjusts overdub latency compensation; applies from the next overdub.
    pub fn nudge_latency(&mut self, delta_ms: f64) {
        self.trim_ms += delta_ms;
    }

    pub fn is_overdubbing(&self) -> bool {
        self.overdub.is_some()
    }

    /// Where playback sits within the loop, from 0.0 up to (not including) 1.0.
    pub fn position(&self) -> f64 {
        self.play_pos.load(Ordering::Relaxed) as f64 / self.loop_frames as f64
    }

    /// Peak level (0.0..=1.0, across all channels) of each of `width` equal
    /// slices of what's audible right now. During an overdub this is taken
    /// from the pre-overdub snapshot, so the take being recorded doesn't show
    /// up until it's kept -- which also means the result only changes when
    /// an overdub ends or the loop is cleared, so it is cached until then.
    pub fn waveform_peaks(&mut self, width: usize) -> &[f32] {
        if self.peaks.as_ref().is_none_or(|(w, _)| *w != width) {
            let peaks = match &self.overdub {
                Some(overdub) => column_peaks(&overdub.pre_overdub, self.loop_frames, width),
                None => column_peaks(&self.loop_buf.lock().unwrap(), self.loop_frames, width),
            };
            self.peaks = Some((width, peaks));
        }
        &self.peaks.as_ref().unwrap().1
    }
}

/// Splits an interleaved, `loop_frames`-long buffer into `width` equal slices
/// and returns the loudest sample (across all channels) in each, capped at 1.0.
fn column_peaks(buf: &[f32], loop_frames: usize, width: usize) -> Vec<f32> {
    let Some(channels) = buf.len().checked_div(loop_frames) else {
        return vec![0.0; width];
    };
    (0..width)
        .map(|col| {
            let start = col * loop_frames / width;
            let end = ((col + 1) * loop_frames / width).max(start + 1).min(loop_frames);
            buf[start * channels..end * channels]
                .iter()
                .fold(0.0f32, |acc, s| acc.max(s.abs()))
                .min(1.0)
        })
        .collect()
}
