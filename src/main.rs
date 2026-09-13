use std::io::{self, Write as _};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal;

enum State {
    Idle,
    Recording {
        in_stream: Stream,
        buffer: Arc<Mutex<Vec<f32>>>,
        started_at: Instant,
    },
    Looping {
        out_stream: Stream,
        loop_buf: Arc<Mutex<Vec<f32>>>,
        play_pos: Arc<AtomicUsize>,
        loop_frames: usize,
    },
    Overdubbing {
        out_stream: Stream,
        in_stream: Stream,
        loop_buf: Arc<Mutex<Vec<f32>>>,
        play_pos: Arc<AtomicUsize>,
        loop_frames: usize,
        /// Snapshot of `loop_buf` taken right before overdubbing started, so
        /// a cancelled overdub can be discarded by restoring it verbatim.
        pre_overdub: Vec<f32>,
    },
}

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

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    device_channels: u16,
    used_channels: u16,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream_config = config.config();
    let device_channels = device_channels as usize;
    let used_channels = used_channels as usize;

    match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(select_channels(data.iter().copied(), device_channels, used_channels));
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(select_channels(
                    data.iter().map(|s| s.to_sample::<f32>()),
                    device_channels,
                    used_channels,
                ));
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(select_channels(
                    data.iter().map(|s| s.to_sample::<f32>()),
                    device_channels,
                    used_channels,
                ));
            },
            err_fn,
            None,
        ),
        sample_format => panic!("unsupported input sample format: {sample_format}"),
    }
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

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    play_pos: Arc<AtomicUsize>,
    loop_frames: usize,
    recorded_channels: u16,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("output stream error: {err}");
    let stream_config = config.config();
    let out_channels = stream_config.channels as usize;
    let rec_channels = recorded_channels as usize;

    macro_rules! advance_frame {
        ($frame:ident) => {{
            $frame = ($frame + 1) % loop_frames.max(1);
        }};
    }

    match config.sample_format() {
        SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                if loop_frames == 0 {
                    data.fill(0.0);
                    return;
                }
                let buf = loop_buf.lock().unwrap();
                let mut frame = play_pos.load(Ordering::Relaxed);
                for out_frame in data.chunks_mut(out_channels) {
                    for (ch, sample) in out_frame.iter_mut().enumerate() {
                        let src_channel = if rec_channels == 1 { 0 } else { ch % rec_channels };
                        *sample = buf[frame * rec_channels + src_channel];
                    }
                    advance_frame!(frame);
                }
                play_pos.store(frame, Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [i16], _| {
                if loop_frames == 0 {
                    data.fill(0);
                    return;
                }
                let buf = loop_buf.lock().unwrap();
                let mut frame = play_pos.load(Ordering::Relaxed);
                for out_frame in data.chunks_mut(out_channels) {
                    for (ch, sample) in out_frame.iter_mut().enumerate() {
                        let src_channel = if rec_channels == 1 { 0 } else { ch % rec_channels };
                        *sample = buf[frame * rec_channels + src_channel].to_sample::<i16>();
                    }
                    advance_frame!(frame);
                }
                play_pos.store(frame, Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [u16], _| {
                if loop_frames == 0 {
                    data.fill(u16::MAX / 2);
                    return;
                }
                let buf = loop_buf.lock().unwrap();
                let mut frame = play_pos.load(Ordering::Relaxed);
                for out_frame in data.chunks_mut(out_channels) {
                    for (ch, sample) in out_frame.iter_mut().enumerate() {
                        let src_channel = if rec_channels == 1 { 0 } else { ch % rec_channels };
                        *sample = buf[frame * rec_channels + src_channel].to_sample::<u16>();
                    }
                    advance_frame!(frame);
                }
                play_pos.store(frame, Ordering::Relaxed);
            },
            err_fn,
            None,
        ),
        sample_format => panic!("unsupported output sample format: {sample_format}"),
    }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    println!("Input device:  {}", input_device.name()?);
    println!("Output device: {}", output_device.name()?);
    println!(
        "Recording at {} Hz, using {} of {} device input channels; playing back at {} Hz / {} ch",
        input_config.sample_rate().0,
        used_channels,
        input_config.channels(),
        output_config.sample_rate().0,
        output_config.channels()
    );
    if input_config.sample_rate() != output_config.sample_rate() {
        println!(
            "Note: input/output sample rates differ, playback speed/pitch may be off (no resampling is done)."
        );
    }

    println!();
    println!("SPACE: idle -> record -> loop -> overdub -> loop -> overdub -> ...");
    println!("R: reset to idle (discard the current loop)    |    Esc/Ctrl+C: quit");
    println!("Backspace: cancel the current recording/overdub");
    println!("Left/Right arrows: nudge overdub timing earlier/later");
    println!();

    terminal::enable_raw_mode()?;
    let result = run(input_device, input_config, output_device, output_config, used_channels);
    terminal::disable_raw_mode()?;
    result
}

/// Base round-trip latency (mic + speaker) assumed for overdub timing
/// compensation, adjustable at runtime with Left/Right.
const BASE_LATENCY_MS: f64 = 100.0;
const LATENCY_STEP_MS: f64 = 5.0;
const DEFAULT_TRIM_MS: f64 = 0.0;

/// Devices/configs/channel counts needed to build streams -- bundled up so
/// `advance` doesn't have to take them all as separate arguments.
struct Audio {
    input_device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    output_device: cpal::Device,
    output_config: cpal::SupportedStreamConfig,
    device_channels: u16,
    used_channels: u16,
}

fn run(
    input_device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    output_device: cpal::Device,
    output_config: cpal::SupportedStreamConfig,
    used_channels: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let sample_rate = input_config.sample_rate().0 as f64;
    let audio = Audio {
        device_channels: input_config.channels(),
        input_device,
        input_config,
        output_device,
        output_config,
        used_channels,
    };
    let mut state = State::Idle;
    let mut trim_ms = DEFAULT_TRIM_MS;

    loop {
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char(' ') => {
                        let latency_ms = BASE_LATENCY_MS + trim_ms;
                        let latency_frames = (latency_ms / 1000.0 * sample_rate).round() as isize;
                        if matches!(state, State::Looping { .. }) {
                            print_status(&format!("Starting overdub with {latency_ms:.0} ms compensation"));
                        }
                        state = advance(state, &audio, latency_frames)?;
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        state = State::Idle;
                        print_status("Reset. Press SPACE to record a new loop.");
                    }
                    KeyCode::Left => {
                        trim_ms -= LATENCY_STEP_MS;
                        print_status(&format!("Overdub timing trim: {trim_ms:+.0} ms (total {:.0} ms)", BASE_LATENCY_MS + trim_ms));
                    }
                    KeyCode::Right => {
                        trim_ms += LATENCY_STEP_MS;
                        print_status(&format!("Overdub timing trim: {trim_ms:+.0} ms (total {:.0} ms)", BASE_LATENCY_MS + trim_ms));
                    }
                    KeyCode::Backspace => {
                        state = cancel(state);
                    }
                    KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => break,
                    _ => {}
                },
                _ => {}
            }
        }
        render_progress(&state);
    }

    Ok(())
}

const BAR_WIDTH: usize = 40;

const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

fn render_bar(fraction: f64, width: usize, recording: bool) -> String {
    let filled = ((fraction.clamp(0.0, 1.0)) * width as f64).round() as usize;
    let bar = format!("[{}{}]", "#".repeat(filled), "-".repeat(width - filled));
    if recording {
        format!("{RED}{bar}{RESET}")
    } else {
        bar
    }
}

/// Row height of the waveform stem, in terminal lines.
const WAVE_HEIGHT: usize = 9;
/// Max number of lines `render_progress` can ever draw (waveform + bar),
/// used by `print_status` to blank out any leftover lines below it.
const MAX_PROGRESS_LINES: usize = WAVE_HEIGHT + 1;

/// Renders a fixed-`width`, `WAVE_HEIGHT`-tall ASCII peak waveform of an
/// interleaved `channels`-wide, `loop_frames`-long audio buffer: one column
/// per width slot, bars growing up from a baseline, each column's height
/// picked from the loudest sample (across all channels) in that column's
/// slice of the loop. Returned top row first, so printing the array in order
/// draws the stem right-side up.
fn render_waveform_rows(buf: &[f32], loop_frames: usize, width: usize) -> [String; WAVE_HEIGHT] {
    let levels: Vec<usize> = if let Some(channels) = buf.len().checked_div(loop_frames) {
        (0..width)
            .map(|col| {
                let start = col * loop_frames / width;
                let end = ((col + 1) * loop_frames / width).max(start + 1).min(loop_frames);
                let peak = buf[start * channels..end * channels]
                    .iter()
                    .fold(0.0f32, |acc, s| acc.max(s.abs()));
                (peak.min(1.0) * WAVE_HEIGHT as f32).round() as usize
            })
            .collect()
    } else {
        vec![0; width]
    };
    std::array::from_fn(|row| {
        let threshold = WAVE_HEIGHT - row;
        levels.iter().map(|&level| if level >= threshold { '#' } else { ' ' }).collect()
    })
}

/// Redraws an in-place status block for the current state: for
/// Looping/Overdubbing, a `WAVE_HEIGHT`-row waveform of what's actually
/// audible right now, stacked above a progress bar showing where playback
/// sits within the loop; for Recording, just an elapsed-time counter (there's
/// no committed loop yet to visualize). During an overdub the waveform is
/// drawn from `pre_overdub` -- the loop as it was *before* this take -- so
/// the take being recorded doesn't show up until it's kept.
///
/// Leaves the cursor at the start of the block's first line so the next call
/// (or `print_status`) can redraw it in place without scrolling.
fn render_progress(state: &State) {
    match state {
        State::Idle => {}
        State::Recording { started_at, .. } => {
            print!("\r\x1b[2KRecording... {:>5.1}s", started_at.elapsed().as_secs_f64());
        }
        State::Looping { play_pos, loop_frames, loop_buf, .. } => {
            let fraction = play_pos.load(Ordering::Relaxed) as f64 / *loop_frames as f64;
            let bar_line = format!("Loop:    {} {:>3.0}%", render_bar(fraction, BAR_WIDTH, false), fraction * 100.0);
            let rows = render_waveform_rows(&loop_buf.lock().unwrap(), *loop_frames, BAR_WIDTH);
            print_wave_and_bar(&rows, &bar_line);
        }
        State::Overdubbing { play_pos, loop_frames, pre_overdub, .. } => {
            let fraction = play_pos.load(Ordering::Relaxed) as f64 / *loop_frames as f64;
            let bar_line = format!(
                "Overdub: {} {:>3.0}% {RED}[REC]{RESET}",
                render_bar(fraction, BAR_WIDTH, true),
                fraction * 100.0
            );
            let rows = render_waveform_rows(pre_overdub, *loop_frames, BAR_WIDTH);
            print_wave_and_bar(&rows, &bar_line);
        }
    }
    io::stdout().flush().ok();
}

/// Prints `WAVE_HEIGHT` waveform rows followed by the bar line beneath them,
/// then moves the cursor back up to the first waveform row.
fn print_wave_and_bar(rows: &[String; WAVE_HEIGHT], bar_line: &str) {
    print!("\r\x1b[2K         [{}]", rows[0]);
    for row in &rows[1..] {
        print!("\r\n\x1b[2K         [{row}]");
    }
    print!("\r\n\x1b[2K{bar_line}");
    print!("\r\x1b[{WAVE_HEIGHT}A");
}

fn advance(state: State, audio: &Audio, latency_frames: isize) -> Result<State, Box<dyn std::error::Error>> {
    match state {
        State::Idle => {
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let in_stream = build_input_stream(
                &audio.input_device,
                &audio.input_config,
                audio.device_channels,
                audio.used_channels,
                buffer.clone(),
            )?;
            in_stream.play()?;
            print_status("Recording... press SPACE to stop and start looping.");
            Ok(State::Recording { in_stream, buffer, started_at: Instant::now() })
        }
        State::Recording { in_stream, buffer, .. } => {
            drop(in_stream); // stop capturing
            let loop_frames = buffer.lock().unwrap().len() / audio.used_channels.max(1) as usize;
            if loop_frames == 0 {
                print_status("Recorded nothing. Back to idle -- press SPACE to record.");
                return Ok(State::Idle);
            }
            let play_pos = Arc::new(AtomicUsize::new(0));
            let out_stream = build_output_stream(
                &audio.output_device,
                &audio.output_config,
                buffer.clone(),
                play_pos.clone(),
                loop_frames,
                audio.used_channels,
            )?;
            out_stream.play()?;
            print_status("Looping! SPACE to overdub, R to reset.");
            Ok(State::Looping {
                out_stream,
                loop_buf: buffer,
                play_pos,
                loop_frames,
            })
        }
        State::Looping {
            out_stream,
            loop_buf,
            play_pos,
            loop_frames,
        } => {
            // Audio captured "now" corresponds to what was heard roughly
            // `latency_frames` ago (mic + speaker round-trip delay), so seed
            // the overdub write cursor that far behind the current playback
            // position instead of exactly on it.
            let start_frame = (play_pos.load(Ordering::Relaxed) as isize - latency_frames)
                .rem_euclid(loop_frames as isize) as usize;
            let pre_overdub = loop_buf.lock().unwrap().clone();
            let in_stream = build_overdub_input_stream(
                &audio.input_device,
                &audio.input_config,
                loop_buf.clone(),
                start_frame,
                loop_frames,
                audio.device_channels,
                audio.used_channels,
            )?;
            in_stream.play()?;
            print_status("Overdubbing onto the loop... SPACE to stop overdubbing, R to reset.");
            Ok(State::Overdubbing {
                out_stream,
                in_stream,
                loop_buf,
                play_pos,
                loop_frames,
                pre_overdub,
            })
        }
        State::Overdubbing {
            out_stream,
            in_stream,
            loop_buf,
            play_pos,
            loop_frames,
            pre_overdub: _,
        } => {
            drop(in_stream); // stop capturing/mixing, keep the loop playing
            print_status("Looping! SPACE to overdub, R to reset.");
            Ok(State::Looping {
                out_stream,
                loop_buf,
                play_pos,
                loop_frames,
            })
        }
    }
}

/// Handles Backspace: aborts a take-in-progress and discards it, rather than
/// keeping it like SPACE would. Recording -> Idle (the capture buffer is
/// simply dropped); Overdubbing -> Looping with `loop_buf` restored to its
/// pre-overdub snapshot, undoing whatever was mixed in so far. Any other
/// state is left untouched.
fn cancel(state: State) -> State {
    match state {
        State::Recording { .. } => {
            print_status("Recording cancelled. Press SPACE to record a new loop.");
            State::Idle
        }
        State::Overdubbing {
            out_stream,
            in_stream,
            loop_buf,
            play_pos,
            loop_frames,
            pre_overdub,
        } => {
            drop(in_stream); // stop capturing/mixing
            *loop_buf.lock().unwrap() = pre_overdub;
            print_status("Overdub cancelled. Looping! SPACE to overdub, R to reset.");
            State::Looping { out_stream, loop_buf, play_pos, loop_frames }
        }
        other => other,
    }
}

/// Prints a one-off status line, clearing whatever waveform/bar lines the
/// previous `render_progress` may have left below it, and leaves the cursor
/// on the (now blank) line right below the message -- the block's home row
/// -- so the next `render_progress` call draws there instead of clobbering
/// the message itself.
fn print_status(msg: &str) {
    print!("\r\x1b[2K{msg}\r\n");
    for _ in 0..MAX_PROGRESS_LINES {
        print!("\x1b[2K\r\n");
    }
    print!("\r\x1b[{MAX_PROGRESS_LINES}A");
    io::stdout().flush().ok();
}
