use std::io::{self, Write as _};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal;

enum State {
    Idle,
    Recording {
        in_stream: Stream,
        buffer: Arc<Mutex<Vec<f32>>>,
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
    let run = |args: &[&str]| -> Option<String> {
        String::from_utf8(std::process::Command::new("pactl").args(args).output().ok()?.stdout).ok()
    };
    let default_source = run(&["get-default-source"])?.trim().to_string();
    let sources = run(&["list", "sources"])?;

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
    println!();

    terminal::enable_raw_mode()?;
    let result = run(input_device, input_config, output_device, output_config, used_channels);
    terminal::disable_raw_mode()?;
    result
}

fn run(
    input_device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    output_device: cpal::Device,
    output_config: cpal::SupportedStreamConfig,
    used_channels: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let device_channels = input_config.channels();
    let mut state = State::Idle;

    loop {
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char(' ') => {
                        state = advance(
                            state,
                            &input_device,
                            &input_config,
                            &output_device,
                            &output_config,
                            device_channels,
                            used_channels,
                        )?;
                    }
                    KeyCode::Char('r') | KeyCode::Char('R') => {
                        state = State::Idle;
                        print_status("Reset. Press SPACE to record a new loop.");
                    }
                    KeyCode::Esc => break,
                    KeyCode::Char('c') if key.modifiers.contains(event::KeyModifiers::CONTROL) => break,
                    _ => {}
                },
                _ => {}
            }
        }
    }

    Ok(())
}

fn advance(
    state: State,
    input_device: &cpal::Device,
    input_config: &cpal::SupportedStreamConfig,
    output_device: &cpal::Device,
    output_config: &cpal::SupportedStreamConfig,
    device_channels: u16,
    used_channels: u16,
) -> Result<State, Box<dyn std::error::Error>> {
    match state {
        State::Idle => {
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let in_stream = build_input_stream(
                input_device,
                input_config,
                device_channels,
                used_channels,
                buffer.clone(),
            )?;
            in_stream.play()?;
            print_status("Recording... press SPACE to stop and start looping.");
            Ok(State::Recording { in_stream, buffer })
        }
        State::Recording { in_stream, buffer } => {
            drop(in_stream); // stop capturing
            let loop_frames = buffer.lock().unwrap().len() / used_channels.max(1) as usize;
            if loop_frames == 0 {
                print_status("Recorded nothing. Back to idle -- press SPACE to record.");
                return Ok(State::Idle);
            }
            let play_pos = Arc::new(AtomicUsize::new(0));
            let out_stream = build_output_stream(
                output_device,
                output_config,
                buffer.clone(),
                play_pos.clone(),
                loop_frames,
                used_channels,
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
            let in_stream = build_overdub_input_stream(
                input_device,
                input_config,
                loop_buf.clone(),
                play_pos.load(Ordering::Relaxed),
                loop_frames,
                device_channels,
                used_channels,
            )?;
            in_stream.play()?;
            print_status("Overdubbing onto the loop... SPACE to stop overdubbing, R to reset.");
            Ok(State::Overdubbing {
                out_stream,
                in_stream,
                loop_buf,
                play_pos,
                loop_frames,
            })
        }
        State::Overdubbing {
            out_stream,
            in_stream,
            loop_buf,
            play_pos,
            loop_frames,
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

fn print_status(msg: &str) {
    print!("\r\x1b[2K{msg}\r\n");
    io::stdout().flush().ok();
}
