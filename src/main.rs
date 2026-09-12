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

fn build_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    buffer: Arc<Mutex<Vec<f32>>>,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream_config = config.config();

    match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                buffer.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(data.iter().map(|s| s.to_sample::<f32>()));
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let mut buf = buffer.lock().unwrap();
                buf.extend(data.iter().map(|s| s.to_sample::<f32>()));
            },
            err_fn,
            None,
        ),
        sample_format => panic!("unsupported input sample format: {sample_format}"),
    }
}

/// Mixes freshly captured audio into the loop buffer, starting at whatever
/// frame the playback stream currently sits on, wrapping at `loop_frames`.
fn mix_into_loop(
    loop_buf: &Mutex<Vec<f32>>,
    play_pos: &AtomicUsize,
    loop_frames: usize,
    channels: usize,
    data: impl Iterator<Item = f32>,
) {
    if loop_frames == 0 {
        return;
    }
    let mut buf = loop_buf.lock().unwrap();
    let start_frame = play_pos.load(Ordering::Relaxed);
    let mut frame_offset = 0usize;
    let mut ch = 0usize;
    for sample in data {
        let frame = (start_frame + frame_offset) % loop_frames;
        let idx = frame * channels + ch;
        if let Some(existing) = buf.get_mut(idx) {
            *existing = (*existing + sample).clamp(-1.0, 1.0);
        }
        ch += 1;
        if ch == channels {
            ch = 0;
            frame_offset += 1;
        }
    }
}

fn build_overdub_input_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    loop_buf: Arc<Mutex<Vec<f32>>>,
    play_pos: Arc<AtomicUsize>,
    loop_frames: usize,
    channels: u16,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("input stream error: {err}");
    let stream_config = config.config();
    let channels = channels as usize;

    match config.sample_format() {
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                mix_into_loop(&loop_buf, &play_pos, loop_frames, channels, data.iter().copied());
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                mix_into_loop(
                    &loop_buf,
                    &play_pos,
                    loop_frames,
                    channels,
                    data.iter().map(|s| s.to_sample::<f32>()),
                );
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                mix_into_loop(
                    &loop_buf,
                    &play_pos,
                    loop_frames,
                    channels,
                    data.iter().map(|s| s.to_sample::<f32>()),
                );
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = cpal::default_host();

    let input_device = host
        .default_input_device()
        .ok_or("no input audio device available")?;
    let output_device = host
        .default_output_device()
        .ok_or("no output audio device available")?;

    let input_config = input_device.default_input_config()?;
    let output_config = output_device.default_output_config()?;

    println!("Input device:  {}", input_device.name()?);
    println!("Output device: {}", output_device.name()?);
    println!(
        "Recording at {} Hz / {} ch, playing back at {} Hz / {} ch",
        input_config.sample_rate().0,
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
    let result = run(input_device, input_config, output_device, output_config);
    terminal::disable_raw_mode()?;
    result
}

fn run(
    input_device: cpal::Device,
    input_config: cpal::SupportedStreamConfig,
    output_device: cpal::Device,
    output_config: cpal::SupportedStreamConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let recorded_channels = input_config.channels();
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
                            recorded_channels,
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
    recorded_channels: u16,
) -> Result<State, Box<dyn std::error::Error>> {
    match state {
        State::Idle => {
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let in_stream = build_input_stream(input_device, input_config, buffer.clone())?;
            in_stream.play()?;
            print_status("Recording... press SPACE to stop and start looping.");
            Ok(State::Recording { in_stream, buffer })
        }
        State::Recording { in_stream, buffer } => {
            drop(in_stream); // stop capturing
            let loop_frames = buffer.lock().unwrap().len() / recorded_channels.max(1) as usize;
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
                recorded_channels,
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
                play_pos.clone(),
                loop_frames,
                recorded_channels,
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
