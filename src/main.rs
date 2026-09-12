use std::io::{self, Write as _};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, Stream};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal;

enum State {
    Idle,
    Recording {
        stream: Stream,
        buffer: Arc<Mutex<Vec<f32>>>,
        channels: u16,
    },
    Looping {
        stream: Stream,
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

fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    recorded: Arc<Vec<f32>>,
    recorded_channels: u16,
) -> Result<Stream, cpal::BuildStreamError> {
    let err_fn = |err| eprintln!("output stream error: {err}");
    let stream_config = config.config();
    let out_channels = stream_config.channels as usize;
    let rec_channels = recorded_channels as usize;
    let total_frames = recorded.len() / rec_channels;
    let mut frame: usize = 0;

    let mut next_sample = move |out_channel: usize| -> f32 {
        if total_frames == 0 {
            return 0.0;
        }
        let src_channel = if rec_channels == 1 { 0 } else { out_channel % rec_channels };
        let value = recorded[frame * rec_channels + src_channel];
        if out_channel == out_channels - 1 {
            frame = (frame + 1) % total_frames;
        }
        value
    };

    match config.sample_format() {
        SampleFormat::F32 => device.build_output_stream(
            &stream_config,
            move |data: &mut [f32], _| {
                for (i, sample) in data.iter_mut().enumerate() {
                    *sample = next_sample(i % out_channels);
                }
            },
            err_fn,
            None,
        ),
        SampleFormat::I16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [i16], _| {
                for (i, sample) in data.iter_mut().enumerate() {
                    *sample = next_sample(i % out_channels).to_sample::<i16>();
                }
            },
            err_fn,
            None,
        ),
        SampleFormat::U16 => device.build_output_stream(
            &stream_config,
            move |data: &mut [u16], _| {
                for (i, sample) in data.iter_mut().enumerate() {
                    *sample = next_sample(i % out_channels).to_sample::<u16>();
                }
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
    println!("SPACE: idle -> record -> loop -> idle    |    Esc/Ctrl+C: quit");
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
    let mut state = State::Idle;

    loop {
        if event::poll(Duration::from_millis(100))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char(' ') => {
                        state = advance(state, &input_device, &input_config, &output_device, &output_config)?;
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
) -> Result<State, Box<dyn std::error::Error>> {
    match state {
        State::Idle => {
            let buffer = Arc::new(Mutex::new(Vec::new()));
            let stream = build_input_stream(input_device, input_config, buffer.clone())?;
            stream.play()?;
            print_status("Recording... press SPACE to stop and start looping.");
            Ok(State::Recording {
                stream,
                buffer,
                channels: input_config.channels(),
            })
        }
        State::Recording { stream, buffer, channels } => {
            drop(stream); // stop capturing; buffer is now only referenced here
            let recorded = Arc::new(
                Arc::try_unwrap(buffer)
                    .expect("input stream dropped, no other references remain")
                    .into_inner()
                    .unwrap(),
            );
            let frames = recorded.len() / channels.max(1) as usize;
            if frames == 0 {
                print_status("Recorded nothing. Back to idle -- press SPACE to record.");
                return Ok(State::Idle);
            }
            let out_stream = build_output_stream(output_device, output_config, recorded, channels)?;
            out_stream.play()?;
            print_status("Looping! press SPACE to stop and go back to idle.");
            Ok(State::Looping { stream: out_stream })
        }
        State::Looping { stream } => {
            drop(stream); // stop playback
            print_status("Stopped. Press SPACE to record a new loop.");
            Ok(State::Idle)
        }
    }
}

fn print_status(msg: &str) {
    print!("\r\x1b[2K{msg}\r\n");
    io::stdout().flush().ok();
}
