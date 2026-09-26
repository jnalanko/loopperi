mod app;
mod engine;

use eframe::egui;

use engine::{Engine, LoopSpec};

fn parse_args() -> Result<LoopSpec, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [tempo, time_signature, measures] = args.as_slice() else {
        return Err("usage: loopperi <tempo-bpm> <time-signature> <measures>   (e.g. loopperi 120 4/4 4)".into());
    };

    let tempo_bpm: f64 = tempo.parse().map_err(|_| format!("invalid tempo: {tempo}"))?;
    if !tempo_bpm.is_finite() || tempo_bpm <= 0.0 {
        return Err(format!("tempo must be positive: {tempo}"));
    }

    let (beats, unit) = time_signature
        .split_once('/')
        .ok_or_else(|| format!("invalid time signature: {time_signature} (expected e.g. 4/4)"))?;
    let beats_per_measure: usize = beats.parse().map_err(|_| format!("invalid time signature: {time_signature}"))?;
    let beat_unit: u32 = unit.parse().map_err(|_| format!("invalid time signature: {time_signature}"))?;
    if beats_per_measure == 0 || beat_unit == 0 {
        return Err(format!("time signature parts must be positive: {time_signature}"));
    }

    let measures: usize = measures.parse().map_err(|_| format!("invalid measure count: {measures}"))?;
    if measures == 0 {
        return Err("measure count must be at least 1".into());
    }

    Ok(LoopSpec { tempo_bpm, beats_per_measure, beat_unit, measures })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let spec = match parse_args() {
        Ok(spec) => spec,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let engine = Engine::new(&spec)?;

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("loopperi")
            .with_inner_size([900.0, 420.0])
            .with_min_inner_size([560.0, 300.0]),
        ..Default::default()
    };
    eframe::run_native("loopperi", options, Box::new(|_cc| Ok(Box::new(app::LoopperiApp::new(engine, spec)))))?;
    Ok(())
}
