//! The loopperi window: transport buttons, a waveform of the loop with its
//! beat grid and playhead, and keyboard shortcuts for everything.

use eframe::egui::{self, Align2, Color32, FontId, Key, Rect, Sense, Stroke, StrokeKind};

use crate::engine::{Engine, LoopSpec};

/// How far one Left/Right press moves overdub latency compensation.
const LATENCY_STEP_MS: f64 = 5.0;
/// Horizontal size of one waveform column, in points.
const COLUMN_WIDTH: f32 = 2.0;
const RECORD_RED: Color32 = Color32::from_rgb(220, 50, 50);

/// Everything the user can ask the looper to do, from a key or a button.
#[derive(Clone, Copy)]
enum Action {
    ToggleOverdub,
    CancelOverdub,
    Clear,
    ToggleMetronome,
    NudgeLatency(f64),
    Quit,
}

impl Action {
    fn for_key(key: Key) -> Option<Action> {
        Some(match key {
            Key::Space => Action::ToggleOverdub,
            Key::Backspace => Action::CancelOverdub,
            Key::R => Action::Clear,
            Key::M => Action::ToggleMetronome,
            Key::ArrowLeft => Action::NudgeLatency(-LATENCY_STEP_MS),
            Key::ArrowRight => Action::NudgeLatency(LATENCY_STEP_MS),
            Key::Escape => Action::Quit,
            _ => return None,
        })
    }
}

pub struct LoopperiApp {
    engine: Engine,
    spec: LoopSpec,
    status: String,
}

impl LoopperiApp {
    pub fn new(engine: Engine, spec: LoopSpec) -> Self {
        LoopperiApp { engine, spec, status: "Empty loop running. Space to overdub, M for metronome.".into() }
    }

    /// Takes the looper's shortcut keys out of this frame's input and returns
    /// what they ask for. Only the initial key-down counts -- auto-repeats of a
    /// held key are dropped, so a long press toggles exactly once. Removing
    /// the events also keeps a focused button from treating Space as a click
    /// on top of the shortcut.
    fn take_key_actions(ctx: &egui::Context) -> Vec<Action> {
        ctx.input_mut(|input| {
            let mut actions = Vec::new();
            input.events.retain(|event| match event {
                egui::Event::Key { key, pressed, repeat, modifiers, .. }
                    if !modifiers.ctrl && !modifiers.alt && !modifiers.command =>
                {
                    let Some(action) = Action::for_key(*key) else { return true };
                    if *pressed && !*repeat {
                        actions.push(action);
                    }
                    false
                }
                _ => true,
            });
            actions
        })
    }

    fn apply(&mut self, action: Action, ctx: &egui::Context) {
        match action {
            Action::ToggleOverdub => {
                let starting = !self.engine.is_overdubbing();
                let latency_ms = self.engine.latency_ms();
                self.status = match self.engine.toggle_overdub() {
                    Ok(()) if starting => format!("Overdubbing with {latency_ms:.0} ms compensation..."),
                    Ok(()) => "Looping! Space to overdub, R to clear.".into(),
                    Err(err) => format!("Couldn't start overdubbing: {err}"),
                };
            }
            Action::CancelOverdub => {
                if self.engine.cancel_overdub() {
                    self.status = "Overdub cancelled. Looping!".into();
                }
            }
            Action::Clear => {
                self.engine.clear();
                self.status = "Loop cleared.".into();
            }
            Action::ToggleMetronome => {
                let on = self.engine.toggle_metronome();
                self.status = if on { "Metronome on." } else { "Metronome off." }.into();
            }
            Action::NudgeLatency(delta_ms) => {
                self.engine.nudge_latency(delta_ms);
                self.status = format!("Overdub timing: {:.0} ms", self.engine.latency_ms());
            }
            Action::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        }
    }

    fn header(&self, ui: &mut egui::Ui) {
        let spec = &self.spec;
        let info = self.engine.info();
        ui.horizontal(|ui| {
            ui.heading("loopperi");
            ui.label(format!(
                "{} BPM · {}/{} · {} measures ({} beats, {:.2} s)",
                spec.tempo_bpm, spec.beats_per_measure, spec.beat_unit, spec.measures, info.beats, info.loop_seconds
            ));
        });
        ui.weak(format!(
            "In: {} ({} Hz, {} of {} ch)    Out: {} ({} Hz, {} ch)",
            info.input_device,
            info.input_rate,
            info.used_channels,
            info.input_device_channels,
            info.output_device,
            info.output_rate,
            info.output_channels
        ));
        if info.input_rate != info.output_rate {
            ui.colored_label(
                ui.visuals().warn_fg_color,
                "Input/output sample rates differ, playback speed/pitch may be off (no resampling is done).",
            );
        }
    }

    /// Transport buttons; returns the action of whichever was clicked.
    fn controls(&self, ui: &mut egui::Ui) -> Option<Action> {
        let recording = self.engine.is_overdubbing();
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            let overdub_label = if recording { "Stop overdub  [Space]" } else { "Overdub  [Space]" };
            let overdub = egui::Button::new(overdub_label).selected(recording);
            if ui.add(overdub).clicked() {
                action = Some(Action::ToggleOverdub);
            }
            if ui.add_enabled(recording, egui::Button::new("Cancel  [Backspace]")).clicked() {
                action = Some(Action::CancelOverdub);
            }
            if ui.button("Clear  [R]").clicked() {
                action = Some(Action::Clear);
            }
            let metronome = egui::Button::new("Metronome  [M]").selected(self.engine.metronome_on());
            if ui.add(metronome).clicked() {
                action = Some(Action::ToggleMetronome);
            }
            ui.separator();
            ui.label("Overdub timing");
            if ui.button("−  [←]").clicked() {
                action = Some(Action::NudgeLatency(-LATENCY_STEP_MS));
            }
            ui.monospace(format!("{:>4.0} ms", self.engine.latency_ms()));
            if ui.button("+  [→]").clicked() {
                action = Some(Action::NudgeLatency(LATENCY_STEP_MS));
            }
        });
        action
    }

    /// The loop as a mirrored peak waveform over its beat grid, with the
    /// playhead on top. During an overdub the frame and playhead turn red and
    /// the waveform still shows the loop as it was before this take.
    fn waveform(&mut self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), Sense::hover());
        let rect = response.rect;
        let visuals = ui.visuals();
        let recording = self.engine.is_overdubbing();
        let accent = if recording { RECORD_RED } else { visuals.selection.bg_fill };

        painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);

        let beats = self.engine.info().beats;
        let grid = visuals.weak_text_color();
        for beat in 0..beats {
            let x = rect.left() + rect.width() * beat as f32 / beats as f32;
            let downbeat = beat % self.spec.beats_per_measure == 0;
            let stroke = if downbeat { Stroke::new(1.0, grid) } else { Stroke::new(1.0, grid.gamma_multiply(0.35)) };
            painter.vline(x, rect.y_range(), stroke);
            if downbeat {
                let measure = beat / self.spec.beats_per_measure + 1;
                painter.text(
                    egui::pos2(x + 4.0, rect.top() + 2.0),
                    Align2::LEFT_TOP,
                    measure.to_string(),
                    FontId::proportional(11.0),
                    grid,
                );
            }
        }

        let columns = ((rect.width() / COLUMN_WIDTH) as usize).max(1);
        let wave_color = visuals.strong_text_color().gamma_multiply(0.8);
        let half_height = rect.height() / 2.0 - 4.0;
        let peaks = self.engine.waveform_peaks(columns);
        for (col, &peak) in peaks.iter().enumerate() {
            if peak <= 0.0 {
                continue;
            }
            let x = rect.left() + col as f32 * COLUMN_WIDTH;
            let h = (peak * half_height).max(0.5);
            let bar = Rect::from_min_max(
                egui::pos2(x, rect.center().y - h),
                egui::pos2(x + COLUMN_WIDTH * 0.75, rect.center().y + h),
            );
            painter.rect_filled(bar, 0.0, wave_color);
        }

        let position = self.engine.position() as f32;
        let x = rect.left() + rect.width() * position;
        painter.vline(x, rect.y_range(), Stroke::new(2.0, accent));

        let border = if recording { Stroke::new(2.0, RECORD_RED) } else { visuals.widgets.noninteractive.bg_stroke };
        painter.rect_stroke(rect, 4.0, border, StrokeKind::Inside);
    }

    /// Where playback is, as "measure · beat" plus a REC badge while overdubbing.
    fn position_line(&self, ui: &mut egui::Ui) {
        let beats = self.engine.info().beats;
        let beat = ((self.engine.position() * beats as f64) as usize).min(beats - 1);
        let per_measure = self.spec.beats_per_measure;
        ui.horizontal(|ui| {
            ui.monospace(format!(
                "Measure {} / {}   Beat {} / {}",
                beat / per_measure + 1,
                self.spec.measures,
                beat % per_measure + 1,
                per_measure
            ));
            if self.engine.is_overdubbing() {
                ui.label(egui::RichText::new("● REC").strong().color(RECORD_RED));
            }
        });
    }
}

impl eframe::App for LoopperiApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        for action in Self::take_key_actions(&ctx) {
            self.apply(action, &ctx);
        }

        egui::Panel::top("header").show(ui, |ui| {
            ui.add_space(4.0);
            self.header(ui);
            ui.add_space(4.0);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            ui.add_space(2.0);
            ui.label(&self.status);
            ui.add_space(2.0);
        });
        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(action) = self.controls(ui) {
                self.apply(action, &ctx);
            }
            ui.add_space(6.0);
            self.position_line(ui);
            ui.add_space(4.0);
            self.waveform(ui);
        });

        // The playhead moves continuously, so keep redrawing.
        ctx.request_repaint();
    }
}

