# cab-ir-lab — Design

**Status:** Proposed · **Created:** 2026-06-10 · **Constraint:** public data only (no capture rig) · **Test UI:** Gradio in-repo; standalone UI project later.

## 0. Background: what "cabinet characteristic" is, and how it's captured

A speaker cabinet + microphone at moderate volume is approximately a **linear time-invariant** system, so its entire character is one impulse response: `output = input ∗ IR`. That's why a 20–200 ms mono `.wav` can replace a mic'd 4×12.

How people capture it (we won't, but the model must respect this physics):

1. Play an **exponential sine sweep** (Farina method, 20 Hz–20 kHz, 20–60 s) through a **flat solid-state power amp** into the cab.
2. Record the mic; **deconvolve** with the inverse sweep → raw IR (ESS pushes harmonic distortion into negative time, so it separates nonlinearity from the linear IR).
3. Trim pre-delay, window the tail, normalize, export.

What changes with mic position — i.e., what our model must learn:

- **Distance** → proximity effect (cardioid bass boost up close), early-reflection comb filtering further out, level.
- **Lateral offset (cap → cone edge)** → the dominant timbre control: dust-cap = bright/harsh, edge = darker; smooth but nonlinear transition with sharp notch movement.
- **Off-axis angle** → HF rolloff.
- **Mic type** → its own transfer function (SM57 presence bump vs ribbon rolloff), partially separable from position.

**Known limitation:** an IR cannot represent **speaker breakup** (drive-level-dependent nonlinearity). All IR products share this limit. Out of scope; the workbench's NAM/whitebox lanes own nonlinearity.

## 1. Data prep (Workstream 1)

No public *research* dataset has dense cab IR position grids (room-IR datasets like MIRACLE prove the method but aren't cabs). The practical sources are **free commercial/community packs whose filenames encode labels**:

| Pack | Contents | Labels in filenames | License note |
|---|---|---|---|
| **Redwirez Marshall 1960A free pack** | G12M-25 Greenbacks, **17 mics**, multiple positions (cap/cap-edge/cone…) × distances (0–12″+) | mic, position name, distance | free w/ signup; check redistribution terms |
| **God's Cab (Wilkinson Audio) v1.4** | Mesa OS 4×12, 700+ IRs, 6 mics (57/SM7B/C414/U87/MD421/NT5) | mic, distance from grill, axis | free; check terms |
| **OwnHammer free packs** (Cheapie, Rock-Box, 112) | several cabs, single-mic captures + mixes | mic, position code | free; terms per pack |
| **TONE3000 API** (`tone3000.com/api`) | community IRs at scale | loose/inconsistent; parse descriptions | API ToS |
| (stretch) purchased packs: OwnHammer/York/Celestion full | dense grids | rich | **train-internal only, never redistribute raw** |

Design decisions:

- **License posture:** raw IRs are *training inputs* kept out of git (`data/raw/` gitignored); we ship model weights + generated IRs. Flag any pack whose terms forbid derivative works before training on it; prefer Redwirez + God's Cab as the core (both widely used for this purpose) and verify terms in W1.
- **Label schema** (`data/parsed/labels.parquet`): `pack, cab, speaker, mic, distance_mm, offset_mm (cap=0), angle_deg, axis (on/off), sr, n_samples, file`. Per-pack parser in `data/parsers/<pack>.py`; unparseable files quarantined, not guessed.
- **Normalization pipeline:** resample to 48 kHz mono → align onset (peak/energy threshold) → trim to 4096 samples (85 ms; covers cab + close room) → loudness-normalize (unit energy) → store float32.
- **Phase strategy:** cab IRs are near minimum-phase; convert targets to **min-phase** (cepstral method) so the model only learns magnitude + the min-phase reconstruction is deterministic. Keeps training stable and avoids phase-wrap losses. Keep an `is_minphase` flag to A/B against raw-phase targets.
- **Coverage matrix report:** script emits per-(cab, mic) position-grid coverage so we know where interpolation is data-supported vs extrapolation.
- **Augmentation (if grids are sparse):** Neural-Cab-style **BEM/physical simulation** priors, or simple physics augmentation (delay/level scaling vs distance, measured off-axis filter families) — Phase 2 only if needed.

## 2. Model (Workstream 2)

**Task:** `f(conditioning) → IR[4096] @ 48 kHz`. Conditioning: `mic` (learned embedding, ~8-dim), `distance` (log-scaled scalar), `offset`, `angle`, `cab/speaker` (embedding).

**Baseline A — conditional spectral decoder (recommended start; ≈ the IEEE Marshall-1960A paper):**
- MLP/1D-conv decoder predicting **log-magnitude spectrum** (2049 bins), then min-phase reconstruction → IR.
- Loss: multi-resolution STFT + log-mag L1 (weighted toward 100 Hz–8 kHz where the ear judges cabs).
- Tiny (<1 M params), trains on a laptop, smooth interpolation across position comes from the continuous conditioning inputs.

**Baseline B — latent interpolation (Neural Cab style):** VAE over IRs + adversarial loss; condition the latent on position. More expressive, slower to get right. Phase 2.

**Stretch — implicit field:** SIREN-style `f(x, y, angle, freq) → magnitude` — an "IR field" per cab; elegant for continuous mic-drag UI. Phase 2 experiment.

**Stretch — factorized mic:** `cab IR (flat reference mic) ∗ mic IR (e.g. MicIRP library) ≈ cab+mic IR`. Model learns only `(cab, position)`; mic coloration becomes swappable post-convolution. Data-efficient and lets users apply any mic, but approximate — proximity effect and off-axis rolloff are position-dependent and not captured by a single static mic IR. Phase 2: A/B against the joint model before adopting.

**Eval:** held-out positions per (cab, mic) — report log-spectral distance vs nearest-neighbor-IR baseline (the model must beat "just pick the closest captured IR" to justify itself); blind A/B audition page in the Gradio app.

**Stack:** PyTorch + torchaudio, `uv` for env, config via plain dataclasses/YAML, checkpoints to `runs/`.

## 3. Inference → `.wav` (Workstream 3)

`cabir/infer.py`:
- `--ckpt --mic sm57 --distance 25 --offset 30 --angle 0 --out exports/sm57_25mm_capedge.wav`
- Pipeline: condition → magnitude → min-phase IR → peak-normalize to −0.3 dBFS → 24-bit 48 kHz mono WAV (+ optional 44.1 resample).
- Batch mode: sweep a parameter (e.g., 11 distances) → IR pack folder with the same filename label grammar we parse — round-trip-able.
- Target consumers: any IR loader; audio-engine's cab lane (`ConvolverNode`) directly.

## 4. Test UI (Workstream 4) — Gradio, in-repo

`app/gradio_app.py`:
- **Position pad:** 2D click/drag pad = (offset, distance) over a speaker-cone illustration; sliders for angle; dropdowns for cab + mic.
- **Visuals:** generated IR waveform + magnitude response, overlaid with nearest captured IR for honesty about interpolation.
- **Audition:** upload (or bundle) a DI guitar clip → convolve (`scipy.signal.fftconvolve`) → A/B player: model IR vs nearest real IR vs dry. Optional NAM-processed DI as source so it sounds like a real rig.
- **Mic-move sweep:** render N positions along a drag path into a crossfaded clip to "hear the mic move."
- Later: the standalone UI project replaces this with real-time WebAudio (the workbench already ships ConvolverNode + worklet infra; model is small enough for onnxruntime-web/WASM, ~ms-level IR generation on drag).

## 5. Milestones

| | Deliverable | Acceptance |
|---|---|---|
| **M1** | Data: 2 packs parsed (Redwirez + God's Cab), labels.parquet, coverage report | ≥1k labeled IRs, license check documented |
| **M2** | Baseline A trained on 1 cab/3 mics | beats nearest-neighbor on held-out positions (log-spectral distance) |
| **M3** | `infer.py` exports valid `.wav` IRs | loads in audio-engine cab lane + any IR loader |
| **M4** | Gradio app with position pad + A/B audition | "hear the mic move" demo |
| **M5** | Scale: all parsed cabs/mics, TONE3000 ingestion, Phase-2 model experiments | multi-cab model card |

## 6. Risks

- **Sparse grids** → model interpolates between few anchors; mitigation: physics augmentation, report extrapolation honestly in UI (M1 coverage report drives this).
- **Label noise across packs** (distance conventions differ: grill vs cap) → per-pack offset calibration; never mix packs in one (cab, mic) grid without alignment.
- **License ambiguity** → W1 gate: written terms check per pack before training; weights-only distribution.
- **Min-phase assumption** breaks for far/room mics → keep raw-phase A/B path (`is_minphase` flag).
