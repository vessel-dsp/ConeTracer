# cab-ir-lab — Design

**Status:** M1 ✅ M2 ✅ M3 ✅ M4 prototype ✅ · **Created:** 2026-06-10 · **Constraint:** public data only (no capture rig) · **Test UI:** Gradio in-repo; standalone UI project later.

## 0. Background: what "cabinet characteristic" is, and how it's captured

A speaker cabinet + microphone at moderate volume is approximately a **linear time-invariant** system, so its entire character is one impulse response: `output = input ∗ IR`. That's why a 20–200 ms mono `.wav` can replace a mic'd 4×12.

How people capture it (we won't, but the model must respect this physics):

1. Play an **exponential sine sweep** (Farina method, 20 Hz–20 kHz, 20–60 s) through a **flat solid-state power amp** into the cab.
2. Record the mic; **deconvolve** with the inverse sweep → raw IR (ESS pushes harmonic distortion into negative time, so it separates nonlinearity from the linear IR).
3. Trim pre-delay, window the tail, normalize, export.

What changes with mic position — i.e., what our model must learn:

- **Distance** → proximity effect (cardioid bass boost up close), early-reflection comb filtering further out, level.
- **Lateral offset (cap → cone edge)** → the dominant timbre control: dust-cap = bright/harsh, edge = darker; smooth but nonlinear transition with sharp notch movement.
- **Off-axis angle** → HF rolloff. Current God's Cab close-mic data is all `angle_deg = 0`, so production angle behavior needs true off-axis captures or another pack with trustworthy angle labels.
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
- **Label schema** (`data/parsed/labels.parquet`): `pack, cab, speaker, mic, distance_mm, distance_ref, offset_mm (cap=0; NaN for distant/room), angle_deg, axis (on/off), capture_type (close/distant/room), ts (bool), presence, sr, n_samples, src_sr, is_minphase, file`. Per-pack parser in `data/parsers/<pack>.py`; unparseable files quarantined, not guessed. Ingest is **lossless** — every documented label dimension (incl. pack-specific ones like God's Cab `presence`/`ts`) is recorded so the loader can filter or condition on it.
- **Normalization pipeline:** resample to 48 kHz mono → align onset (peak/energy threshold) → trim to 4096 samples (85 ms; covers cab + close room) → loudness-normalize (unit energy) → store float32.
- **Phase strategy:** cab IRs are near minimum-phase; convert targets to **min-phase** (cepstral method) so the model only learns magnitude + the min-phase reconstruction is deterministic. Keeps training stable and avoids phase-wrap losses. Keep an `is_minphase` flag to A/B against raw-phase targets.
- **Coverage matrix report:** script emits per-(cab, mic) position-grid coverage so we know where interpolation is data-supported vs extrapolation.
- **Augmentation (if grids are sparse):** Neural-Cab-style **BEM/physical simulation** priors, or simple physics augmentation (delay/level scaling vs distance, measured off-axis filter families) — Phase 2 only if needed.

**Implemented & running — M1 complete:** ingest pipeline self-tested (`python3 scripts/selftest_ingest.py`) and run on **12 packs → 1,159 labeled IRs** (967 close / 112 distant / 80 room; 20 mic types across 8 normalized cab labels).
- `cabir/labels.py` — schema, declarative `PackConfig` (+ `path_include`/`path_exclude`), generic token parser `parse_path`, `missing_fields` for quarantine hints, parquet IO, coverage report.
- `data/parsers/godscab.py` — **custom `parse()`** matching the manual grammar (grill/inch/foot distance, cap/edge/cone_near/cone_far, TS, presence, close/distant/room); ingests only the 48 kHz folder, skips 44.1/96 duplicates + Axe-FX `.syx` + Legacy. **659 IRs** (Mesa OS Rectifier 4×12 / V30, 6 mics, presence 1–5 × ts conditioning).
- `data/parsers/overdriven.py` — one parser for all 11 overdriven.fr SSP2-series packs (11 different cabs, up to 5 mics each). Grammar: `OD-{CAB}-{MIC}-P{OFFSET_MM}-{DIST_MM}[-SUFFIX].wav`; skips -UA610/-EQ/-CUT/-P1 variants. **500 IRs** licensed "free for musical/video creations, any commercial or non-commercial purpose."
- `data/parsers/redwirez_1960a.py` — declarative config, real 17-mic set (⚠ parser ready; verify position tokens against files when downloaded).
- `cabir/ingest.py` — driver: pack detection → path filter → parse (custom or generic) → normalize (mono/48k/onset/4096/window/unit-energy, `--minphase`) → `data/parsed/{labels.parquet, irs.npy, coverage.md, quarantine/}`.
- `cabir/dsp.py` — added `to_mono`, `resample_to`, `align_onset`, `fit_length`, `normalize_ir`.

**Current parsed artifact note:** the checked-in `data/parsed/` artifact was produced without `--minphase` (`is_minphase=False`), though the inference path still reconstructs generated IRs as minimum phase from predicted magnitude. Rerun ingest with `--minphase` before the next benchmark if we want the stored targets to match the phase strategy exactly.

## 2. Model (Workstream 2)

**Task:** `f(conditioning) → IR[4096] @ 48 kHz`. Conditioning: `mic` (learned embedding, ~8-dim), `distance` (log-scaled scalar), `offset`, `angle`, `cab/speaker` (embedding).

**Baseline A — conditional spectral decoder (≈ the IEEE Marshall-1960A paper) — DONE:**
- `cabir/model.py::SpectralDecoder` — learned mic + cab embeddings (dim=8) + 3 continuous scalars → 3-layer MLP → log-magnitude spectrum (2049 bins) → min-phase reconstruction → IR.
- Loss: frequency-weighted L1 on log-magnitude (2× weight for 100 Hz–8 kHz). Adam + cosine LR decay, dropout=0.1, weight_decay=1e-4.
- 628k params, trains in ~15 s on GPU. Checkpoint: `runs/real/best.pt`.
- **Result:** presence-conditioned hidden-256 checkpoint val LSD **2.55 dB** vs nearest-neighbor baseline **4.89 dB** (−48%) while training on all five presence values. On the `presence=3` captured grid it improves mean guitar-band IR error from **1.21 dB → 1.01 dB** vs the presence-filtered hidden-256 model. Checkpoint: `runs/real_pcond_h256/best.pt`.

**Baseline B — latent interpolation (Neural Cab style):** VAE over IRs + adversarial loss; condition the latent on position. More expressive, slower to get right. Phase 2.

**Stretch — implicit field:** SIREN-style `f(x, y, angle, freq) → magnitude` — an "IR field" per cab; elegant for continuous mic-drag UI. Phase 2 experiment.

**Stretch — factorized mic:** `cab IR (flat reference mic) ∗ mic IR (e.g. MicIRP library) ≈ cab+mic IR`. Model learns only `(cab, position)`; mic coloration becomes swappable post-convolution. Data-efficient and lets users apply any mic, but approximate — proximity effect and off-axis rolloff are position-dependent and not captured by a single static mic IR. Phase 2: A/B against the joint model before adopting.

**Eval:** held-out positions per (cab, mic) — report log-spectral distance vs nearest-neighbor-IR baseline (the model must beat "just pick the closest captured IR" to justify itself); blind A/B audition page in the Gradio app.

**Stack:** PyTorch 2.6 + CUDA, `uv` for env, checkpoints to `runs/`.

**Supporting files added (M2):**
- `cabir/dataset.py` — `IRDataset`: filters by cab/mic/ts/presence/capture_type; position-level train/val split (holds out entire `(distance_mm, offset_mm)` positions, not just presence variants); returns logmag target + conditioning dict.
- `cabir/train.py --real` — real-data training mode alongside the existing synthetic smoke pipeline.
- Training-time augmentation flags exist for conditioning jitter and edge weighting (`--distance-jitter-mm`, `--offset-jitter-mm`, `--angle-jitter-deg`, `--edge-weight`). Initial experiments show a tradeoff: edge weighting improves some cone-edge worst cases, but the plain hidden-256 checkpoint remains best on average.

## 3. Inference → `.wav` (Workstream 3) — **done**

`cabir/infer.py`:
- `--ckpt runs/real/best.pt --mic sm57 --distance 25 --offset 30 --angle 0 --out exports/sm57_25mm_capedge.wav`
- Pipeline: load `SpectralDecoder` from checkpoint → condition → log-magnitude → `minphase_from_magnitude` → peak-normalize to −0.3 dBFS → 24-bit 48 kHz mono WAV.
- Batch sweep mode remains next: iterate (e.g.) 11 offsets across the cone → export an IR pack folder with the parsed filename grammar (round-trip-able).
- Target consumers: any IR loader or WebAudio `ConvolverNode` host.
- Current implementation supports both smoke `.npz` checkpoints and real PyTorch `.pt` checkpoints.

## 4. Test UI (Workstream 4) — Gradio, in-repo

`app/gradio_app.py`:
- **Controls:** cab + mic dropdowns from the checkpoint vocabulary; sliders for distance, offset, and angle.
- **Tabs:** separate Normal render, Mic sweep, Model vs real, and Validation grid views so static export, moving-audition, and validation workflows stay uncluttered.
- **Visuals:** generated IR waveform + magnitude response, overlaid with nearest captured IR for honesty about interpolation.
- **Delta plots:** smoothed average spectrum delta and broad-band delta bars; red = boost, blue = reduction.
- **Validation:** generated IR vs nearest captured IR, RMS dB error over 100 Hz-6 kHz, and model-rendered vs nearest-real-rendered audio comparison.
- **Grid validation:** heatmap and worst-position table over all captured distance/offset cells for the selected mic.
- **Spectrogram comparison:** selected original audio vs static model render or mic-move sweep, plus a rendered-minus-original dB difference panel.
- **Export:** generated 24-bit 48 kHz mono IR WAV download.
- **Audition:** choose a bundled test clip from `data/test_audio/inputs/` or upload a DI guitar clip → convolve (`scipy.signal.fftconvolve`) → A/B players for model IR vs nearest real IR.
- **Mic-move sweep:** render one selected clip through a cap-to-edge offset sweep, with a static reference player beside it.
- **Browser realtime prototype:** `app/realtime.html` embeds the presence-conditioned hidden-256 checkpoint, runs the spectral decoder in JavaScript, and crossfades WebAudio `ConvolverNode` updates while dragging a 2D mic pad. It can audition either the continuous model simulation or the nearest captured God's Cab reference IR; captured-reference mode snaps to measured grid points. Monitoring stays centered because the model labels have radial offset, not signed left/right position.
- **ONNX/WASM runtime:** `scripts/export_onnx.py` exports the release checkpoint to `docs/assets/conetrace-godscab-v0.1.onnx`; `scripts/build_onnx_wasm_demo.py` builds `docs/wasm.html`, which loads the ONNX file through ONNX Runtime Web/WASM and drives the same WebAudio convolver path. Local verification currently reports ONNX-vs-PyTorch max absolute log-magnitude difference **2.86102e-06**.
- **Production readiness report:** `scripts/production_readiness_report.py` writes a markdown gate report plus row-level CSV for the active checkpoint. Current `runs/real_pcond_h256/best.pt`: mean guitar-band grid error **1.01 dB**, worst **3.30 dB**, and **100%** of rows beat a leave-one-position nearest-real baseline; off-axis angle coverage remains WARN.
- **External TONE3000 trend validation:** `scripts/tone3000_external_validation.py` compares the model's cap-center-relative SM57 offset trend against the approved TONE3000 `45023` mini-grid. Current result: 9 offsets from 0 to 50.8 mm, mean trend RMS **1.64 dB** over 100 Hz-6 kHz, mean trend correlation **0.49**, and matching high-frequency direction (`5-10 kHz`: TONE3000 **-2.56 dB**, model **-3.61 dB**). This is directional validation only, not same-cab accuracy.
- **Internal trend validation:** `scripts/internal_trend_validation.py` compares captured vs model cap-to-edge trends inside the God’s Cab grid. Current result: SM57 trend correlation **0.90** and mean trend RMS **1.38 dB** over 100 Hz-6 kHz. The remaining same-domain weakness is SM57 cone-far under-movement in the **2-5 kHz** presence band, not the external TONE3000 `5-10 kHz` over-darkening pattern.
- **Trend-loss training experiment:** `cabir/train.py` now has optional `--trend-loss-*` flags for cap-relative band trend matching. Tested SM57 high-offset variants improved specific SM57 trend/worst-row metrics but did not beat the current default across production gates, so `runs/real_pcond_h256/best.pt` remains the release checkpoint.
- **Checkpoint promotion gate:** `scripts/checkpoint_promotion_gate.py` compares candidate checkpoints against the current release using production rows, internal trend rows, and TONE3000 external trend rows. Current result: all tested trend-loss candidates are rejected; keep `runs/real_pcond_h256/best.pt`.
- **Blind listening validation:** `scripts/build_abx_listening_pack.py` renders randomized, level-matched model-vs-captured ABX WAV trials; `scripts/score_abx_results.py` scores listener guesses and reports a binomial p-value against chance.
- **Model card:** `docs/model-card.md` is the current production-readiness truth sheet: supported controls, validation numbers, listening smoke results, non-claims, and release gates.
- **TONE3000 metadata audit:** `scripts/tone3000_discover.py` performs authenticated API discovery and writes ignored local triage artifacts under `data/tone3000/`. First run: 100 IR-like records scanned, 11 loose-text candidates with mic/distance/offset present, deduped to 9 manual audit rows via `scripts/tone3000_make_audit.py`. `scripts/tone3000_enrich_audit.py` then fetched tone/model metadata and model-name samples for those rows without downloading audio; 6/9 rows remain candidate packs after model-name/API checks. `scripts/tone3000_import_approved.py` is the guarded download gate: it only considers manually approved rows and defaults to dry-run mode. Tone `45023` has been approved for a small local validation probe; `data/parsers/tone3000.py` parses its reviewed filename grammar and `data/parsed_tone3000_probe/` currently contains 13 parsed WAV IRs, including a clean 9-point SM57 on-axis offset line from 0 to 50.8 mm. No TONE3000 IR audio is part of the main training set yet.
- **Still next:** move browser realtime UI from prototype HTML into the standalone UI/workbench.
- Later: the standalone UI project replaces this with a product-native realtime WebAudio surface (the workbench already ships ConvolverNode + worklet infra).

## 5. Milestones

| | Deliverable | Acceptance |
|---|---|---|
| **M1** | Data: multiple packs parsed, labels.parquet, coverage report | ≥1k labeled IRs, license check documented — **DONE: 1,159 IRs (God's Cab 659 + 11 Overdriven SSP2 packs 500), 8 normalized cab labels, 20 mics** |
| **M2** | Baseline A trained on 1 cab/3 mics | beats nearest-neighbor on held-out positions (log-spectral distance) — **DONE: presence-conditioned model 2.55 dB vs 4.89 dB baseline, `runs/real_pcond_h256/best.pt`** |
| **M3** | `infer.py` exports valid `.wav` IRs | **DONE:** real `SpectralDecoder` checkpoint loads and exports 24-bit 48 kHz mono WAV |
| **M4** | Gradio app with position controls + A/B audition | **PROTOTYPE DONE:** waveform/magnitude overlays, WAV export, DI convolve A/B, mic-move sweep, browser realtime drag pad |
| **M5** | Scale: all parsed cabs/mics, TONE3000 ingestion, Phase-2 model experiments | **STARTED:** metadata-only TONE3000 discovery/enrichment ran; 11/100 loose candidates, 9 audit rows, 6 API/model-name candidates. Guarded importer exists but requires manual approval. Still needs final Terms decision, label mapping, and ingest parser before training |

## 6. Risks

- **Sparse grids** → model interpolates between few anchors; mitigation: physics augmentation, report extrapolation honestly in UI (M1 coverage report drives this).
- **Label noise across packs** (distance conventions differ: grill vs cap) → per-pack offset calibration; never mix packs in one (cab, mic) grid without alignment.
- **License ambiguity** → W1 gate: written terms check per pack before training; weights-only distribution.
- **Min-phase assumption** breaks for far/room mics → keep raw-phase A/B path (`is_minphase` flag).
