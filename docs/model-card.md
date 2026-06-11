# ConeTrace Model Card

## Model

- Product/model name: `ConeTrace`
- Release name: `conetrace-godscab-v0.1`
- Internal checkpoint name: `real_pcond_h256`
- Checkpoint: `runs/real_pcond_h256/best.pt`
- Type: conditional spectral decoder
- Output: 4096-sample, 48 kHz cabinet impulse response
- Reconstruction: predicted log-magnitude -> minimum-phase IR
- Runtime surfaces: Python inference, Gradio app, browser realtime demo, ONNX
  Runtime Web/WASM demo

## Intended Use

ConeTrace is intended for continuous guitar cabinet IR auditioning across
close-mic position controls. It is best described as:

> A continuous mic-position cab IR approximation trained on open/public IR
> captures.

It is suitable for demos, internal evaluation, creative guitar tone exploration,
and prototype product validation.

## Supported Controls

- Cab: `Mesa Oversized Rectifier 4x12`
- Mics: `c414`, `sm57`, `sm7b`
- Distance from grille: captured grid from `0` to `50.8 mm`
- Radial offset from dust-cap center: captured grid from `0` to `83 mm`
- Presence: conditioned on God's Cab presence values, default evaluation at `3`
- Tube Screamer state: `NO-TS`

## Known Non-Claims

- Not a physical 3D mic simulation.
- Not a room model.
- Not a speaker-breakup or nonlinear amp model.
- Not trained on signed left/right mic coordinates.
- Not production-validated for off-axis angle. Current close-mic rows are all
  `angle_deg = 0`, so angle controls are extrapolation.
- Not yet validated across multiple independent listeners.

## Training Data

- Parsed data artifact: `data/parsed/{labels.parquet, irs.npy}`
- Core evaluated pack: God's Cab 1.4, 48 kHz close-mic captures
- Broader ingest currently covers 1,159 labeled IRs across God's Cab and
  Overdriven SSP2-series packs.
- TONE3000 is metadata-audit-only at this point. `scripts/tone3000_discover.py`
  probed 100 authenticated IR-like metadata records into ignored local files
  under `data/tone3000/`; 11/100 looked position-grid-labelable by loose text
  heuristics, deduped to 9 manual audit rows with
  `scripts/tone3000_make_audit.py`. `scripts/tone3000_enrich_audit.py` fetched
  tone/model metadata and model-name samples for those rows without downloading
  audio; 6/9 rows remain candidate packs after API/model-name checks.
  `scripts/tone3000_import_approved.py` is a dry-run-by-default guarded
  importer for rows explicitly approved in the audit CSV. Tone `45023` has been
  imported as a small local validation probe and parsed by
  `data/parsers/tone3000.py`; no TONE3000 IR audio is part of this checkpoint or
  the main training set.
- Raw IR packs are training inputs and are not committed to git.

The current parsed artifact was produced without `--minphase`
(`is_minphase=False`), while generated model IRs are reconstructed as
minimum-phase from predicted magnitude.

## Data Credits And Redistribution

ConeTrace v0.1 depends on public/open cabinet IR captures, with attribution and
redistribution boundaries kept separate from this repository's code license:

- **God's Cab 1.4**: primary training and captured-grid validation source for
  `conetrace-godscab-v0.1`, specifically the 48 kHz close-mic Mesa Oversized
  Rectifier 4x12 grid.
- **Overdriven SSP2-series**: used for parser, ingest, and label-coverage work;
  not the promoted v0.1 checkpoint's reported validation domain.
- **TONE3000**: used as a manually approved, small local external trend probe
  only. TONE3000 audio is not included in the checkpoint training data and is
  not redistributed.

Raw third-party IR packs, downloaded TONE3000 files, captured reference audio,
and generated local artifacts are intentionally kept out of git. Users should
obtain third-party packs from their original sources and follow the original
license or Terms. The repository MIT license covers ConeTrace source code and
documentation, not third-party audio captures.

## License

ConeTrace source code and documentation are licensed under the MIT License. The
GitHub Pages app embeds the release model weights for browser inference. Raw
third-party IR packs, downloaded TONE3000 files, generated audio, generated
reports, non-release checkpoints, and listening-test artifacts may have separate
licensing or redistribution restrictions and are intentionally kept out of git.

## Objective Validation

Report:

```bash
python3 scripts/production_readiness_report.py \
  --ckpt runs/real_pcond_h256/best.pt \
  --out runs/production_report
```

Current gate report: `runs/production_report/production_readiness.md`

| Gate | Result |
|---|---:|
| Captured-grid mean guitar-band error | 1.01 dB, PASS |
| Captured-grid worst guitar-band error | 3.30 dB, PASS |
| Leave-one-position nearest-real baseline | 100% rows beat baseline, PASS |
| Browser export matches checkpoint shape | PASS |
| Off-axis angle data coverage | WARN |

Per-mic guitar-band grid error:

| Mic | Mean | Worst |
|---|---:|---:|
| c414 | 0.88 dB | 1.40 dB |
| sm57 | 1.15 dB | 3.30 dB |
| sm7b | 0.99 dB | 1.34 dB |

Worst known area: SM57 cone-far / high-offset positions.

## External Trend Validation

TONE3000 tone `45023` is used as a small local validation probe only, not as
training data and not as same-cab ground truth. The validation compares
cap-center-relative SM57 offset trends between the current model and a clean
TONE3000 on-axis line.

Report:

```bash
python3 scripts/tone3000_external_validation.py \
  --ckpt runs/real_pcond_h256/best.pt \
  --out runs/tone3000_external_validation
```

Current result:

| Metric | Result |
|---|---:|
| Rows selected | 9 |
| Offset range | 0 to 50.8 mm |
| Mean trend RMS, 100 Hz-6 kHz | 1.64 dB |
| Mean trend correlation, 100 Hz-6 kHz | 0.49 |
| TONE3000 5-10 kHz end-to-end change | -2.56 dB |
| Model 5-10 kHz end-to-end change | -3.61 dB |

Interpretation: the external pack shows the same broad high-frequency reduction
as offset increases, while the model darkens more strongly than this TONE3000
line. Treat this as encouraging directional validation, not a production gate.

## Internal Trend Validation

Same-domain trend validation compares captured God’s Cab cap-to-edge movement
against model cap-to-edge movement at the same mic/distance/presence grid
points.

Report:

```bash
LD_LIBRARY_PATH=/home/joseph/anaconda3/lib:$LD_LIBRARY_PATH \
python3 scripts/internal_trend_validation.py \
  --ckpt runs/real_pcond_h256/best.pt \
  --out runs/internal_trend_validation
```

Current result:

| Mic | Mean trend RMS 100 Hz-6 kHz | Mean trend corr | Mean 2-5k diff | Mean abs 2-5k diff | Mean 5-10k diff | Mean abs 5-10k diff |
|---|---:|---:|---:|---:|---:|---:|
| c414 | 1.23 dB | 0.81 | -0.05 dB | 0.17 dB | 0.12 dB | 0.86 dB |
| sm57 | 1.38 dB | 0.90 | 0.39 dB | 0.41 dB | 0.27 dB | 0.66 dB |
| sm7b | 1.27 dB | 0.87 | 0.18 dB | 0.37 dB | 0.29 dB | 0.52 dB |

Interpretation: inside the core God’s Cab domain, offset trend direction is
strong, especially for SM57 (`corr=0.90`). The main remaining same-domain issue
is not the TONE3000-style 5-10 kHz over-darkening; it is SM57 cone-far
under-movement in the 2-5 kHz presence band.

### Trend-Loss Experiment

An optional auxiliary trend loss was added to `cabir/train.py` and tested on
SM57 high-offset rows. It compares each high-offset row to the cap reference in
the same mic/distance/presence line.

| Checkpoint | Production mean | Production worst | Beat nearest | SM57 worst | SM57 abs 2-5k trend diff | SM57 abs 5-10k trend diff | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `real_pcond_h256` | 1.006 dB | 3.297 dB | 100.0% | 3.297 dB | 0.410 dB | 0.660 dB | keep default |
| `real_pcond_h256_trend2k5_sm57` | 1.038 dB | 3.326 dB | 97.2% | 3.326 dB | 0.348 dB | 1.171 dB | reject |
| `real_pcond_h256_trend2k5_sm57_w005` | 1.061 dB | 3.184 dB | 97.2% | 3.184 dB | 0.236 dB | 0.921 dB | reject |
| `real_pcond_h256_trend2k10_sm57_w005` | 1.045 dB | 3.049 dB | 97.2% | 3.049 dB | 0.297 dB | 0.750 dB | experimental |

Interpretation: trend loss can reduce the SM57 worst-row error and improve the
2-5 kHz trend, but the tested variants hurt overall gates or 5-10 kHz behavior.
Do not promote these checkpoints yet. The next model-improvement path should use
a multi-objective selection gate, not training loss alone.

Promotion gate:

```bash
python3 scripts/checkpoint_promotion_gate.py \
  --out runs/checkpoint_promotion_gate
```

Current recommendation: keep `real_pcond_h256` as the release checkpoint. All
tested trend-loss candidates fail at least one promotion rule, most importantly
the 100% leave-one-position nearest-baseline beat rate and/or SM57 5-10 kHz
trend-regression guard.

## Listening Validation

Single-listener ABX smoke tests were run with level-matched model-vs-captured
renders.

| Pack | Focus | Result | Interpretation |
|---|---|---:|---|
| `runs/abx_listening_pack` | broad 12-trial grid, `jazz-hop-guitar.wav` | 5/12 correct | indistinguishable from chance |
| `runs/abx_sm57_edge_pack` | SM57 high-offset positions, `fast-thrash-guitar.wav` | 3/6 correct | indistinguishable from chance |

This is encouraging but not a substitute for multi-listener validation.
Multi-listener testing is deferred.

## Browser Demo

`docs/index.html` is the GitHub Pages build. It embeds the
presence-conditioned ConeTrace checkpoint for browser inference, but does not
embed captured reference IRs or raw third-party audio.

`docs/wasm.html` is the ONNX Runtime Web/WASM build. It loads
`docs/assets/conetrace-godscab-v0.1.onnx` and runs the same checkpoint through a
browser WASM backend. Local export verification currently reports max absolute
ONNX-vs-PyTorch log-magnitude difference of `2.86102e-06`.

`app/realtime.html` is the local validation build. It embeds the same checkpoint
plus 36 captured reference IRs for comparison. It supports:

- continuous model simulation while dragging the mic
- nearest captured reference IR audition, snapping to measured grid points
- centered monitoring only, because left/right position is not labeled

## Release Status

Current status: **alpha / technical preview**.

Recommended production wording:

- OK: "ConeTrace"
- OK: "continuous mic-position cabinet IR modeling"
- OK: "continuous mic-position cab IR approximation"
- OK: "trained and validated against open/public captured IRs"
- OK: "browser realtime audition prototype"
- Avoid: "physically exact mic simulation"
- Avoid: "models room profile"
- Avoid: "validated off-axis mic angle"

## Next Gates

Before a product/beta release:

- Add true off-axis captures or a trustworthy angle-labeled pack.
- Finish the TONE3000 gate: final Terms decision, manual label mapping,
  importer approval, and a no-raw-audio redistribution check.
- Run ABX with at least 3 listeners or 20+ scored trials.
- Add reproducible export smoke tests for Python inference and browser HTML.
- Document raw-pack license posture for every training source.
- Version the checkpoint and browser export together.
