# cab-ir-lab

Neural guitar **cabinet IR simulation** for VesselDSP: train a model on labeled public IR packs, then *generate* a `.wav` impulse response for any (cab, speaker, mic, position) — including positions never captured — and audition mic movement live.

Part of the VesselDSP family: `audio-engine` (workbench), `@vessel-dsp/react-pedal-schematic` (.vdsp), `pedal-stompbox` (physical builds). Output IRs are drop-in compatible with audio-engine's `ConvolverNode` cab lane.

```
cab-ir-lab/
├── docs/design.md      # full project design: data, model, inference, UI, capture theory
├── data/
│   ├── raw/            # downloaded IR packs (gitignored — license)
│   ├── parsed/         # normalized IRs + labels.parquet
│   └── parsers/        # per-pack filename→label parsers
├── cabir/              # python package
│   ├── dataset.py      # torch Dataset: IR + conditioning vector
│   ├── model.py        # conditional IR generator
│   ├── losses.py       # multi-res STFT + log-mag losses
│   ├── train.py        # training entrypoint
│   ├── infer.py        # checkpoint + condition → IR .wav
│   └── dsp.py          # min-phase reconstruction, resampling, windowing
├── app/gradio_app.py   # test UI: mic position pad → IR plot + live audition
└── exports/            # generated .wav IRs
```

## Quickstart (zero downloads — synthetic smoke pipeline)

Everything below runs on numpy/scipy only (no PyTorch, no datasets, <20 MB disk):

```bash
python3 -m cabir.train                # train conditional spectral decoder (~30s CPU)
python3 -m cabir.infer --mic dyn57 --distance 25 --offset 30   # -> exports/*.wav
python3 scripts/build_demo.py         # -> app/demo.html (single file, open in browser)
```

`app/demo.html` is fully self-contained: model weights baked in, forward pass +
min-phase reconstruction in JS, Karplus-Strong guitar riff through a WebAudio
ConvolverNode. Drag the mic across the speaker and hear the position change.

Smoke results (synthetic cab, 128 held-out positions): model **0.83 dB** log-spectral
distance vs **1.86 dB** for the nearest-captured-IR baseline.

Real-data phase (Redwirez / God's Cab parsers, PyTorch port) happens on a machine
with disk space — only `data/parsers/` and the training backend change; the
contract (`condition -> log-magnitude -> min-phase IR -> .wav`) is now proven.

See [docs/design.md](docs/design.md) for the full design.
