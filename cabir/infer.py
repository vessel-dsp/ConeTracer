"""Generate an IR .wav from a trained checkpoint.

Usage:
  python -m cabir.infer --ckpt runs/smoke/model.npz --mic dyn57 \
      --distance 25 --offset 30 --angle 0 --out exports/dyn57_25mm.wav
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from . import SR
from .dsp import apply_predelay, fade_tail, minphase_from_magnitude, write_wav
from .model import MLP, encode_condition
from .synth import C_MM_S, MIC_NAMES, Condition


def generate_ir(model: MLP, cond: Condition) -> np.ndarray:
    log_mag = model.forward(encode_condition(cond)[None, :])[0]
    ir = minphase_from_magnitude(np.exp(log_mag))
    ir = apply_predelay(ir, int(round(cond.distance_mm / C_MM_S * SR)))
    return fade_tail(ir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/smoke/model.npz")
    ap.add_argument("--mic", choices=MIC_NAMES, default="dyn57")
    ap.add_argument("--distance", type=float, default=25, help="mm from grill")
    ap.add_argument("--offset", type=float, default=30, help="mm from dust cap center")
    ap.add_argument("--angle", type=float, default=0, help="off-axis degrees")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cond = Condition(args.mic, args.distance, args.offset, args.angle)
    out = args.out or f"exports/{args.mic}_d{args.distance:.0f}_o{args.offset:.0f}_a{args.angle:.0f}.wav"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    model = MLP.load(args.ckpt)
    write_wav(out, generate_ir(model, cond))
    print(f"wrote {out}  ({args.mic}, {args.distance}mm, offset {args.offset}mm, {args.angle} deg)")


if __name__ == "__main__":
    main()
