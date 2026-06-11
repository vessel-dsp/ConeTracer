"""Generate an IR .wav from a trained checkpoint.

Usage:
  python -m cabir.infer --ckpt runs/smoke/model.npz --mic dyn57 \
      --distance 25 --offset 30 --angle 0 --out exports/dyn57_25mm.wav

  python -m cabir.infer --ckpt runs/real/best.pt --cab "Mesa Oversized Rectifier 4x12" \
      --mic sm57 --distance 25 --offset 32 --angle 0 --out exports/sm57_real.wav
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from . import SR
from .dsp import apply_predelay, fade_tail, minphase_from_magnitude, write_wav
from .model import MLP, SpectralDecoder, encode_condition
from .synth import C_MM_S, MIC_NAMES, Condition


def generate_smoke_ir(model: MLP, cond: Condition) -> np.ndarray:
    log_mag = model.forward(encode_condition(cond)[None, :])[0]
    ir = minphase_from_magnitude(np.exp(log_mag))
    ir = apply_predelay(ir, int(round(cond.distance_mm / C_MM_S * SR)))
    return fade_tail(ir)


def _infer_hidden_from_state(state: dict) -> int:
    first = state.get("net.0.weight")
    if first is None:
        raise ValueError("checkpoint is missing net.0.weight; cannot infer hidden size")
    return int(first.shape[0])


def load_real_checkpoint(path: str | Path):
    import torch

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"]
    hidden = int(ckpt.get("hidden") or _infer_hidden_from_state(state))
    actual_hidden = _infer_hidden_from_state(state)
    if hidden != actual_hidden:
        hidden = actual_hidden

    model = SpectralDecoder(
        n_mics=int(ckpt["n_mics"]),
        n_cabs=int(ckpt["n_cabs"]),
        emb_dim=int(ckpt.get("emb_dim", 8)),
        hidden=hidden,
        dropout=0.0,
        n_extra=int(ckpt.get("n_extra", 0)),
    )
    model.load_state_dict(state)
    model.eval()
    return ckpt, model


def generate_real_ir(model: SpectralDecoder, ckpt: dict, *, cab: str, mic: str,
                     distance_mm: float, offset_mm: float, angle_deg: float,
                     presence: float | None = None) -> np.ndarray:
    import torch

    mics = list(ckpt["mics"])
    cabs = list(ckpt["cabs"])
    if mic not in mics:
        raise ValueError(f"unknown mic {mic!r}; checkpoint supports: {', '.join(mics)}")
    if cab not in cabs:
        raise ValueError(f"unknown cab {cab!r}; checkpoint supports: {', '.join(cabs)}")

    dist_norm = float(ckpt["dist_norm"])
    off_norm = float(ckpt["off_norm"])
    extra = None
    if int(ckpt.get("n_extra", 0)) and ckpt.get("condition_presence"):
        if presence is None:
            presence = ckpt.get("presence")
        if presence is None:
            presence = 3.0
        extra = torch.tensor([[(float(presence) - 1.0) / 4.0]], dtype=torch.float32)
    with torch.no_grad():
        log_mag = model(
            torch.tensor([mics.index(mic)], dtype=torch.long),
            torch.tensor([cabs.index(cab)], dtype=torch.long),
            torch.tensor([math.log1p(distance_mm) / dist_norm], dtype=torch.float32),
            torch.tensor([offset_mm / off_norm], dtype=torch.float32),
            torch.tensor([angle_deg / 90.0], dtype=torch.float32),
            extra=extra,
        )[0].numpy()
    return fade_tail(minphase_from_magnitude(np.exp(log_mag)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/smoke/model.npz")
    ap.add_argument("--cab", default=None, help="cab name for real .pt checkpoints")
    ap.add_argument("--mic", default="dyn57")
    ap.add_argument("--distance", type=float, default=25, help="mm from grill")
    ap.add_argument("--offset", type=float, default=30, help="mm from dust cap center")
    ap.add_argument("--angle", type=float, default=0, help="off-axis degrees")
    ap.add_argument("--presence", type=float, default=None, help="presence setting for presence-conditioned real checkpoints")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or f"exports/{args.mic}_d{args.distance:.0f}_o{args.offset:.0f}_a{args.angle:.0f}.wav"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if str(args.ckpt).endswith(".pt"):
        try:
            ckpt, model = load_real_checkpoint(args.ckpt)
            cab = args.cab or ckpt["cabs"][0]
            ir = generate_real_ir(
                model, ckpt, cab=cab, mic=args.mic,
                distance_mm=args.distance, offset_mm=args.offset, angle_deg=args.angle,
                presence=args.presence,
            )
        except ValueError as exc:
            ap.error(str(exc))
        write_wav(out, ir)
        print(f"wrote {out}  ({cab}, {args.mic}, {args.distance}mm, offset {args.offset}mm, {args.angle} deg)")
        return

    if args.mic not in MIC_NAMES:
        ap.error(f"unknown smoke-model mic {args.mic!r}; choices: {', '.join(MIC_NAMES)}")
    cond = Condition(args.mic, args.distance, args.offset, args.angle)
    model = MLP.load(args.ckpt)
    write_wav(out, generate_smoke_ir(model, cond))
    print(f"wrote {out}  ({args.mic}, {args.distance}mm, offset {args.offset}mm, {args.angle} deg)")


if __name__ == "__main__":
    main()
