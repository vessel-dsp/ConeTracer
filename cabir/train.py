"""Train the conditional spectral decoder on synthetic IRs.

Usage: python -m cabir.train [--steps 3000] [--n-train 2048] [--out runs/smoke]

Also evaluates against the nearest-neighbor-IR baseline (the model must beat
"just pick the closest captured IR" to justify itself).
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .dsp import log_spectral_distance
from .model import MLP, encode_condition
from .synth import Condition, cab_magnitude, sample_conditions


def build_split(n_train: int, n_anchor: int, n_val: int, cab_seed: int = 0):
    train = sample_conditions(n_train, seed=1, cab_seed=cab_seed)
    anchors = sample_conditions(n_anchor, seed=2, cab_seed=cab_seed)  # "captured pack"
    val = sample_conditions(n_val, seed=3, cab_seed=cab_seed)
    return train, anchors, val


def dataset(conds) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([encode_condition(c) for c in conds])
    y = np.stack([np.log(cab_magnitude(c)) for c in conds])
    return x, y


def nn_baseline(anchors, anchor_mags, val, val_mags) -> float:
    """Nearest captured IR (same mic, closest position) — the bar to beat."""
    dists = []
    for cond, true_mag in zip(val, val_mags):
        best, best_d = None, np.inf
        for a, a_mag in zip(anchors, anchor_mags):
            if a.mic != cond.mic:
                continue
            d = (
                (np.log(a.distance_mm) - np.log(cond.distance_mm)) ** 2
                + ((a.offset_mm - cond.offset_mm) / 65) ** 2
                + ((a.angle_deg - cond.angle_deg) / 22) ** 2
            )
            if d < best_d:
                best, best_d = a_mag, d
        dists.append(log_spectral_distance(np.exp(true_mag), np.exp(best)))
    return float(np.mean(dists))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-anchor", type=int, default=160)
    ap.add_argument("--n-val", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs/smoke")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train, anchors, val = build_split(args.n_train, args.n_anchor, args.n_val)
    print(f"building dataset: {args.n_train} train / {args.n_anchor} anchors / {args.n_val} val")
    x_train, y_train = dataset(train)
    x_val, y_val = dataset(val)
    _, y_anchor = dataset(anchors)

    model = MLP()
    n_params = sum(v.size for v in model.params.values())
    print(f"model params: {n_params:,}")

    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, len(x_train), args.batch)
        cache = {}
        pred = model.forward(x_train[idx], cache)
        err = pred - y_train[idx]
        model.adam_step(model.backward(cache, 2 * err), lr=args.lr)
        if step % 500 == 0 or step == args.steps:
            val_pred = model.forward(x_val)
            lsd = np.mean(
                [log_spectral_distance(np.exp(t), np.exp(p)) for t, p in zip(y_val, val_pred)]
            )
            print(f"step {step:5d}  train_mse {np.mean(err**2):.4f}  val_LSD {lsd:.2f} dB  ({time.time()-t0:.0f}s)")

    model.save(os.path.join(args.out, "model.npz"))
    model.export_json(os.path.join(args.out, "model.json"))

    nn_lsd = nn_baseline(anchors, y_anchor, val, y_val)
    val_pred = model.forward(x_val)
    model_lsd = np.mean(
        [log_spectral_distance(np.exp(t), np.exp(p)) for t, p in zip(y_val, val_pred)]
    )
    print(f"\n== eval on {len(val)} held-out positions ==")
    print(f"nearest-captured-IR baseline: {nn_lsd:.2f} dB LSD")
    print(f"model:                        {model_lsd:.2f} dB LSD")
    print("PASS — model beats nearest-neighbor" if model_lsd < nn_lsd else "FAIL — model does not beat baseline")


if __name__ == "__main__":
    main()
