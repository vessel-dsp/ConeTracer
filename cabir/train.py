"""Train the conditional spectral decoder.

Synthetic smoke (default, zero downloads):
    python -m cabir.train [--steps 3000] [--out runs/smoke]

Real-data training (M2, requires data/parsed/):
    python -m cabir.train --real [--cab "Mesa Oversized Rectifier 4x12"]
                                 [--mics sm57 sm7b c414]
                                 [--presence 3] [--epochs 500] [--out runs/real]

Both modes evaluate against the nearest-captured-IR baseline (the model must
beat "just pick the closest captured IR" to justify itself).
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from .dsp import log_spectral_distance
from .model import MLP, encode_condition
from .synth import Condition, cab_magnitude, sample_conditions


# ── synthetic smoke pipeline (unchanged) ──────────────────────────────────────

def build_split(n_train: int, n_anchor: int, n_val: int, cab_seed: int = 0):
    train = sample_conditions(n_train, seed=1, cab_seed=cab_seed)
    anchors = sample_conditions(n_anchor, seed=2, cab_seed=cab_seed)
    val = sample_conditions(n_val, seed=3, cab_seed=cab_seed)
    return train, anchors, val


def dataset(conds) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack([encode_condition(c) for c in conds])
    y = np.stack([np.log(cab_magnitude(c)) for c in conds])
    return x, y


def nn_baseline_synth(anchors, anchor_mags, val, val_mags) -> float:
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


# ── real-data training (PyTorch / SpectralDecoder) ────────────────────────────

def _freq_weights(n_bins: int, sr: int = 48_000, n_taps: int = 4096) -> "torch.Tensor":
    """2× weight for perceptually important 100 Hz–8 kHz region."""
    import torch
    freqs = np.arange(n_bins) * sr / n_taps
    w = np.where((freqs >= 100) & (freqs <= 8000), 2.0, 1.0).astype(np.float32)
    return torch.from_numpy(w)


def _band_mask(n_bins: int, lo_hz: float, hi_hz: float, sr: int = 48_000, n_taps: int = 4096) -> "torch.Tensor":
    import torch

    freqs = np.arange(n_bins) * sr / n_taps
    return torch.from_numpy(((freqs >= lo_hz) & (freqs <= hi_hz)).astype(bool))


def _build_trend_pairs(train_ds, args) -> list[tuple[int, int]]:
    """Pairs of (cap/reference row, high-offset row) for trend auxiliary loss."""
    if args.trend_loss_weight <= 0:
        return []
    mics = set(args.trend_loss_mics or train_ds.mics)
    pairs: list[tuple[int, int]] = []
    df = train_ds.df.reset_index(drop=True)
    group_cols = ["mic", "distance_mm", "angle_deg", "presence"]
    for _, group in df[df["mic"].isin(mics)].groupby(group_cols, dropna=False):
        if len(group["offset_mm"].dropna().unique()) < 2:
            continue
        ref_idx = int(group["offset_mm"].astype(float).idxmin())
        targets = group[group["offset_mm"].astype(float) >= float(args.trend_loss_min_offset_mm)]
        for target_idx in targets.index.tolist():
            if int(target_idx) != ref_idx:
                pairs.append((ref_idx, int(target_idx)))
    return pairs


def _trend_pair_tensor(train_ds, pairs, device):
    import torch

    def stack(key: str, indices: list[int]):
        return torch.stack([train_ds[i][key] for i in indices]).to(device)

    if not pairs:
        return None
    ref_idx = [a for a, _ in pairs]
    tgt_idx = [b for _, b in pairs]
    return {
        "ref": {
            "mic_idx": stack("mic_idx", ref_idx),
            "cab_idx": stack("cab_idx", ref_idx),
            "distance": stack("distance", ref_idx),
            "offset": stack("offset", ref_idx),
            "angle": stack("angle", ref_idx),
            "presence": stack("presence", ref_idx),
            "logmag": stack("logmag", ref_idx),
        },
        "target": {
            "mic_idx": stack("mic_idx", tgt_idx),
            "cab_idx": stack("cab_idx", tgt_idx),
            "distance": stack("distance", tgt_idx),
            "offset": stack("offset", tgt_idx),
            "angle": stack("angle", tgt_idx),
            "presence": stack("presence", tgt_idx),
            "logmag": stack("logmag", tgt_idx),
        },
    }


def _trend_loss(model, trend_batch, band_mask, condition_presence: bool):
    import torch

    if trend_batch is None:
        return torch.tensor(0.0, device=band_mask.device)
    ref = trend_batch["ref"]
    tgt = trend_batch["target"]
    extra_ref = ref["presence"].unsqueeze(-1) if condition_presence else None
    extra_tgt = tgt["presence"].unsqueeze(-1) if condition_presence else None
    pred_ref = model(ref["mic_idx"], ref["cab_idx"], ref["distance"], ref["offset"], ref["angle"], extra=extra_ref)
    pred_tgt = model(tgt["mic_idx"], tgt["cab_idx"], tgt["distance"], tgt["offset"], tgt["angle"], extra=extra_tgt)
    pred_delta = (pred_tgt[:, band_mask] - pred_ref[:, band_mask]).mean(dim=-1)
    true_delta = (tgt["logmag"][:, band_mask] - ref["logmag"][:, band_mask]).mean(dim=-1)
    return (pred_delta - true_delta).abs().mean()


def _lsd_np(pred_logmag: np.ndarray, true_logmag: np.ndarray) -> float:
    """RMS log-spectral distance in dB (matches dsp.log_spectral_distance)."""
    diff_db = (pred_logmag - true_logmag) * (20.0 / math.log(10))
    return float(np.sqrt(np.mean(diff_db ** 2)))


def nn_baseline_real(train_ds, val_ds) -> float:
    """Nearest train IR (same mic, closest normalised position) for each val sample."""
    lsds = []
    for vi in range(len(val_ds)):
        vrow = val_ds.df.iloc[vi]
        vlogmag = val_ds[vi]["logmag"].numpy()
        best_lsd, best_logmag = np.inf, None
        for ti in range(len(train_ds)):
            trow = train_ds.df.iloc[ti]
            if trow.mic != vrow.mic:
                continue
            d_dist = (math.log1p(float(trow.distance_mm)) - math.log1p(float(vrow.distance_mm))) ** 2
            d_off = ((float(trow.offset_mm) - float(vrow.offset_mm)) / train_ds.off_norm) ** 2
            d = math.sqrt(d_dist + d_off)
            if d < best_lsd:
                best_lsd = d
                best_logmag = train_ds[ti]["logmag"].numpy()
        if best_logmag is not None:
            lsds.append(_lsd_np(best_logmag, vlogmag))
    return float(np.mean(lsds)) if lsds else float("nan")


def train_real(args) -> None:
    import torch
    from torch.utils.data import DataLoader

    from .dataset import IRDataset
    from .model import SpectralDecoder
    from . import N_BINS, SR, N_TAPS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    mics = args.mics if args.mics else ["sm57", "sm7b", "c414"]

    presence = None if (args.all_presence or args.condition_presence) else args.presence
    ds_kwargs = dict(
        cab=args.cab or "Mesa Oversized Rectifier 4x12",
        mics=mics,
        ts=False,
        presence=presence,
        capture_type="close",
        val_frac=0.2,
        seed=42,
    )
    train_ds = IRDataset(**ds_kwargs, split="train")
    val_ds   = IRDataset(**ds_kwargs, split="val")

    # pass normalization from train to val so they share the same scale
    val_ds.dist_norm = train_ds.dist_norm
    val_ds.off_norm  = train_ds.off_norm

    print(f"dataset  : {len(train_ds)} train  /  {len(val_ds)} val")
    print(f"mics     : {train_ds.mics}")
    print(f"cabs     : {train_ds.cabs}")
    print(f"presence : {'all' if presence is None else presence}")
    print(f"off_norm : {train_ds.off_norm:.1f} mm  dist_norm : {train_ds.dist_norm:.3f}")
    print(
        "augment : "
        f"dist_jitter={args.distance_jitter_mm:g}mm  "
        f"offset_jitter={args.offset_jitter_mm:g}mm  "
        f"angle_jitter={args.angle_jitter_deg:g}deg  "
        f"edge_weight={args.edge_weight:g}x@{args.edge_weight_offset_mm:g}mm"
    )
    print(f"extra    : condition_presence={args.condition_presence}")
    trend_pairs = _build_trend_pairs(train_ds, args)
    print(
        "trend   : "
        f"weight={args.trend_loss_weight:g}  "
        f"band={args.trend_loss_band_lo:g}-{args.trend_loss_band_hi:g}Hz  "
        f"min_offset={args.trend_loss_min_offset_mm:g}mm  "
        f"pairs={len(trend_pairs)}"
    )

    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    hidden = args.hidden
    emb_dim = 8
    dropout = 0.1
    n_extra = 1 if args.condition_presence else 0
    model = SpectralDecoder(
        n_mics=len(train_ds.mics),
        n_cabs=len(train_ds.cabs),
        emb_dim=emb_dim,
        hidden=hidden,
        dropout=dropout,
        n_extra=n_extra,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params   : {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    freq_w = _freq_weights(N_BINS, SR, N_TAPS).to(device)
    trend_band = _band_mask(N_BINS, args.trend_loss_band_lo, args.trend_loss_band_hi, SR, N_TAPS).to(device)
    trend_batch = _trend_pair_tensor(train_ds, trend_pairs, device)

    os.makedirs(args.out, exist_ok=True)
    best_val_lsd = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            mic_idx  = batch["mic_idx"].to(device)
            cab_idx  = batch["cab_idx"].to(device)
            distance = batch["distance"].to(device)
            offset   = batch["offset"].to(device)
            angle    = batch["angle"].to(device)
            target   = batch["logmag"].to(device)
            extra = batch["presence"].to(device).unsqueeze(-1) if args.condition_presence else None

            if args.distance_jitter_mm > 0:
                sigma = math.log1p(args.distance_jitter_mm) / train_ds.dist_norm
                distance = (distance + torch.randn_like(distance) * sigma).clamp(min=0.0)
            if args.offset_jitter_mm > 0:
                sigma = args.offset_jitter_mm / train_ds.off_norm
                offset = (offset + torch.randn_like(offset) * sigma).clamp(0.0, 1.0)
            if args.angle_jitter_deg > 0:
                sigma = args.angle_jitter_deg / 90.0
                angle = (angle + torch.randn_like(angle) * sigma).clamp(0.0, 1.0)

            pred = model(mic_idx, cab_idx, distance, offset, angle, extra=extra)
            per_sample_loss = (freq_w * (pred - target).abs()).mean(dim=-1)
            if args.edge_weight != 1.0:
                edge_norm = args.edge_weight_offset_mm / train_ds.off_norm
                sample_w = torch.where(
                    offset >= edge_norm,
                    torch.full_like(offset, args.edge_weight),
                    torch.ones_like(offset),
                )
                loss = (per_sample_loss * sample_w).mean()
            else:
                loss = per_sample_loss.mean()
            if args.trend_loss_weight > 0 and trend_batch is not None:
                loss = loss + args.trend_loss_weight * _trend_loss(
                    model,
                    trend_batch,
                    trend_band,
                    args.condition_presence,
                )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()

        if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for vbatch in DataLoader(val_ds, batch_size=len(val_ds)):
                    pred_v = model(
                        vbatch["mic_idx"].to(device),
                        vbatch["cab_idx"].to(device),
                        vbatch["distance"].to(device),
                        vbatch["offset"].to(device),
                        vbatch["angle"].to(device),
                        extra=vbatch["presence"].to(device).unsqueeze(-1) if args.condition_presence else None,
                    )
                    # LSD in dB per sample
                    diff_db = (pred_v.cpu() - vbatch["logmag"]) * (20.0 / math.log(10))
                    val_lsd = diff_db.pow(2).mean(dim=-1).sqrt().mean().item()
                    val_losses.append(val_lsd)
            val_lsd = float(np.mean(val_losses))
            lr_now = sched.get_last_lr()[0]
            print(
                f"epoch {epoch:4d}/{args.epochs}  "
                f"train_loss {epoch_loss/len(loader):.4f}  "
                f"val_LSD {val_lsd:.2f} dB  "
                f"lr {lr_now:.1e}  "
                f"({time.time()-t0:.0f}s)"
            )
            if val_lsd < best_val_lsd:
                best_val_lsd = val_lsd
                ckpt = {
                    "model_type": "SpectralDecoder",
                    "model_state": model.state_dict(),
                    "n_mics": len(train_ds.mics),
                    "n_cabs": len(train_ds.cabs),
                    "emb_dim": emb_dim,
                    "hidden": hidden,
                    "dropout": dropout,
                    "n_extra": n_extra,
                    "condition_presence": args.condition_presence,
                    "mics": train_ds.mics,
                    "cabs": train_ds.cabs,
                    "dist_norm": train_ds.dist_norm,
                    "off_norm": train_ds.off_norm,
                    "presence": presence,
                    "ts": False,
                    "capture_type": "close",
                    "augmentation": {
                        "distance_jitter_mm": args.distance_jitter_mm,
                        "offset_jitter_mm": args.offset_jitter_mm,
                        "angle_jitter_deg": args.angle_jitter_deg,
                        "edge_weight": args.edge_weight,
                        "edge_weight_offset_mm": args.edge_weight_offset_mm,
                        "trend_loss_weight": args.trend_loss_weight,
                        "trend_loss_band_lo": args.trend_loss_band_lo,
                        "trend_loss_band_hi": args.trend_loss_band_hi,
                        "trend_loss_min_offset_mm": args.trend_loss_min_offset_mm,
                        "trend_loss_mics": args.trend_loss_mics,
                    },
                }
                torch.save(ckpt, os.path.join(args.out, "best.pt"))

    # ── final eval ────────────────────────────────────────────────────
    print(f"\n== nearest-neighbor baseline (computing over {len(val_ds)} val samples) ==")
    nn_lsd = nn_baseline_real(train_ds, val_ds)
    print(f"nearest-captured-IR baseline : {nn_lsd:.2f} dB LSD")
    print(f"model (best checkpoint)      : {best_val_lsd:.2f} dB LSD")
    if best_val_lsd < nn_lsd:
        print("PASS — model beats nearest-neighbor")
    else:
        print("FAIL — model does not beat baseline (try more epochs or check data)")

    print(f"\ncheckpoint saved to {args.out}/best.pt")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="train on real parsed IRs (M2)")
    # real-data args
    ap.add_argument("--cab", default=None, help="cab filter (default: Mesa OS Rectifier)")
    ap.add_argument("--mics", nargs="+", default=None, help="mic filter (default: sm57 sm7b c414)")
    ap.add_argument("--presence", type=float, default=3.0,
                    help="God's Cab power-amp presence value to train on (default: 3)")
    ap.add_argument("--all-presence", action="store_true",
                    help="pool all presence values; use only for explicit augmentation experiments")
    ap.add_argument("--condition-presence", action="store_true",
                    help="include God's Cab presence setting as an extra conditioning scalar and train on all presence values")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--distance-jitter-mm", type=float, default=0.0,
                    help="training-time Gaussian jitter on distance conditioning")
    ap.add_argument("--offset-jitter-mm", type=float, default=0.0,
                    help="training-time Gaussian jitter on offset conditioning")
    ap.add_argument("--angle-jitter-deg", type=float, default=0.0,
                    help="training-time Gaussian jitter on angle conditioning")
    ap.add_argument("--edge-weight", type=float, default=1.0,
                    help="loss multiplier for cone-edge samples")
    ap.add_argument("--edge-weight-offset-mm", type=float, default=80.0,
                    help="offset threshold for --edge-weight")
    ap.add_argument("--trend-loss-weight", type=float, default=0.0,
                    help="auxiliary loss weight for cap-relative band trend matching")
    ap.add_argument("--trend-loss-band-lo", type=float, default=2000.0,
                    help="low edge of auxiliary trend-loss band")
    ap.add_argument("--trend-loss-band-hi", type=float, default=5000.0,
                    help="high edge of auxiliary trend-loss band")
    ap.add_argument("--trend-loss-min-offset-mm", type=float, default=57.0,
                    help="target offsets included in auxiliary trend loss")
    ap.add_argument("--trend-loss-mics", nargs="+", default=None,
                    help="mic labels included in auxiliary trend loss")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    # synthetic args
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--n-anchor", type=int, default=160)
    ap.add_argument("--n-val", type=int, default=128)
    # shared
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.real:
        args.out = args.out or "runs/real"
        train_real(args)
        return

    # ── synthetic smoke (unchanged) ───────────────────────────────────
    args.out = args.out or "runs/smoke"
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

    nn_lsd = nn_baseline_synth(anchors, y_anchor, val, y_val)
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
