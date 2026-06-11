"""Generate a repeatable production-readiness report for a cab IR checkpoint.

The report is intentionally conservative: it scores generated IRs against
captured IRs at the measured grid points, compares against a leave-one-position
nearest-real baseline, and records data caveats that matter for product claims.

Usage:
    python scripts/production_readiness_report.py \
      --ckpt runs/real_pcond_h256/best.pt \
      --out runs/production_report
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir import N_TAPS, SR  # noqa: E402
from cabir.infer import generate_real_ir, load_real_checkpoint  # noqa: E402


def _mag_db(ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(N_TAPS, 1.0 / SR)
    mag = np.abs(np.fft.rfft(ir, n=N_TAPS))
    return freqs, 20 * np.log10(np.maximum(mag, 1e-8))


def _gain_normalize_db(freqs: np.ndarray, db: np.ndarray, lo: float = 100, hi: float = 6000) -> np.ndarray:
    band = (freqs >= lo) & (freqs <= hi)
    ref = float(np.median(db[band])) if np.any(band) else float(np.median(db))
    return db - ref


def _ir_error_metrics(pred_ir: np.ndarray, target_ir: np.ndarray) -> tuple[float, float]:
    freqs, pred_db = _mag_db(pred_ir)
    _, target_db = _mag_db(target_ir)
    pred_norm = _gain_normalize_db(freqs, pred_db)
    target_norm = _gain_normalize_db(freqs, target_db)
    delta = pred_norm - target_norm
    guitar_band = (freqs >= 100) & (freqs <= 6000)
    wide_band = (freqs >= 100) & (freqs <= 12_000)
    guitar = float(np.sqrt(np.mean(delta[guitar_band] ** 2))) if np.any(guitar_band) else float("nan")
    wide = float(np.sqrt(np.mean(delta[wide_band] ** 2))) if np.any(wide_band) else float("nan")
    return guitar, wide


def _presence_for_checkpoint(ckpt: dict) -> float | None:
    presence = ckpt.get("presence")
    if presence is None and ckpt.get("condition_presence"):
        return 3.0
    return presence


def _filter_eval_rows(labels: pd.DataFrame, ckpt: dict, cab: str, mic: str) -> pd.DataFrame:
    mask = (
        (labels["cab"] == cab)
        & (labels["mic"] == mic)
        & (labels["capture_type"] == "close")
        & (labels["ts"] == bool(ckpt.get("ts", False)))
    )
    presence = _presence_for_checkpoint(ckpt)
    if presence is not None:
        mask &= labels["presence"] == float(presence)
    return labels[mask].copy().reset_index(drop=True)


def _nearest_other_ir(row: pd.Series, rows: pd.DataFrame, irs: np.ndarray, dist_norm: float, off_norm: float):
    candidates = rows[rows.index != row.name].copy()
    if candidates.empty:
        return None, None
    candidates["_d"] = (
        (np.log1p(candidates["distance_mm"].astype(float)) / dist_norm - math.log1p(float(row.distance_mm)) / dist_norm) ** 2
        + (candidates["offset_mm"].astype(float) / off_norm - float(row.offset_mm) / off_norm) ** 2
        + (candidates["angle_deg"].astype(float) / 90.0 - float(row.angle_deg) / 90.0) ** 2
    )
    nearest = candidates.sort_values("_d").iloc[0]
    return np.asarray(irs[int(nearest["index"])], dtype=np.float64), nearest


def _browser_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    match = re.search(r"const META = (.*?);\nconst BYTES", text, re.S)
    if not match:
        return None
    return json.loads(match.group(1))


def _status_for_gate(ok: bool, warn: bool = False) -> str:
    if ok:
        return "PASS"
    return "WARN" if warn else "FAIL"


def build_report(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    ckpt, model = load_real_checkpoint(args.ckpt)
    labels = pd.read_parquet(args.labels)
    irs = np.load(args.irs, mmap_mode="r")
    cab = args.cab or ckpt["cabs"][0]
    rows_out: list[dict] = []

    for mic in ckpt["mics"]:
        rows = _filter_eval_rows(labels, ckpt, cab, mic)
        for idx, row in rows.iterrows():
            row = row.copy()
            row.name = idx
            presence = float(row.presence) if not np.isnan(row.presence) else _presence_for_checkpoint(ckpt)
            model_ir = generate_real_ir(
                model,
                ckpt,
                cab=cab,
                mic=mic,
                distance_mm=float(row.distance_mm),
                offset_mm=float(row.offset_mm),
                angle_deg=float(row.angle_deg),
                presence=presence,
            )
            real_ir = np.asarray(irs[int(row["index"])], dtype=np.float64)
            model_guitar, model_wide = _ir_error_metrics(model_ir, real_ir)
            nearest_ir, nearest_row = _nearest_other_ir(
                row,
                rows,
                irs,
                dist_norm=float(ckpt["dist_norm"]),
                off_norm=float(ckpt["off_norm"]),
            )
            if nearest_ir is None:
                nn_guitar = nn_wide = float("nan")
                nn_file = ""
            else:
                nn_guitar, nn_wide = _ir_error_metrics(nearest_ir, real_ir)
                nn_file = str(nearest_row.file)
            rows_out.append(
                {
                    "cab": cab,
                    "mic": mic,
                    "presence": presence,
                    "distance_mm": float(row.distance_mm),
                    "offset_mm": float(row.offset_mm),
                    "angle_deg": float(row.angle_deg),
                    "model_guitar_rms_db": model_guitar,
                    "model_wide_rms_db": model_wide,
                    "nearest_other_guitar_rms_db": nn_guitar,
                    "nearest_other_wide_rms_db": nn_wide,
                    "beats_nearest_other": bool(model_guitar < nn_guitar) if np.isfinite(nn_guitar) else False,
                    "file": row.file,
                    "nearest_other_file": nn_file,
                }
            )

    results = pd.DataFrame(rows_out)
    if results.empty:
        raise SystemExit("No evaluation rows found for this checkpoint/filter.")

    mic_summary = (
        results.groupby("mic")
        .agg(
            positions=("file", "count"),
            mean_guitar_db=("model_guitar_rms_db", "mean"),
            worst_guitar_db=("model_guitar_rms_db", "max"),
            mean_wide_db=("model_wide_rms_db", "mean"),
            worst_wide_db=("model_wide_rms_db", "max"),
            nearest_mean_guitar_db=("nearest_other_guitar_rms_db", "mean"),
            beat_rate=("beats_nearest_other", "mean"),
        )
        .reset_index()
    )

    angle_counts = labels[
        (labels["cab"] == cab)
        & (labels["mic"].isin(ckpt["mics"]))
        & (labels["capture_type"] == "close")
    ]["angle_deg"].value_counts(dropna=False).sort_index()
    unique_angles = [float(x) for x in angle_counts.index.tolist() if not pd.isna(x)]
    browser = _browser_meta(args.browser_html)
    browser_ok = (
        browser is not None
        and bool(browser.get("condition_presence")) == bool(ckpt.get("condition_presence", False))
        and int(browser.get("input_dim", -1)) == int(ckpt["model_state"]["net.0.weight"].shape[1])
        and list(browser.get("mics", [])) == list(ckpt["mics"])
    )

    mean_guitar = float(results["model_guitar_rms_db"].mean())
    worst_guitar = float(results["model_guitar_rms_db"].max())
    beat_rate = float(results["beats_nearest_other"].mean())
    mean_gate = mean_guitar <= args.mean_guitar_gate
    worst_gate = worst_guitar <= args.worst_guitar_gate
    baseline_gate = beat_rate >= args.beat_rate_gate
    angle_gate = len(set(unique_angles)) > 1

    gates = [
        ("Captured-grid mean guitar error", _status_for_gate(mean_gate), f"{mean_guitar:.2f} dB <= {args.mean_guitar_gate:.2f} dB"),
        ("Captured-grid worst guitar error", _status_for_gate(worst_gate, warn=True), f"{worst_guitar:.2f} dB <= {args.worst_guitar_gate:.2f} dB"),
        ("Leave-one-position nearest baseline", _status_for_gate(baseline_gate, warn=True), f"{beat_rate * 100:.0f}% rows beat nearest-other real IR >= {args.beat_rate_gate * 100:.0f}%"),
        ("Browser export matches checkpoint shape", _status_for_gate(browser_ok), str(browser_ok)),
        ("Off-axis angle data coverage", _status_for_gate(angle_gate, warn=True), f"{len(set(unique_angles))} unique angle(s): {unique_angles}"),
    ]

    lines = [
        "# cab-ir-lab production readiness report",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Cab: `{cab}`",
        f"- Mics: `{', '.join(ckpt['mics'])}`",
        f"- Presence mode: `{'conditioned' if ckpt.get('condition_presence') else _presence_for_checkpoint(ckpt)}`",
        f"- Evaluation rows: `{len(results)}`",
        "",
        "## Gates",
        "",
        "| Gate | Status | Detail |",
        "|---|---:|---|",
    ]
    lines.extend(f"| {name} | **{status}** | {detail} |" for name, status, detail in gates)
    lines.extend(
        [
            "",
            "## Summary By Mic",
            "",
            mic_summary.to_markdown(index=False, floatfmt=".2f"),
            "",
            "## Notes",
            "",
            "- `model_*_rms_db` is gain-normalized RMS magnitude error, matching the Gradio validation grid.",
            "- `nearest_other_*` is a leave-one-position baseline using the closest captured IR for the same mic/cab/presence, excluding the target row.",
            "- The current labels record radial offset from dust-cap center, not signed left/right position.",
            "- Angle should not be marketed as production behavior until the data has true off-axis captures.",
            "",
            "## Worst Rows",
            "",
            results.sort_values("model_guitar_rms_db", ascending=False)
            .head(args.worst_rows)
            [
                [
                    "mic",
                    "distance_mm",
                    "offset_mm",
                    "model_guitar_rms_db",
                    "model_wide_rms_db",
                    "nearest_other_guitar_rms_db",
                    "file",
                ]
            ]
            .to_markdown(index=False, floatfmt=".2f"),
            "",
        ]
    )
    return results, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/real_pcond_h256/best.pt")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/parsed/labels.parquet")
    ap.add_argument("--irs", type=Path, default=REPO_ROOT / "data/parsed/irs.npy")
    ap.add_argument("--browser-html", type=Path, default=REPO_ROOT / "app/realtime.html")
    ap.add_argument("--cab", default=None)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs/production_report")
    ap.add_argument("--mean-guitar-gate", type=float, default=1.5)
    ap.add_argument("--worst-guitar-gate", type=float, default=3.5)
    ap.add_argument("--beat-rate-gate", type=float, default=0.75)
    ap.add_argument("--worst-rows", type=int, default=10)
    args = ap.parse_args()

    results, report = build_report(args)
    args.out.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "validation_rows.csv", index=False)
    (args.out / "production_readiness.md").write_text(report)
    print(f"wrote {args.out / 'production_readiness.md'}")
    print(f"wrote {args.out / 'validation_rows.csv'}")


if __name__ == "__main__":
    main()
