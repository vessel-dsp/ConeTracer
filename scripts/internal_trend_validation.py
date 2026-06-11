"""Internal cap-to-edge trend validation against captured grid rows.

Unlike the TONE3000 external trend check, this compares the model to captured
IRs from the same cab/mic/distance/presence grid used for validation. It asks:
when real captures move from cap to edge/cone, does the model move the same
spectral bands by the same amount?

Usage:
    python scripts/internal_trend_validation.py \
      --ckpt runs/real_pcond_h256/best.pt \
      --out runs/internal_trend_validation
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir import N_TAPS, SR  # noqa: E402
from cabir.infer import generate_real_ir, load_real_checkpoint  # noqa: E402

BANDS = [
    ("80-150", 80.0, 150.0),
    ("150-300", 150.0, 300.0),
    ("300-800", 300.0, 800.0),
    ("0.8-2k", 800.0, 2000.0),
    ("2-5k", 2000.0, 5000.0),
    ("5-10k", 5000.0, 10000.0),
]


def _mag_db(ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(N_TAPS, 1.0 / SR)
    mag = np.abs(np.fft.rfft(ir, n=N_TAPS))
    return freqs, 20 * np.log10(np.maximum(mag, 1e-8))


def _gain_normalize_db(freqs: np.ndarray, db: np.ndarray, lo: float = 100.0, hi: float = 6000.0) -> np.ndarray:
    band = (freqs >= lo) & (freqs <= hi)
    ref = float(np.median(db[band])) if np.any(band) else float(np.median(db))
    return db - ref


def _band_means(freqs: np.ndarray, delta_db: np.ndarray) -> dict[str, float]:
    out = {}
    for name, lo, hi in BANDS:
        mask = (freqs >= lo) & (freqs <= hi)
        out[name] = float(np.mean(delta_db[mask])) if np.any(mask) else float("nan")
    return out


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    aa = a[mask] - float(np.mean(a[mask]))
    bb = b[mask] - float(np.mean(b[mask]))
    denom = float(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)))
    return float(np.sum(aa * bb) / denom) if denom else float("nan")


def _presence_for_checkpoint(ckpt: dict) -> float | None:
    presence = ckpt.get("presence")
    if presence is None and ckpt.get("condition_presence"):
        return 3.0
    return presence


def _eval_rows(labels: pd.DataFrame, ckpt: dict, cab: str) -> pd.DataFrame:
    rows = labels[
        (labels["cab"] == cab)
        & (labels["mic"].isin(ckpt["mics"]))
        & (labels["capture_type"] == "close")
        & (labels["ts"] == bool(ckpt.get("ts", False)))
    ].copy()
    presence = _presence_for_checkpoint(ckpt)
    if presence is not None:
        rows = rows[rows["presence"] == float(presence)]
    return rows


def _line_metrics(
    line: pd.DataFrame,
    irs: np.ndarray,
    model,
    ckpt: dict,
    cab: str,
    mic: str,
    presence: float | None,
) -> tuple[list[dict], list[dict]]:
    line = line.sort_values("offset_mm").reset_index(drop=True)
    ref = line.iloc[0]
    offsets = [float(x) for x in line["offset_mm"].tolist()]
    ref_offset = float(ref.offset_mm)
    freqs, real_ref = _mag_db(np.asarray(irs[int(ref["index"])], dtype=np.float64))
    real_ref = _gain_normalize_db(freqs, real_ref)
    model_ref_ir = generate_real_ir(
        model,
        ckpt,
        cab=cab,
        mic=mic,
        distance_mm=float(ref.distance_mm),
        offset_mm=ref_offset,
        angle_deg=float(ref.angle_deg),
        presence=presence,
    )
    _, model_ref = _mag_db(model_ref_ir)
    model_ref = _gain_normalize_db(freqs, model_ref)

    rows: list[dict] = []
    band_rows: list[dict] = []
    guitar_band = (freqs >= 100) & (freqs <= 6000)
    wide_band = (freqs >= 100) & (freqs <= 12000)
    for _, row in line.iterrows():
        offset = float(row.offset_mm)
        real_ir = np.asarray(irs[int(row["index"])], dtype=np.float64)
        model_ir = generate_real_ir(
            model,
            ckpt,
            cab=cab,
            mic=mic,
            distance_mm=float(row.distance_mm),
            offset_mm=offset,
            angle_deg=float(row.angle_deg),
            presence=presence,
        )
        _, real_db = _mag_db(real_ir)
        _, model_db = _mag_db(model_ir)
        real_delta = _gain_normalize_db(freqs, real_db) - real_ref
        model_delta = _gain_normalize_db(freqs, model_db) - model_ref
        diff = model_delta - real_delta
        real_bands = _band_means(freqs, real_delta)
        model_bands = _band_means(freqs, model_delta)
        rows.append(
            {
                "mic": mic,
                "distance_mm": float(row.distance_mm),
                "angle_deg": float(row.angle_deg),
                "offset_mm": offset,
                "ref_offset_mm": ref_offset,
                "trend_rms_100_6k_db": float(np.sqrt(np.mean(diff[guitar_band] ** 2))),
                "trend_rms_100_12k_db": float(np.sqrt(np.mean(diff[wide_band] ** 2))),
                "trend_corr_100_6k": _corr(model_delta[guitar_band], real_delta[guitar_band]),
                "real_2_5k_delta_db": real_bands["2-5k"],
                "model_2_5k_delta_db": model_bands["2-5k"],
                "real_5_10k_delta_db": real_bands["5-10k"],
                "model_5_10k_delta_db": model_bands["5-10k"],
                "file": row.file,
            }
        )
        for band_name, _, _ in BANDS:
            band_rows.append(
                {
                    "mic": mic,
                    "distance_mm": float(row.distance_mm),
                    "offset_mm": offset,
                    "band": band_name,
                    "real_delta_db": real_bands[band_name],
                    "model_delta_db": model_bands[band_name],
                    "model_minus_real_db": model_bands[band_name] - real_bands[band_name],
                }
            )
    return rows, band_rows


def _write_plot(path: Path, band_df: pd.DataFrame) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc).replace("`", "'")

    focus = band_df[band_df["band"].isin(["2-5k", "5-10k"])].copy()
    lines = list(focus.groupby(["mic", "distance_mm", "band"]))
    if not lines:
        return None
    fig, axes = plt.subplots(len(lines), 1, figsize=(8, max(3, len(lines) * 2.2)), squeeze=False)
    for ax, ((mic, distance, band), g) in zip(axes.ravel(), lines):
        g = g.sort_values("offset_mm")
        ax.plot(g["offset_mm"], g["real_delta_db"], marker="o", label="captured")
        ax.plot(g["offset_mm"], g["model_delta_db"], marker="o", label="model")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_title(f"{mic} {distance:g} mm {band}")
        ax.set_xlabel("offset mm")
        ax.set_ylabel("delta dB")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return None


def build_report(args: argparse.Namespace) -> str:
    args.out.mkdir(parents=True, exist_ok=True)
    ckpt, model = load_real_checkpoint(args.ckpt)
    labels = pd.read_parquet(args.labels)
    irs = np.load(args.irs, mmap_mode="r")
    cab = args.cab or ckpt["cabs"][0]
    presence = args.presence if args.presence is not None else _presence_for_checkpoint(ckpt)
    rows = _eval_rows(labels, ckpt, cab)

    metric_rows: list[dict] = []
    band_rows: list[dict] = []
    for (mic, distance, angle), line in rows.groupby(["mic", "distance_mm", "angle_deg"]):
        if len(line["offset_mm"].unique()) < 2:
            continue
        m_rows, b_rows = _line_metrics(line, irs, model, ckpt, cab, str(mic), presence)
        metric_rows.extend(m_rows)
        band_rows.extend(b_rows)

    metrics = pd.DataFrame(metric_rows)
    bands = pd.DataFrame(band_rows)
    if metrics.empty:
        raise SystemExit("No internal trend lines found.")

    metrics.to_csv(args.out / "trend_metrics.csv", index=False)
    bands.to_csv(args.out / "band_trends.csv", index=False)
    plot_error = _write_plot(args.out / "band_trends_focus.png", bands)

    non_ref = metrics[metrics["offset_mm"] != metrics["ref_offset_mm"]]
    summary = (
        non_ref.groupby("mic")
        .agg(
            rows=("file", "count"),
            mean_trend_rms_100_6k_db=("trend_rms_100_6k_db", "mean"),
            mean_trend_corr_100_6k=("trend_corr_100_6k", "mean"),
            mean_2_5k_diff_db=("model_2_5k_delta_db", lambda s: float(np.mean(s.to_numpy() - non_ref.loc[s.index, "real_2_5k_delta_db"].to_numpy()))),
            mean_abs_2_5k_diff_db=("model_2_5k_delta_db", lambda s: float(np.mean(np.abs(s.to_numpy() - non_ref.loc[s.index, "real_2_5k_delta_db"].to_numpy())))),
            mean_5_10k_diff_db=("model_5_10k_delta_db", lambda s: float(np.mean(s.to_numpy() - non_ref.loc[s.index, "real_5_10k_delta_db"].to_numpy()))),
            mean_abs_5_10k_diff_db=("model_5_10k_delta_db", lambda s: float(np.mean(np.abs(s.to_numpy() - non_ref.loc[s.index, "real_5_10k_delta_db"].to_numpy())))),
        )
        .reset_index()
    )

    worst = non_ref.copy()
    worst["abs_2_5k_diff_db"] = (worst["model_2_5k_delta_db"] - worst["real_2_5k_delta_db"]).abs()
    worst["abs_5_10k_diff_db"] = (worst["model_5_10k_delta_db"] - worst["real_5_10k_delta_db"]).abs()

    lines = [
        "# Internal captured-grid trend validation",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Cab: `{cab}`",
        f"- Presence: `{presence}`",
        f"- Trend lines: `{metrics[['mic', 'distance_mm', 'angle_deg']].drop_duplicates().shape[0]}`",
        f"- Rows: `{len(metrics)}`",
        "",
        "## Interpretation",
        "",
        "This is same-domain validation: captured cap-to-edge spectral movement vs model cap-to-edge spectral movement at the same mic/distance grid points.",
        "",
        "## Summary By Mic",
        "",
        summary.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Worst 2-5k Trend Rows",
        "",
        worst.sort_values("abs_2_5k_diff_db", ascending=False)
        .head(args.worst_rows)[
            [
                "mic",
                "distance_mm",
                "offset_mm",
                "real_2_5k_delta_db",
                "model_2_5k_delta_db",
                "abs_2_5k_diff_db",
                "file",
            ]
        ]
        .to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Worst 5-10k Trend Rows",
        "",
        worst.sort_values("abs_5_10k_diff_db", ascending=False)
        .head(args.worst_rows)[
            [
                "mic",
                "distance_mm",
                "offset_mm",
                "real_5_10k_delta_db",
                "model_5_10k_delta_db",
                "abs_5_10k_diff_db",
                "file",
            ]
        ]
        .to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Files",
        "",
        "- `trend_metrics.csv`",
        "- `band_trends.csv`",
        "- `band_trends_focus.png`" if plot_error is None else f"- Plot skipped: `{plot_error}`",
        "",
    ]
    report = "\n".join(lines)
    (args.out / "internal_trend_validation.md").write_text(report + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs" / "real_pcond_h256" / "best.pt")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "parsed" / "labels.parquet")
    ap.add_argument("--irs", type=Path, default=REPO_ROOT / "data" / "parsed" / "irs.npy")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "internal_trend_validation")
    ap.add_argument("--cab", default=None)
    ap.add_argument("--presence", type=float, default=None)
    ap.add_argument("--worst-rows", type=int, default=8)
    args = ap.parse_args()
    print(build_report(args))


if __name__ == "__main__":
    main()
