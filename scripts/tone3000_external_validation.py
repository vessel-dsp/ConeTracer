"""External trend validation against a small approved TONE3000 probe.

This does not score TONE3000 as same-cab ground truth. Instead it compares the
relative cap-to-edge spectral trend in the model against the relative trend in a
clean TONE3000 SM57 offset line.

Usage:
    python scripts/tone3000_external_validation.py \
      --ckpt runs/real_pcond_h256/best.pt \
      --labels data/parsed_tone3000_probe/labels.parquet \
      --irs data/parsed_tone3000_probe/irs.npy \
      --out runs/tone3000_external_validation
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
    db = 20 * np.log10(np.maximum(mag, 1e-8))
    return freqs, db


def _gain_normalize_db(freqs: np.ndarray, db: np.ndarray, lo: float = 100.0, hi: float = 6000.0) -> np.ndarray:
    band = (freqs >= lo) & (freqs <= hi)
    ref = float(np.median(db[band])) if np.any(band) else float(np.median(db))
    return db - ref


def _band_means(freqs: np.ndarray, delta_db: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
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


def _trend_rows(
    offsets: list[float],
    source_irs: dict[float, np.ndarray],
    model_irs: dict[float, np.ndarray],
    ref_offset: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    freqs, source_ref = _mag_db(source_irs[ref_offset])
    _, model_ref = _mag_db(model_irs[ref_offset])
    source_ref = _gain_normalize_db(freqs, source_ref)
    model_ref = _gain_normalize_db(freqs, model_ref)

    rows: list[dict] = []
    band_rows: list[dict] = []
    guitar_band = (freqs >= 100) & (freqs <= 6000)
    wide_band = (freqs >= 100) & (freqs <= 12000)

    for offset in offsets:
        _, source_db = _mag_db(source_irs[offset])
        _, model_db = _mag_db(model_irs[offset])
        source_delta = _gain_normalize_db(freqs, source_db) - source_ref
        model_delta = _gain_normalize_db(freqs, model_db) - model_ref
        diff = model_delta - source_delta
        source_bands = _band_means(freqs, source_delta)
        model_bands = _band_means(freqs, model_delta)
        rows.append(
            {
                "offset_mm": offset,
                "trend_rms_100_6k_db": float(np.sqrt(np.mean(diff[guitar_band] ** 2))),
                "trend_rms_100_12k_db": float(np.sqrt(np.mean(diff[wide_band] ** 2))),
                "trend_corr_100_6k": _corr(model_delta[guitar_band], source_delta[guitar_band]),
                "source_5_10k_delta_db": source_bands["5-10k"],
                "model_5_10k_delta_db": model_bands["5-10k"],
            }
        )
        for band_name, _, _ in BANDS:
            band_rows.append(
                {
                    "offset_mm": offset,
                    "band": band_name,
                    "source_delta_db": source_bands[band_name],
                    "model_delta_db": model_bands[band_name],
                    "model_minus_source_db": model_bands[band_name] - source_bands[band_name],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(band_rows)


def _select_clean_rows(labels: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = labels[
        (labels["pack"] == args.pack)
        & (labels["mic"] == args.mic)
        & (labels["capture_type"] == "close")
        & (labels["distance_mm"].astype(float) == float(args.distance_mm))
        & (labels["angle_deg"].astype(float) == float(args.angle_deg))
    ].copy()
    if args.file_include:
        rows = rows[rows["file"].str.contains(args.file_include, regex=True)]
    if args.file_exclude:
        rows = rows[~rows["file"].str.contains(args.file_exclude, regex=True)]
    return rows.sort_values("offset_mm").reset_index(drop=True)


def _write_plot(out_path: Path, band_df: pd.DataFrame) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    axes = axes.ravel()
    for ax, (band_name, _, _) in zip(axes, BANDS):
        g = band_df[band_df["band"] == band_name].sort_values("offset_mm")
        ax.plot(g["offset_mm"], g["source_delta_db"], marker="o", label="TONE3000")
        ax.plot(g["offset_mm"], g["model_delta_db"], marker="o", label="model")
        ax.axhline(0, color="#999999", linewidth=0.8)
        ax.set_title(band_name)
        ax.set_xlabel("offset mm")
        ax.set_ylabel("delta dB")
        ax.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Cap-center-relative spectral trend: model vs TONE3000 probe")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return None


def build_report(args: argparse.Namespace) -> str:
    args.out.mkdir(parents=True, exist_ok=True)
    labels = pd.read_parquet(args.labels)
    irs = np.load(args.irs, mmap_mode="r")
    rows = _select_clean_rows(labels, args)
    if rows.empty:
        raise SystemExit("No TONE3000 validation rows matched the requested filters.")

    offsets = [float(x) for x in rows["offset_mm"].tolist()]
    ref_offset = min(offsets)
    source_irs = {
        float(row.offset_mm): np.asarray(irs[int(row["index"])], dtype=np.float64)
        for _, row in rows.iterrows()
    }

    ckpt, model = load_real_checkpoint(args.ckpt)
    cab = args.cab or ckpt["cabs"][0]
    model_irs = {
        offset: generate_real_ir(
            model,
            ckpt,
            cab=cab,
            mic=args.mic,
            distance_mm=float(args.distance_mm),
            offset_mm=offset,
            angle_deg=float(args.angle_deg),
            presence=args.presence,
        )
        for offset in offsets
    }

    trend_df, band_df = _trend_rows(offsets, source_irs, model_irs, ref_offset)
    trend_df.to_csv(args.out / "trend_metrics.csv", index=False)
    band_df.to_csv(args.out / "band_trends.csv", index=False)
    rows.to_csv(args.out / "selected_tone3000_rows.csv", index=False)
    plot_error = _write_plot(args.out / "band_trends.png", band_df)
    if plot_error is not None:
        plot_error = plot_error.replace("`", "'")

    non_ref = trend_df[trend_df["offset_mm"] != ref_offset]
    mean_rms = float(non_ref["trend_rms_100_6k_db"].mean()) if not non_ref.empty else float("nan")
    mean_corr = float(non_ref["trend_corr_100_6k"].mean()) if not non_ref.empty else float("nan")

    source_hf_slope = float(
        band_df[band_df["band"] == "5-10k"].sort_values("offset_mm")["source_delta_db"].iloc[-1]
        - band_df[band_df["band"] == "5-10k"].sort_values("offset_mm")["source_delta_db"].iloc[0]
    )
    model_hf_slope = float(
        band_df[band_df["band"] == "5-10k"].sort_values("offset_mm")["model_delta_db"].iloc[-1]
        - band_df[band_df["band"] == "5-10k"].sort_values("offset_mm")["model_delta_db"].iloc[0]
    )

    lines = [
        "# TONE3000 external trend validation",
        "",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Model cab used: `{cab}`",
        f"- External pack: `{args.pack}`",
        f"- Rows selected: `{len(rows)}`",
        f"- Offsets: `{offsets}`",
        f"- Reference offset: `{ref_offset} mm`",
        "",
        "## Interpretation",
        "",
        "This is not a same-cab accuracy score. The external IRs come from a different source/chain, so the report compares cap-center-relative spectral trends only.",
        "",
        "## Summary",
        "",
        f"- Mean trend RMS, 100 Hz-6 kHz: **{mean_rms:.2f} dB**",
        f"- Mean trend correlation, 100 Hz-6 kHz: **{mean_corr:.2f}**",
        f"- TONE3000 5-10 kHz end-to-end change: **{source_hf_slope:.2f} dB**",
        f"- Model 5-10 kHz end-to-end change: **{model_hf_slope:.2f} dB**",
        "",
        "## Per Offset",
        "",
        trend_df.to_markdown(index=False, floatfmt=".2f"),
        "",
        "## Files",
        "",
        "- `trend_metrics.csv`",
        "- `band_trends.csv`",
        "- `selected_tone3000_rows.csv`",
        "- `band_trends.png`" if plot_error is None else f"- Plot skipped: `{plot_error}`",
        "",
    ]
    report = "\n".join(lines)
    (args.out / "tone3000_external_validation.md").write_text(report + "\n")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs" / "real_pcond_h256" / "best.pt")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data" / "parsed_tone3000_probe" / "labels.parquet")
    ap.add_argument("--irs", type=Path, default=REPO_ROOT / "data" / "parsed_tone3000_probe" / "irs.npy")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "tone3000_external_validation")
    ap.add_argument("--pack", default="tone3000_45023")
    ap.add_argument("--cab", default=None)
    ap.add_argument("--mic", default="sm57")
    ap.add_argument("--distance-mm", type=float, default=0.0)
    ap.add_argument("--angle-deg", type=float, default=0.0)
    ap.add_argument("--presence", type=float, default=3.0)
    ap.add_argument("--file-include", default=r"V30 LL 4FB 4x12 SM57")
    ap.add_argument("--file-exclude", default=r"OA")
    args = ap.parse_args()
    report = build_report(args)
    print(report)


if __name__ == "__main__":
    main()
