"""Compare candidate checkpoints against the current release checkpoint.

The gate reads already-generated reports:
  - production_report*/validation_rows.csv
  - internal_trend_validation*/trend_metrics.csv
  - tone3000_external_validation*/trend_metrics.csv

It does not retrain or re-run validation. Generate missing reports first with:
  scripts/production_readiness_report.py
  scripts/internal_trend_validation.py
  scripts/tone3000_external_validation.py

Usage:
    python scripts/checkpoint_promotion_gate.py
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class RunSpec:
    name: str
    ckpt: Path
    production: Path
    internal_trend: Path
    external_trend: Path


DEFAULT_RUNS = [
    RunSpec(
        "current",
        REPO_ROOT / "runs/real_pcond_h256/best.pt",
        REPO_ROOT / "runs/production_report/validation_rows.csv",
        REPO_ROOT / "runs/internal_trend_validation/trend_metrics.csv",
        REPO_ROOT / "runs/tone3000_external_validation/trend_metrics.csv",
    ),
    RunSpec(
        "trend2k5_w020",
        REPO_ROOT / "runs/real_pcond_h256_trend2k5_sm57/best.pt",
        REPO_ROOT / "runs/production_report_trend2k5_sm57/validation_rows.csv",
        REPO_ROOT / "runs/internal_trend_validation_trend2k5_sm57/trend_metrics.csv",
        REPO_ROOT / "runs/tone3000_external_validation_trend2k5_sm57/trend_metrics.csv",
    ),
    RunSpec(
        "trend2k5_w005",
        REPO_ROOT / "runs/real_pcond_h256_trend2k5_sm57_w005/best.pt",
        REPO_ROOT / "runs/production_report_trend2k5_sm57_w005/validation_rows.csv",
        REPO_ROOT / "runs/internal_trend_validation_trend2k5_sm57_w005/trend_metrics.csv",
        REPO_ROOT / "runs/tone3000_external_validation_trend2k5_sm57_w005/trend_metrics.csv",
    ),
    RunSpec(
        "trend2k10_w005",
        REPO_ROOT / "runs/real_pcond_h256_trend2k10_sm57_w005/best.pt",
        REPO_ROOT / "runs/production_report_trend2k10_sm57_w005/validation_rows.csv",
        REPO_ROOT / "runs/internal_trend_validation_trend2k10_sm57_w005/trend_metrics.csv",
        REPO_ROOT / "runs/tone3000_external_validation_trend2k10_sm57_w005/trend_metrics.csv",
    ),
]


def _require(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing required report file: {path}")


def _trend_stats(path: Path, mic: str = "sm57") -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[(df["mic"] == mic) & (df["offset_mm"] != df["ref_offset_mm"])].copy()
    diff2 = df["model_2_5k_delta_db"] - df["real_2_5k_delta_db"]
    diff5 = df["model_5_10k_delta_db"] - df["real_5_10k_delta_db"]
    return {
        "sm57_trend_rms_db": float(df["trend_rms_100_6k_db"].mean()),
        "sm57_trend_corr": float(df["trend_corr_100_6k"].mean()),
        "sm57_abs_2_5k_diff_db": float(diff2.abs().mean()),
        "sm57_abs_5_10k_diff_db": float(diff5.abs().mean()),
        "sm57_bias_2_5k_diff_db": float(diff2.mean()),
        "sm57_bias_5_10k_diff_db": float(diff5.mean()),
    }


def _external_stats(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df = df[df["offset_mm"] != float(df["offset_mm"].min())].copy()
    end = pd.read_csv(path).sort_values("offset_mm").iloc[-1]
    return {
        "external_trend_rms_db": float(df["trend_rms_100_6k_db"].mean()),
        "external_trend_corr": float(df["trend_corr_100_6k"].mean()),
        "external_model_5_10k_end_db": float(end["model_5_10k_delta_db"]),
        "external_source_5_10k_end_db": float(end["source_5_10k_delta_db"]),
    }


def _production_stats(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    sm57 = df[df["mic"] == "sm57"]
    return {
        "production_mean_db": float(df["model_guitar_rms_db"].mean()),
        "production_worst_db": float(df["model_guitar_rms_db"].max()),
        "beat_nearest_rate": float(df["beats_nearest_other"].mean()),
        "sm57_mean_db": float(sm57["model_guitar_rms_db"].mean()),
        "sm57_worst_db": float(sm57["model_guitar_rms_db"].max()),
    }


def summarize_run(spec: RunSpec) -> dict[str, float | str]:
    for path in [spec.ckpt, spec.production, spec.internal_trend, spec.external_trend]:
        _require(path)
    row: dict[str, float | str] = {"name": spec.name, "checkpoint": str(spec.ckpt.relative_to(REPO_ROOT))}
    row.update(_production_stats(spec.production))
    row.update(_trend_stats(spec.internal_trend))
    row.update(_external_stats(spec.external_trend))
    return row


def _decision(row: pd.Series, base: pd.Series, args: argparse.Namespace) -> tuple[str, str]:
    if row["name"] == base["name"]:
        return "baseline", "current release checkpoint"

    checks = []
    checks.append((row["production_mean_db"] <= base["production_mean_db"] + args.max_mean_regression_db, "production mean regression"))
    checks.append((row["production_worst_db"] <= base["production_worst_db"] + args.max_worst_regression_db, "production worst regression"))
    checks.append((row["beat_nearest_rate"] >= args.min_beat_rate, "beat-nearest rate"))
    checks.append((row["sm57_worst_db"] <= base["sm57_worst_db"] - args.min_sm57_worst_improvement_db, "SM57 worst improvement"))
    checks.append((row["sm57_abs_2_5k_diff_db"] <= base["sm57_abs_2_5k_diff_db"] - args.min_sm57_2_5k_improvement_db, "SM57 2-5k trend improvement"))
    checks.append((row["sm57_abs_5_10k_diff_db"] <= base["sm57_abs_5_10k_diff_db"] + args.max_sm57_5_10k_regression_db, "SM57 5-10k trend regression"))
    checks.append((row["external_trend_rms_db"] <= base["external_trend_rms_db"] + args.max_external_regression_db, "external trend regression"))

    failed = [label for ok, label in checks if not ok]
    if failed:
        return "reject", "; ".join(failed)
    return "promote", "passes all promotion gates"


def build_report(args: argparse.Namespace) -> str:
    rows = [summarize_run(spec) for spec in DEFAULT_RUNS]
    df = pd.DataFrame(rows)
    base = df[df["name"] == args.baseline].iloc[0]
    decisions = [_decision(row, base, args) for _, row in df.iterrows()]
    df["decision"] = [d for d, _ in decisions]
    df["reason"] = [r for _, r in decisions]

    metric_cols = [
        "name",
        "decision",
        "production_mean_db",
        "production_worst_db",
        "beat_nearest_rate",
        "sm57_worst_db",
        "sm57_abs_2_5k_diff_db",
        "sm57_abs_5_10k_diff_db",
        "external_trend_rms_db",
        "external_trend_corr",
        "reason",
    ]
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "checkpoint_promotion_gate.csv", index=False)

    lines = [
        "# Checkpoint promotion gate",
        "",
        f"- Baseline: `{args.baseline}`",
        f"- Candidates: `{', '.join(x.name for x in DEFAULT_RUNS if x.name != args.baseline)}`",
        "",
        "## Promotion Rules",
        "",
        f"- Production mean may regress by at most `{args.max_mean_regression_db:.3f} dB`.",
        f"- Production worst may regress by at most `{args.max_worst_regression_db:.3f} dB`.",
        f"- Beat-nearest rate must be at least `{args.min_beat_rate * 100:.1f}%`.",
        f"- SM57 worst must improve by at least `{args.min_sm57_worst_improvement_db:.3f} dB`.",
        f"- SM57 2-5 kHz trend abs diff must improve by at least `{args.min_sm57_2_5k_improvement_db:.3f} dB`.",
        f"- SM57 5-10 kHz trend abs diff may regress by at most `{args.max_sm57_5_10k_regression_db:.3f} dB`.",
        f"- TONE3000 external trend RMS may regress by at most `{args.max_external_regression_db:.3f} dB`.",
        "",
        "## Result",
        "",
        df[metric_cols].to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Recommendation",
        "",
    ]
    promoted = df[df["decision"] == "promote"]
    if promoted.empty:
        lines.append(f"Keep `{args.baseline}` as the release checkpoint. No candidate passes all gates.")
    else:
        best = promoted.sort_values(["production_mean_db", "sm57_worst_db"]).iloc[0]
        lines.append(f"Promote `{best['name']}`: `{best['checkpoint']}`.")
    lines.append("")
    report = "\n".join(lines)
    (out / "checkpoint_promotion_gate.md").write_text(report)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs" / "checkpoint_promotion_gate")
    ap.add_argument("--baseline", default="current")
    ap.add_argument("--max-mean-regression-db", type=float, default=0.02)
    ap.add_argument("--max-worst-regression-db", type=float, default=0.0)
    ap.add_argument("--min-beat-rate", type=float, default=1.0)
    ap.add_argument("--min-sm57-worst-improvement-db", type=float, default=0.05)
    ap.add_argument("--min-sm57-2-5k-improvement-db", type=float, default=0.05)
    ap.add_argument("--max-sm57-5-10k-regression-db", type=float, default=0.05)
    ap.add_argument("--max-external-regression-db", type=float, default=0.05)
    print(build_report(ap.parse_args()))


if __name__ == "__main__":
    main()
