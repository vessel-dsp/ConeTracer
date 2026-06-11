"""Score ABX listener guesses against an answer key.

Usage:
    python scripts/score_abx_results.py \
      --answers runs/abx_listening_pack/answer_key.csv \
      --results runs/abx_listening_pack/listener_results_template.csv
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def _binomial_tail(k: int, n: int, p: float = 0.5) -> float:
    return float(sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()

    answers = pd.read_csv(args.answers, dtype={"trial": str})
    results = pd.read_csv(args.results, dtype={"trial": str})
    answers["trial"] = answers["trial"].astype(str).str.strip().str.zfill(3)
    results["trial"] = results["trial"].astype(str).str.strip().str.zfill(3)
    if "guess" not in results.columns:
        raise SystemExit("results CSV must contain a 'guess' column with A or B")
    merged = results.merge(answers[["trial", "X_answer"]], on="trial", how="inner")
    merged["guess"] = merged["guess"].astype(str).str.upper().str.strip()
    merged = merged[merged["guess"].isin(["A", "B"])].copy()
    if merged.empty:
        raise SystemExit("No scored guesses found. Fill guess with A or B.")
    merged["correct"] = merged["guess"] == merged["X_answer"]
    correct = int(merged["correct"].sum())
    total = int(len(merged))
    p_value = _binomial_tail(correct, total)
    print(f"scored: {total}")
    print(f"correct: {correct}")
    print(f"accuracy: {correct / total * 100:.1f}%")
    print(f"one-sided binomial p-value vs chance: {p_value:.4f}")
    if "listener" in merged.columns:
        print("\nby listener:")
        for listener, group in merged.groupby("listener", dropna=False):
            name = str(listener) if str(listener).strip() else "(blank)"
            c = int(group["correct"].sum())
            n = int(len(group))
            print(f"  {name}: {c}/{n} ({c / n * 100:.1f}%), p={_binomial_tail(c, n):.4f}")


if __name__ == "__main__":
    main()
