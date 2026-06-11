"""Build a blind ABX listening pack: model IR vs captured reference IR.

Each trial renders the same dry test clip through:
  - the generated model IR for a captured grid position
  - the measured captured IR at that same grid position

The script randomizes A/B assignment and X, level-matches the rendered clips,
and writes a public manifest plus a private answer key.

Usage:
    python scripts/build_abx_listening_pack.py \
      --ckpt runs/real_pcond_h256/best.pt \
      --audio data/test_audio/inputs/jazz-hop-guitar.wav \
      --out runs/abx_listening_pack
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir import SR  # noqa: E402
from cabir.dsp import normalize_peak, resample_to, to_mono  # noqa: E402
from cabir.infer import generate_real_ir, load_real_checkpoint  # noqa: E402


def _read_audio(path: Path, max_seconds: float) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    audio = to_mono(audio)
    audio = resample_to(audio, int(sr), SR)
    if max_seconds > 0:
        audio = audio[: int(max_seconds * SR)]
    audio = audio - float(np.mean(audio))
    peak = float(np.max(np.abs(audio)))
    return audio / peak * 0.8 if peak > 0 else audio


def _render(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    n_full = len(audio) + len(ir) - 1
    n_fft = 1 << (n_full - 1).bit_length()
    wet = np.fft.irfft(np.fft.rfft(audio, n_fft) * np.fft.rfft(ir, n_fft), n_fft)[: len(audio)]
    wet = wet - float(np.mean(wet))
    return wet.astype(np.float64)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _level_match_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = (_rms(a) + _rms(b)) * 0.5
    a = a * (target / _rms(a))
    b = b * (target / _rms(b))
    peak = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-12)
    ceiling = 10 ** (-1.0 / 20.0)
    if peak > ceiling:
        gain = ceiling / peak
        a *= gain
        b *= gain
    return a.astype(np.float32), b.astype(np.float32)


def _presence_for_checkpoint(ckpt: dict) -> float | None:
    presence = ckpt.get("presence")
    if presence is None and ckpt.get("condition_presence"):
        return 3.0
    return presence


def _eval_rows(labels: pd.DataFrame, ckpt: dict, cab: str) -> pd.DataFrame:
    mask = (
        (labels["cab"] == cab)
        & (labels["mic"].isin(ckpt["mics"]))
        & (labels["capture_type"] == "close")
        & (labels["ts"] == bool(ckpt.get("ts", False)))
    )
    presence = _presence_for_checkpoint(ckpt)
    if presence is not None:
        mask &= labels["presence"] == float(presence)
    return labels[mask].copy().reset_index(drop=True)


def _filter_rows_for_focus(rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.mics:
        rows = rows[rows["mic"].isin(args.mics)]
    if args.min_offset_mm is not None:
        rows = rows[rows["offset_mm"].astype(float) >= args.min_offset_mm]
    if args.max_offset_mm is not None:
        rows = rows[rows["offset_mm"].astype(float) <= args.max_offset_mm]
    if args.min_distance_mm is not None:
        rows = rows[rows["distance_mm"].astype(float) >= args.min_distance_mm]
    if args.max_distance_mm is not None:
        rows = rows[rows["distance_mm"].astype(float) <= args.max_distance_mm]
    return rows.reset_index(drop=True)


def _pick_rows(rows: pd.DataFrame, trials: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    picked = []
    by_mic = {mic: group.copy() for mic, group in rows.groupby("mic")}
    mic_names = sorted(by_mic)
    while len(picked) < min(trials, len(rows)):
        for mic in mic_names:
            if len(picked) >= min(trials, len(rows)):
                break
            group = by_mic[mic]
            remaining = group[~group.index.isin([p.name for p in picked])]
            if remaining.empty:
                continue
            idx = rng.choice(remaining.index.tolist())
            picked.append(rows.loc[idx])
    return pd.DataFrame(picked).reset_index(drop=True)


def build_pack(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ckpt, model = load_real_checkpoint(args.ckpt)
    labels = pd.read_parquet(args.labels)
    irs = np.load(args.irs, mmap_mode="r")
    cab = args.cab or ckpt["cabs"][0]
    rows = _filter_rows_for_focus(_eval_rows(labels, ckpt, cab), args)
    rows = _pick_rows(rows, args.trials, args.seed)
    if rows.empty:
        raise SystemExit("No rows found for ABX pack.")

    audio = _read_audio(args.audio, args.max_seconds)
    args.out.mkdir(parents=True, exist_ok=True)

    public_rows: list[dict] = []
    answer_rows: list[dict] = []
    x_answers = ["A"] * ((len(rows) + 1) // 2) + ["B"] * (len(rows) // 2)
    rng.shuffle(x_answers)
    for trial_idx, row in enumerate(rows.itertuples(index=False), start=1):
        trial = f"{trial_idx:03d}"
        trial_dir = args.out / trial
        trial_dir.mkdir(parents=True, exist_ok=True)

        presence = float(row.presence) if not math.isnan(float(row.presence)) else _presence_for_checkpoint(ckpt)
        model_ir = generate_real_ir(
            model,
            ckpt,
            cab=cab,
            mic=str(row.mic),
            distance_mm=float(row.distance_mm),
            offset_mm=float(row.offset_mm),
            angle_deg=float(row.angle_deg),
            presence=presence,
        )
        captured_ir = np.asarray(irs[int(row.index)], dtype=np.float64)

        model_wet = _render(audio, model_ir)
        captured_wet = _render(audio, captured_ir)
        model_wet, captured_wet = _level_match_pair(model_wet, captured_wet)

        sources = {"model": model_wet, "captured": captured_wet}
        a_source, b_source = ("model", "captured") if rng.random() < 0.5 else ("captured", "model")
        x_answer = x_answers[trial_idx - 1]
        x_source = a_source if x_answer == "A" else b_source

        sf.write(trial_dir / "A.wav", sources[a_source], SR, subtype="PCM_24")
        sf.write(trial_dir / "B.wav", sources[b_source], SR, subtype="PCM_24")
        sf.write(trial_dir / "X.wav", sources[x_source], SR, subtype="PCM_24")

        public_rows.append(
            {
                "trial": trial,
                "mic": row.mic,
                "distance_mm": float(row.distance_mm),
                "offset_mm": float(row.offset_mm),
                "angle_deg": float(row.angle_deg),
                "presence": presence,
                "audio": str(args.audio),
                "notes": "Decide whether X matches A or B. Answer key is separate.",
            }
        )
        answer_rows.append(
            {
                "trial": trial,
                "A_source": a_source,
                "B_source": b_source,
                "X_source": x_source,
                "X_answer": x_answer,
                "captured_file": row.file,
            }
        )

    with (args.out / "manifest_public.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0]))
        writer.writeheader()
        writer.writerows(public_rows)
    with (args.out / "answer_key.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(answer_rows[0]))
        writer.writeheader()
        writer.writerows(answer_rows)
    with (args.out / "listener_results_template.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial", "listener", "guess"])
        writer.writeheader()
        for row in public_rows:
            writer.writerow({"trial": row["trial"], "listener": "", "guess": ""})
    (args.out / "README.md").write_text(
        "# ABX listening pack\n\n"
        "For each numbered trial, listen to `A.wav`, `B.wav`, and `X.wav`.\n"
        "Write down whether X matches A or B in `listener_results_template.csv` "
        "before opening `answer_key.csv`.\n"
        "A and B are randomized model-vs-captured renders of the same dry clip, "
        "level-matched to reduce loudness bias.\n"
    )
    print(f"wrote {len(public_rows)} ABX trials to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/real_pcond_h256/best.pt")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/parsed/labels.parquet")
    ap.add_argument("--irs", type=Path, default=REPO_ROOT / "data/parsed/irs.npy")
    ap.add_argument("--audio", type=Path, default=REPO_ROOT / "data/test_audio/inputs/jazz-hop-guitar.wav")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "runs/abx_listening_pack")
    ap.add_argument("--cab", default=None)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-seconds", type=float, default=12.0)
    ap.add_argument("--mics", nargs="+", default=None, help="optional mic subset, e.g. --mics sm57")
    ap.add_argument("--min-offset-mm", type=float, default=None, help="optional focus filter")
    ap.add_argument("--max-offset-mm", type=float, default=None, help="optional focus filter")
    ap.add_argument("--min-distance-mm", type=float, default=None, help="optional focus filter")
    ap.add_argument("--max-distance-mm", type=float, default=None, help="optional focus filter")
    args = ap.parse_args()
    build_pack(args)


if __name__ == "__main__":
    main()
