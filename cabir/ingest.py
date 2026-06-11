"""Ingest raw IR packs -> normalized dataset (Workstream 1 / M1).

    python -m cabir.ingest [--raw data/raw] [--out data/parsed] [--minphase]

For each pack folder under ``--raw``:
  1. detect the matching parser in ``data/parsers/`` (by filename signatures),
  2. skip raw paths the parser's path filters exclude (duplicate sample rates,
     hardware-format dumps, deprecated folders),
  3. parse every remaining audio filename into a label (a parser may export a
     custom ``parse(rel, cfg)``; otherwise the generic ``parse_path`` is used).
     Unparseable files are quarantined, never guessed,
  4. normalize the audio: mono -> 48 kHz -> onset-align -> trim to 4096 ->
     window tail -> (optional min-phase) -> unit-energy float32.

Outputs to ``--out``:
  - ``labels.parquet``  one row per IR (schema = cabir.labels.LABEL_COLUMNS)
  - ``irs.npy``         float32 (N, 4096); row i <-> labels row with index i
  - ``coverage.md``     position-grid coverage per (cab, mic)
  - ``quarantine/<pack>.txt``  files that didn't yield a complete label, + why

Safe to run with an empty ``data/raw`` — it reports that nothing was found.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf

from . import N_TAPS, SR
from .dsp import align_onset, fit_length, normalize_ir, resample_to, to_mono
from .labels import (
    PackConfig,
    coverage_report,
    missing_fields,
    parse_path,
    write_parquet,
)

AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_parsers(parsers_dir: Path) -> list[tuple[PackConfig, callable]]:
    """Import every ``data/parsers/*.py``; return (CONFIG, parse_fn) pairs.
    A module may export a custom ``parse(rel, cfg)``; otherwise generic ``parse_path``."""
    parsers = []
    for f in sorted(parsers_dir.glob("*.py")):
        if f.stem.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"cabir_parser_{f.stem}", f)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = getattr(mod, "CONFIG", None)
        if isinstance(cfg, PackConfig):
            parse_fn = getattr(mod, "parse", None) or parse_path
            parsers.append((cfg, parse_fn))
    return parsers


def detect_pack(folder_name: str, parsers):
    for cfg, parse_fn in parsers:
        if cfg.matches(folder_name) or folder_name.lower() == cfg.pack:
            return cfg, parse_fn
    return parsers[0] if len(parsers) == 1 else None


def _normalize_audio(path: Path, minphase: bool) -> tuple[np.ndarray, int]:
    audio, src_sr = sf.read(str(path), dtype="float64", always_2d=False)
    ir = to_mono(np.asarray(audio))
    ir = resample_to(ir, src_sr, SR)
    ir = align_onset(ir)
    ir = fit_length(ir, N_TAPS)
    return normalize_ir(ir, minphase=minphase), int(src_sr)


def ingest(raw_dir: Path, parsers_dir: Path, out_dir: Path, *, minphase: bool = False) -> dict:
    parsers = load_parsers(parsers_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quarantine").mkdir(exist_ok=True)

    pack_dirs = [p for p in sorted(raw_dir.iterdir()) if p.is_dir()] if raw_dir.is_dir() else []
    rows: list[dict] = []
    irs: list[np.ndarray] = []
    unknown_packs: list[str] = []
    quarantined = 0
    total_skipped = 0

    print(f"parsers: {', '.join(c.pack for c, _ in parsers) or '(none)'}")
    print(f"scanning {raw_dir}/ — {len(pack_dirs)} pack folder(s)\n")

    for pack_dir in pack_dirs:
        match = detect_pack(pack_dir.name, parsers)
        if match is None:
            unknown_packs.append(pack_dir.name)
            print(f"  ?  {pack_dir.name}: no parser matched — add one in data/parsers/")
            continue
        cfg, parse_fn = match

        all_audio = [p for p in pack_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        ok, skipped, quar_lines = 0, 0, []
        for path in sorted(all_audio):
            rel = str(path.relative_to(raw_dir))
            if not cfg.path_ok(rel):
                skipped += 1
                continue
            label = parse_fn(rel, cfg)
            if label is None:
                quar_lines.append(f"{rel}\t(missing: {', '.join(missing_fields(rel, cfg)) or '?'})")
                continue
            ir, src_sr = _normalize_audio(path, minphase)
            row = label.as_row()
            row.update(index=len(rows), sr=SR, n_samples=N_TAPS,
                       src_sr=src_sr, is_minphase=minphase)
            rows.append(row)
            irs.append(ir)
            ok += 1

        quarantined += len(quar_lines)
        total_skipped += skipped
        if quar_lines:
            (out_dir / "quarantine" / f"{cfg.pack}.txt").write_text("\n".join(quar_lines) + "\n")
        print(f"  ✓  {pack_dir.name} [{cfg.pack}]: {ok} parsed, "
              f"{len(quar_lines)} quarantined, {skipped} skipped (path filter)")

    # Write outputs (always, even when empty, so downstream paths exist).
    df = write_parquet(rows, out_dir / "labels.parquet")
    ir_array = np.stack(irs).astype(np.float32) if irs else np.zeros((0, N_TAPS), np.float32)
    np.save(out_dir / "irs.npy", ir_array)
    (out_dir / "coverage.md").write_text(coverage_report(df))

    print(f"\nlabels.parquet : {len(rows)} rows -> {out_dir / 'labels.parquet'}")
    print(f"irs.npy        : {ir_array.shape} float32 -> {out_dir / 'irs.npy'}")
    print(f"coverage.md    : {out_dir / 'coverage.md'}")
    if quarantined:
        print(f"quarantined    : {quarantined} file(s) -> {out_dir / 'quarantine'}/ (fix tokens, re-run)")
    if unknown_packs:
        print(f"unknown packs  : {', '.join(unknown_packs)} (no parser matched)")
    if not pack_dirs:
        print(f"\nNothing to ingest. Download packs into {raw_dir}/ — see {raw_dir / 'README.md'}.")

    return {"parsed": len(rows), "quarantined": quarantined,
            "skipped": total_skipped, "unknown_packs": unknown_packs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--parsers", type=Path, default=REPO_ROOT / "data" / "parsers")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "data" / "parsed")
    ap.add_argument("--minphase", action="store_true",
                    help="store the min-phase reconstruction as the target (sets is_minphase)")
    args = ap.parse_args()
    ingest(args.raw, args.parsers, args.out, minphase=args.minphase)


if __name__ == "__main__":
    main()
