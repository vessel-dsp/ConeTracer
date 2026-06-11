"""Create a human audit CSV from TONE3000 discovery metadata.

The discovery manifest contains loose text-heuristic label guesses. This script
deduplicates the candidates and writes a review sheet with explicit columns for
manual approval before any TONE3000 audio is downloaded or ingested.

Usage:
    python scripts/tone3000_make_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "tone3000" / "manifest.jsonl"
DEFAULT_OUT = REPO_ROOT / "data" / "tone3000" / "audit_candidates.csv"


def _flatten_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k.lower() in {"download_url", "file_url", "model_url", "avatar_url", "image"}:
                continue
            parts.append(str(k))
            parts.append(_flatten_text(v))
    elif isinstance(value, list):
        for item in value:
            parts.append(_flatten_text(item))
    elif value is not None:
        parts.append(str(value))
    return " ".join(parts)


def _compact_text(text: str, limit: int = 420) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[: limit - 1] + "…" if len(text) > limit else text


def _tone_url(tone_id: str) -> str:
    return f"https://www.tone3000.com/tones/{tone_id}" if tone_id else ""


def load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_audit_rows(records: list[dict], include_incomplete: bool) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    for rec in records:
        if not rec.get("is_ir_like"):
            continue
        missing = rec.get("missing_for_grid_label") or []
        if missing and not include_incomplete:
            continue
        tone_id = str(rec.get("tone_id") or "")
        dedupe_key = tone_id or rec.get("title") or json.dumps(rec.get("label_guess", {}), sort_keys=True)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        guess = rec.get("label_guess", {})
        raw = rec.get("raw", {})
        rows.append(
            {
                "tone_id": tone_id,
                "tone_url": _tone_url(tone_id),
                "title": rec.get("title") or "",
                "mic_guess": guess.get("mic"),
                "distance_mm_guess": guess.get("distance_mm"),
                "offset_mm_guess": guess.get("offset_mm"),
                "angle_deg_guess": guess.get("angle_deg"),
                "missing_for_grid_label": ", ".join(missing),
                "usable_for_grid": "review",
                "reviewed_mic": "",
                "reviewed_distance_mm": "",
                "reviewed_offset_mm": "",
                "reviewed_angle_deg": "",
                "cab_or_speaker_notes": "",
                "license_tos_ok": "",
                "download_models_ok": "",
                "review_notes": "",
                "metadata_hint": _compact_text(_flatten_text(raw)),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include-incomplete", action="store_true", help="include IR-like rows with missing mic/distance/offset guesses")
    args = ap.parse_args()

    records = load_records(args.manifest)
    rows = build_audit_rows(records, include_incomplete=args.include_incomplete)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tone_id",
        "tone_url",
        "title",
        "mic_guess",
        "distance_mm_guess",
        "offset_mm_guess",
        "angle_deg_guess",
        "missing_for_grid_label",
        "usable_for_grid",
        "reviewed_mic",
        "reviewed_distance_mm",
        "reviewed_offset_mm",
        "reviewed_angle_deg",
        "cab_or_speaker_notes",
        "license_tos_ok",
        "download_models_ok",
        "review_notes",
        "metadata_hint",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = [
        "# TONE3000 audit candidates",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Audit rows: {len(rows)}",
        "- Status: manual review required before any download or ingest",
        "",
        "## Review Columns",
        "",
        "- `usable_for_grid`: set to `yes`, `no`, or `maybe`.",
        "- `reviewed_*`: replace heuristic guesses with confirmed labels.",
        "- `license_tos_ok`: set only after Terms/API use is reviewed.",
        "- `download_models_ok`: set only after confirming model files are IR audio and accessible.",
        "",
        "## Caution",
        "",
        "The guesses are loose metadata triage. A candidate title/description can mention multiple mics or ranges; do not treat guesses as training labels.",
    ]
    (args.out.parent / "audit_summary.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {len(rows)} audit rows to {args.out}")
    print(f"wrote {args.out.parent / 'audit_summary.md'}")


if __name__ == "__main__":
    main()
