"""Metadata-only TONE3000 IR discovery spike.

This does not train, ingest audio, or download raw IR files. It probes the
authenticated TONE3000 API for IR-like tones, stores metadata locally, and
estimates whether each result contains enough structured text to become a
cab-ir-lab training label.

Environment:
    TONE3000_ACCESS_TOKEN   OAuth access token from TONE3000

Usage:
    python scripts/tone3000_discover.py --limit 100

Outputs are ignored by git:
    data/tone3000/manifest.jsonl
    data/tone3000/summary.md
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "tone3000"
DEFAULT_API_BASE = "https://www.tone3000.com/api/v1"

MIC_ALIASES = {
    "sm57": "sm57",
    "57": "sm57",
    "sm7b": "sm7b",
    "sm7": "sm7b",
    "c414": "c414",
    "414": "c414",
    "u87": "u87",
    "md421": "md421",
    "421": "md421",
    "r121": "r121",
    "royer": "r121",
    "m160": "m160",
    "nt5": "nt5",
    "e906": "e906",
    "906": "e906",
}

POSITION_ALIASES = {
    "cap": 0.0,
    "dustcap": 0.0,
    "dust cap": 0.0,
    "cap edge": 32.0,
    "cap-edge": 32.0,
    "edge": 57.0,
    "cone near": 57.0,
    "cone": 57.0,
    "cone far": 83.0,
}

DISTANCE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mm|cm|in|inch|inches|\"|ft|foot|feet)\b", re.I)
ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees|°)\b", re.I)


def _request_json(url: str, token: str) -> Any:
    req = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"TONE3000 API request failed: HTTP {exc.code}\n{url}\n{body[:1000]}") from exc


def _iter_items(payload: Any) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "tones", "results", "items", "models"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return [payload]


def _flatten_text(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        for k, v in value.items():
            if k.lower() in {"download_url", "file_url", "url", "avatar", "image"}:
                continue
            parts.append(str(k))
            parts.append(_flatten_text(v))
    elif isinstance(value, list):
        for item in value:
            parts.append(_flatten_text(item))
    elif value is not None:
        parts.append(str(value))
    return " ".join(parts)


def _find_mic(text: str) -> str | None:
    norm = f" {re.sub(r'[^a-z0-9]+', ' ', text.lower())} "
    best = None
    for alias, mic in MIC_ALIASES.items():
        if f" {alias} " in norm and (best is None or len(alias) > len(best[0])):
            best = (alias, mic)
    return best[1] if best else None


def _find_position(text: str) -> float | None:
    lower = text.lower()
    best = None
    for alias, offset in POSITION_ALIASES.items():
        if alias in lower and (best is None or len(alias) > len(best[0])):
            best = (alias, offset)
    return best[1] if best else None


def _find_distance(text: str) -> float | None:
    match = DISTANCE_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"in", "inch", "inches", '"'}:
        return round(value * 25.4, 2)
    if unit in {"ft", "foot", "feet"}:
        return round(value * 304.8, 2)
    if unit == "cm":
        return round(value * 10.0, 2)
    return round(value, 2)


def _find_angle(text: str) -> float | None:
    match = ANGLE_RE.search(text)
    if match:
        return float(match.group(1))
    lower = text.lower()
    if "off axis" in lower or "off-axis" in lower:
        return 45.0
    if "on axis" in lower or "on-axis" in lower:
        return 0.0
    return None


def _is_ir_like(item: dict, text: str) -> bool:
    lower = text.lower()
    fields = json.dumps(item, default=str).lower()
    return any(token in lower or token in fields for token in ["impulse response", " ir ", "cab ir", "cabinet ir"])


def _tone_id(item: dict) -> str:
    for key in ("id", "tone_id", "uuid", "slug"):
        if key in item and item[key] is not None:
            return str(item[key])
    return ""


def analyze_item(item: dict) -> dict:
    text = _flatten_text(item)
    mic = _find_mic(text)
    distance = _find_distance(text)
    offset = _find_position(text)
    angle = _find_angle(text)
    missing = []
    if mic is None:
        missing.append("mic")
    if distance is None:
        missing.append("distance")
    if offset is None:
        missing.append("offset")
    return {
        "tone_id": _tone_id(item),
        "title": str(item.get("title") or item.get("name") or item.get("display_name") or ""),
        "is_ir_like": _is_ir_like(item, text),
        "label_guess": {
            "mic": mic,
            "distance_mm": distance,
            "offset_mm": offset,
            "angle_deg": angle,
        },
        "missing_for_grid_label": missing,
        "raw": item,
    }


def _search_url(base: str, page: int, per_page: int) -> str:
    params = {
        "query": "impulse response",
        "gears": "ir",
        "page": page,
        "page_size": per_page,
    }
    return f"{base.rstrip('/')}/tones/search?{urlencode(params)}"


def discover(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("TONE3000_ACCESS_TOKEN")
    if not token and not args.fixture:
        raise SystemExit(
            "No TONE3000 token found. Set TONE3000_ACCESS_TOKEN or pass --fixture "
            "with a saved API JSON response for offline analysis."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    if args.fixture:
        payload = json.loads(Path(args.fixture).read_text())
        records.extend(analyze_item(item) for item in _iter_items(payload))
    else:
        page = 1
        while len(records) < args.limit:
            payload = _request_json(_search_url(args.api_base, page, args.per_page), token)
            items = _iter_items(payload)
            if not items:
                break
            for item in items:
                records.append(analyze_item(item))
                if len(records) >= args.limit:
                    break
            page += 1

    manifest = args.out / "manifest.jsonl"
    with manifest.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n")

    ir_like = [r for r in records if r["is_ir_like"]]
    labelable = [r for r in ir_like if not r["missing_for_grid_label"]]
    summary = [
        "# TONE3000 discovery summary",
        "",
        f"- Records scanned: {len(records)}",
        f"- IR-like records: {len(ir_like)}",
        f"- Position-grid labelable records: {len(labelable)}",
        "",
        "## Labelability",
        "",
        "| Missing fields | Count |",
        "|---|---:|",
    ]
    counts: dict[str, int] = {}
    for rec in ir_like:
        key = ", ".join(rec["missing_for_grid_label"]) or "(none)"
        counts[key] = counts.get(key, 0) + 1
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        summary.append(f"| {key} | {count} |")
    summary.extend(
        [
            "",
            "## Notes",
            "",
            "- This is metadata-only discovery. No IR audio is downloaded.",
            "- Loose description parsing is for triage only; ingest must never guess labels.",
            "- Records with missing mic/distance/offset are not usable for the current position-grid model without manual or stronger metadata.",
        ]
    )
    (args.out / "summary.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {manifest}")
    print(f"wrote {args.out / 'summary.md'}")
    print(f"records={len(records)} ir_like={len(ir_like)} labelable={len(labelable)}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--token", default=None, help="OAuth access token; defaults to TONE3000_ACCESS_TOKEN")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--per-page", type=int, default=50)
    ap.add_argument("--fixture", type=Path, default=None, help="analyze a saved API JSON payload instead of calling the API")
    raise SystemExit(discover(ap.parse_args()))


if __name__ == "__main__":
    main()
