"""Enrich TONE3000 audit candidates with API/model metadata.

This is still metadata-only: it reads the manual audit candidate CSV, fetches
tone detail and model listings from the authenticated TONE3000 API, and writes
review artifacts under data/tone3000/. It deliberately does not download model
files from model_url.

Environment:
    TONE3000_ACCESS_TOKEN   OAuth access token from TONE3000

Usage:
    python scripts/tone3000_enrich_audit.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API_BASE = "https://www.tone3000.com/api/v1"
DEFAULT_AUDIT = REPO_ROOT / "data" / "tone3000" / "audit_candidates.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "tone3000" / "audit_enriched.csv"
DEFAULT_MODEL_SAMPLES = REPO_ROOT / "data" / "tone3000" / "model_name_samples.jsonl"

MIC_ALIASES = {
    "sm57": "sm57",
    "sm 57": "sm57",
    "sm7b": "sm7b",
    "sm 7b": "sm7b",
    "c414": "c414",
    "c 414": "c414",
    "u87": "u87",
    "u 87": "u87",
    "md421": "md421",
    "md 421": "md421",
    "m201": "m201",
    "m201tg": "m201",
    "m 201": "m201",
    "r121": "r121",
    "r 121": "r121",
    "royer": "r121",
    "m160": "m160",
    "m 160": "m160",
    "nt5": "nt5",
    "nt 5": "nt5",
    "e906": "e906",
    "e 906": "e906",
}

INCH_RE = re.compile(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*(?:in|inch|inches|\")\b", re.I)
MODEL_NAME_SAMPLE_LIMIT = 20


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
    for key in ("data", "models", "items", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return [payload]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\n", " ").strip()


def _models_url(base: str, tone_id: str, page: int, page_size: int) -> str:
    params = {"tone_id": tone_id, "page": page, "page_size": page_size}
    return f"{base.rstrip('/')}/models?{urlencode(params)}"


def _tone_url(base: str, tone_id: str) -> str:
    return f"{base.rstrip('/')}/tones/{tone_id}"


def _page_count(payload: Any, current_page: int, item_count: int) -> int | None:
    if not isinstance(payload, dict):
        return None
    for key in ("total_pages", "last_page", "pages"):
        if payload.get(key):
            return int(payload[key])
    total = payload.get("total")
    page_size = payload.get("page_size") or payload.get("per_page")
    if total and page_size:
        return max(1, int((int(total) + int(page_size) - 1) / int(page_size)))
    if item_count == 0:
        return current_page
    return None


def fetch_models(base: str, token: str, tone_id: str, page_size: int, sleep_s: float) -> list[dict]:
    models: list[dict] = []
    page = 1
    while True:
        payload = _request_json(_models_url(base, tone_id, page, page_size), token)
        items = _iter_items(payload)
        models.extend(items)
        total_pages = _page_count(payload, page, len(items))
        if total_pages is not None and page >= total_pages:
            break
        if total_pages is None and len(items) < page_size:
            break
        page += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
    return models


def _find_mics(model_names: list[str]) -> list[str]:
    found: set[str] = set()
    for name in model_names:
        norm = f" {re.sub(r'[^a-z0-9]+', ' ', name.lower())} "
        for alias, mic in MIC_ALIASES.items():
            alias_norm = f" {re.sub(r'[^a-z0-9]+', ' ', alias.lower()).strip()} "
            if alias_norm in norm:
                found.add(mic)
    return sorted(found)


def _distance_tokens_mm(model_names: list[str]) -> list[str]:
    values: set[float] = set()
    for name in model_names:
        for match in INCH_RE.finditer(name):
            values.add(round(float(match.group(1)) * 25.4, 2))
    return [f"{value:g}" for value in sorted(values)]


def _position_token_shapes(model_names: list[str]) -> list[str]:
    shapes: set[str] = set()
    for name in model_names:
        count = len(INCH_RE.findall(name))
        if count:
            shapes.add(f"{count}_inch_tokens")
    return sorted(shapes)


def _recommend_status(tone: dict, models: list[dict], mics: list[str], distance_tokens: list[str]) -> str:
    if not tone:
        return "reject:no_tone_detail"
    if not bool(tone.get("is_public", True)):
        return "reject:not_public"
    if not models:
        return "reject:no_models"
    if not any(model.get("model_url") for model in models):
        return "reject:no_model_urls"
    if not mics or not distance_tokens:
        return "needs_manual:weak_model_name_labels"
    return "candidate:manual_terms_and_label_review"


def _join(values: list[str], limit: int | None = None) -> str:
    if limit is not None:
        values = values[:limit]
    return "; ".join(values)


def enrich(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("TONE3000_ACCESS_TOKEN")
    if not token:
        raise SystemExit("No TONE3000 token found. Set TONE3000_ACCESS_TOKEN or pass --token.")

    rows = list(csv.DictReader(args.audit.open()))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    enriched: list[dict] = []
    sample_records: list[dict] = []

    for row in rows:
        tone_id = row.get("tone_id", "").strip()
        if not tone_id:
            continue
        tone = _request_json(_tone_url(args.api_base, tone_id), token)
        if args.sleep_s > 0:
            time.sleep(args.sleep_s)
        models = fetch_models(args.api_base, token, tone_id, args.page_size, args.sleep_s)
        model_names = [_safe_text(model.get("name") or model.get("title") or model.get("display_name")) for model in models]
        model_names = [name for name in model_names if name]
        mics = _find_mics(model_names)
        distance_tokens = _distance_tokens_mm(model_names)
        token_shapes = _position_token_shapes(model_names)
        has_model_urls = any(bool(model.get("model_url")) for model in models)
        sample_names = model_names[:MODEL_NAME_SAMPLE_LIMIT]

        sample_records.append(
            {
                "tone_id": tone_id,
                "title": row.get("title", ""),
                "model_count": len(models),
                "sample_names": sample_names,
            }
        )
        enriched.append(
            {
                "tone_id": tone_id,
                "api_url": _safe_text(tone.get("url") if isinstance(tone, dict) else ""),
                "title": row.get("title", ""),
                "api_title": _safe_text(tone.get("title") if isinstance(tone, dict) else ""),
                "gear": _safe_text(tone.get("gear") if isinstance(tone, dict) else ""),
                "platform": _safe_text(tone.get("platform") if isinstance(tone, dict) else ""),
                "is_public": _safe_text(tone.get("is_public") if isinstance(tone, dict) else ""),
                "license": _safe_text(tone.get("license") if isinstance(tone, dict) else ""),
                "tone_models_count": _safe_text(tone.get("models_count") if isinstance(tone, dict) else ""),
                "tone_irs_count": _safe_text(tone.get("irs_count") if isinstance(tone, dict) else ""),
                "api_models_total": len(models),
                "has_model_urls": "yes" if has_model_urls else "no",
                "inferred_model_mics": _join(mics),
                "model_name_distance_tokens_mm": _join(distance_tokens),
                "model_name_position_token_shapes": _join(token_shapes),
                "model_name_samples": _join(sample_names, MODEL_NAME_SAMPLE_LIMIT),
                "manual_status_recommendation": _recommend_status(tone if isinstance(tone, dict) else {}, models, mics, distance_tokens),
            }
        )
        print(f"enriched tone_id={tone_id} models={len(models)} mics={','.join(mics) or '-'}")

    fieldnames = [
        "tone_id",
        "api_url",
        "title",
        "api_title",
        "gear",
        "platform",
        "is_public",
        "license",
        "tone_models_count",
        "tone_irs_count",
        "api_models_total",
        "has_model_urls",
        "inferred_model_mics",
        "model_name_distance_tokens_mm",
        "model_name_position_token_shapes",
        "model_name_samples",
        "manual_status_recommendation",
    ]
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)
    with args.model_samples.open("w") as f:
        for record in sample_records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    candidate_count = sum(1 for row in enriched if row["manual_status_recommendation"].startswith("candidate:"))
    summary = [
        "# TONE3000 enriched audit summary",
        "",
        f"- Audit input: `{args.audit}`",
        f"- Enriched rows: {len(enriched)}",
        f"- Candidate rows after model-name/API metadata check: {candidate_count}",
        f"- Output CSV: `{args.out}`",
        f"- Model-name samples: `{args.model_samples}`",
        "",
        "## Interpretation",
        "",
        "- This is metadata-only. `model_url` availability is recorded, but no model files are downloaded.",
        "- `model_name_distance_tokens_mm` lists inch values parsed from model names, converted to millimeters. It does not decide which token is distance-from-grille vs offset-from-cap.",
        "- Rows marked `candidate:*` still need Terms/license review and manual label mapping before ingest.",
    ]
    (args.out.parent / "audit_enriched_summary.md").write_text("\n".join(summary) + "\n")
    print(f"wrote {args.out}")
    print(f"wrote {args.model_samples}")
    print(f"wrote {args.out.parent / 'audit_enriched_summary.md'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--token", default=None, help="OAuth access token; defaults to TONE3000_ACCESS_TOKEN")
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model-samples", type=Path, default=DEFAULT_MODEL_SAMPLES)
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--sleep-s", type=float, default=0.05)
    raise SystemExit(enrich(ap.parse_args()))


if __name__ == "__main__":
    main()
