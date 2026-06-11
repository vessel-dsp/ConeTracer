"""Download explicitly approved TONE3000 model files.

This is a guarded gate, not a crawler. It only considers rows that have been
manually approved in the audit CSV and it defaults to dry-run mode. Use it for a
small, reviewed pack import after Terms/license and label mapping are approved.

Required audit row values:
    usable_for_grid=yes
    license_tos_ok=yes
    download_models_ok=yes

Environment:
    TONE3000_ACCESS_TOKEN   OAuth access token from TONE3000

Usage:
    python scripts/tone3000_import_approved.py --dry-run
    python scripts/tone3000_import_approved.py --download --max-tones 1 --max-files-per-tone 5
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_OUT = REPO_ROOT / "data" / "raw" / "tone3000"
DEFAULT_PAGE_SIZE = 100


def _yes(value: str | None) -> bool:
    return (value or "").strip().lower() in {"yes", "y", "true", "1", "approved"}


def _request(url: str, token: str) -> bytes:
    req = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=60) as resp:
            return resp.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"TONE3000 request failed: HTTP {exc.code}\n{url}\n{body[:1000]}") from exc


def _request_json(url: str, token: str) -> Any:
    return json.loads(_request(url, token).decode("utf-8"))


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


def _models_url(base: str, tone_id: str, page: int, page_size: int) -> str:
    params = {"tone_id": tone_id, "page": page, "page_size": page_size}
    return f"{base.rstrip('/')}/models?{urlencode(params)}"


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


def _safe_name(value: str, fallback: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value[:120] or fallback


def approved_rows(audit_path: Path) -> list[dict]:
    rows = list(csv.DictReader(audit_path.open()))
    return [
        row
        for row in rows
        if _yes(row.get("usable_for_grid")) and _yes(row.get("license_tos_ok")) and _yes(row.get("download_models_ok"))
    ]


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def import_approved(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("TONE3000_ACCESS_TOKEN")
    if not token:
        raise SystemExit("No TONE3000 token found. Set TONE3000_ACCESS_TOKEN or pass --token.")
    if args.download and args.dry_run:
        raise SystemExit("Choose either --dry-run or --download, not both.")

    rows = approved_rows(args.audit)
    if args.max_tones is not None:
        rows = rows[: args.max_tones]
    if not rows:
        print("No approved TONE3000 rows found.")
        print("Set usable_for_grid=yes, license_tos_ok=yes, and download_models_ok=yes after manual review.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    planned: list[dict] = []
    downloaded = 0

    for row in rows:
        tone_id = (row.get("tone_id") or "").strip()
        if not tone_id:
            continue
        models = fetch_models(args.api_base, token, tone_id, args.page_size, args.sleep_s)
        models = [model for model in models if model.get("model_url")]
        if args.include_name_regex:
            include_re = re.compile(args.include_name_regex, re.I)
            models = [model for model in models if include_re.search(str(model.get("name") or ""))]
        if args.exclude_name_regex:
            exclude_re = re.compile(args.exclude_name_regex, re.I)
            models = [model for model in models if not exclude_re.search(str(model.get("name") or ""))]
        if args.max_files_per_tone is not None:
            models = models[: args.max_files_per_tone]

        tone_dir = args.out / tone_id
        planned.append(
            {
                "tone_id": tone_id,
                "title": row.get("title", ""),
                "models_with_url": len(models),
                "dry_run": bool(args.dry_run),
            }
        )
        print(f"tone_id={tone_id} approved models_with_url={len(models)}")
        if args.dry_run:
            continue

        tone_dir.mkdir(parents=True, exist_ok=True)
        _write_json(tone_dir / "audit_row.json", row)
        _write_json(tone_dir / "models_metadata.json", models)
        for model in models:
            model_id = str(model.get("id") or downloaded)
            model_name = _safe_name(str(model.get("name") or model_id), model_id)
            suffix = Path(str(model.get("model_url"))).suffix or ".bin"
            out_path = tone_dir / f"{model_id}_{model_name}{suffix}"
            if out_path.exists() and not args.overwrite:
                print(f"skip existing {out_path}")
                continue
            body = _request(str(model["model_url"]), token)
            digest = hashlib.sha256(body).hexdigest()
            out_path.write_bytes(body)
            (out_path.with_suffix(out_path.suffix + ".sha256")).write_text(f"{digest}  {out_path.name}\n")
            downloaded += 1
            print(f"downloaded {out_path.name} sha256={digest[:12]}...")
            if args.sleep_s > 0:
                time.sleep(args.sleep_s)

    _write_json(args.out / "import_plan.json", planned)
    print(f"wrote {args.out / 'import_plan.json'}")
    if args.download:
        print(f"downloaded_files={downloaded}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-base", default=DEFAULT_API_BASE)
    ap.add_argument("--token", default=None, help="OAuth access token; defaults to TONE3000_ACCESS_TOKEN")
    ap.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    ap.add_argument("--sleep-s", type=float, default=0.25)
    ap.add_argument("--max-tones", type=int, default=1)
    ap.add_argument("--max-files-per-tone", type=int, default=5)
    ap.add_argument("--include-name-regex", default=None, help="only import models whose name matches this regex")
    ap.add_argument("--exclude-name-regex", default=None, help="skip models whose name matches this regex")
    ap.add_argument("--overwrite", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        args.dry_run = False
    raise SystemExit(import_approved(args))


if __name__ == "__main__":
    main()
