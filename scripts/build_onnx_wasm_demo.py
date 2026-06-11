"""Build the ConeTrace ONNX Runtime Web/WASM demo page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", type=Path, default=REPO_ROOT / "docs/assets/conetrace-godscab-v0.1.onnx.json")
    ap.add_argument("--template", type=Path, default=REPO_ROOT / "scripts/onnx_wasm_template.html")
    ap.add_argument("--onnx-url", default="assets/conetrace-godscab-v0.1.onnx")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/wasm.html")
    args = ap.parse_args()

    meta = json.loads(args.metadata.read_text())
    html = args.template.read_text()
    html = html.replace("/*__META__*/", json.dumps(meta, separators=(",", ":")))
    html = html.replace("__ONNX_URL__", args.onnx_url)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
