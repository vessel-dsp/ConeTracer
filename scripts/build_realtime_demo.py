"""Build a browser-native realtime mic-move demo from a real PyTorch checkpoint.

Usage:
    python scripts/build_realtime_demo.py \
      --ckpt runs/real_pcond_h256/best.pt \
      --out app/realtime.html
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.gradio_app import AppState, _validation_rows  # noqa: E402


def _pack_array(arr: np.ndarray, blobs: list[bytes], layers: list[dict], key: str) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    offset = sum(len(b) for b in blobs) // 4
    blobs.append(arr.tobytes())
    layers.append({"key": key, "shape": list(arr.shape), "offset": offset, "size": int(arr.size)})


def pack_checkpoint(ckpt_path: Path, labels: Path, irs: Path, *, include_reference_irs: bool = True) -> tuple[str, str]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    blobs: list[bytes] = []
    arrays: list[dict] = []
    for key in [
        "mic_emb.weight",
        "cab_emb.weight",
        "net.0.weight",
        "net.0.bias",
        "net.3.weight",
        "net.3.bias",
        "net.6.weight",
        "net.6.bias",
        "net.8.weight",
        "net.8.bias",
    ]:
        _pack_array(state[key].numpy(), blobs, arrays, key)

    validation = {mic: [] for mic in ckpt["mics"]}
    if include_reference_irs:
        app_state = AppState(ckpt_path, labels, irs)
        import app.gradio_app as gradio_app

        gradio_app.STATE = app_state
        cab = ckpt["cabs"][0]
        for mic in ckpt["mics"]:
            df = _validation_rows(cab, mic)
            validation[mic] = []
            for i, r in enumerate(df.itertuples()):
                ir_key = f"ref_ir.{mic}.{i}"
                label_row = app_state.labels[app_state.labels["file"] == r.file].iloc[0]
                _pack_array(app_state.irs[int(label_row["index"])], blobs, arrays, ir_key)
                validation[mic].append(
                    {
                        "distance": float(r.distance_mm),
                        "offset": float(r.offset_mm),
                        "guitar": float(r.guitar_rms_db),
                        "wide": float(r.wide_rms_db),
                        "ir_key": ir_key,
                        "file": str(r.file),
                    }
                )

    meta = {
        "model": "SpectralDecoder",
        "mics": ckpt["mics"],
        "cabs": ckpt["cabs"],
        "dist_norm": float(ckpt["dist_norm"]),
        "off_norm": float(ckpt["off_norm"]),
        "condition_presence": bool(ckpt.get("condition_presence", False)),
        "n_extra": int(ckpt.get("n_extra", 0)),
        "presence": 3.0 if ckpt.get("condition_presence") and ckpt.get("presence") is None else ckpt.get("presence"),
        "input_dim": int(state["net.0.weight"].shape[1]),
        "sr": 48_000,
        "n_taps": 4096,
        "n_bins": 2049,
        "arrays": arrays,
        "validation": validation,
        "include_reference_irs": include_reference_irs,
    }
    return json.dumps(meta, separators=(",", ":")), base64.b64encode(b"".join(blobs)).decode()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/real_pcond_h256/best.pt")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "data/parsed/labels.parquet")
    ap.add_argument("--irs", type=Path, default=REPO_ROOT / "data/parsed/irs.npy")
    ap.add_argument("--template", type=Path, default=REPO_ROOT / "scripts/realtime_template.html")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "app/realtime.html")
    ap.add_argument(
        "--public-page",
        action="store_true",
        help="build a GitHub Pages-safe app: model weights only, no captured reference IR audio embedded",
    )
    args = ap.parse_args()

    meta, weights = pack_checkpoint(
        args.ckpt,
        args.labels,
        args.irs,
        include_reference_irs=not args.public_page,
    )
    html = args.template.read_text()
    html = html.replace("/*__META__*/", meta).replace("__WEIGHTS_B64__", weights)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
