"""Export a ConeTrace PyTorch checkpoint to ONNX.

The exported graph keeps the same model-facing inputs as ``SpectralDecoder``:
mic index, cab index, normalized distance, normalized offset, normalized angle,
and optional extra scalars such as presence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir.model import SpectralDecoder  # noqa: E402


def load_model(ckpt_path: Path) -> tuple[SpectralDecoder, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    model = SpectralDecoder(
        n_mics=int(ckpt["n_mics"]),
        n_cabs=int(ckpt["n_cabs"]),
        emb_dim=int(ckpt.get("emb_dim", state["mic_emb.weight"].shape[1])),
        hidden=int(ckpt.get("hidden", state["net.0.weight"].shape[0])),
        dropout=0.0,
        n_extra=int(ckpt.get("n_extra", 0)),
    )
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def export_onnx(ckpt_path: Path, out: Path, metadata_out: Path | None, *, opset: int) -> None:
    model, ckpt = load_model(ckpt_path)
    n_extra = int(ckpt.get("n_extra", 0))
    extra_shape = (1, max(1, n_extra))

    mic_idx = torch.zeros((1,), dtype=torch.long)
    cab_idx = torch.zeros((1,), dtype=torch.long)
    distance = torch.zeros((1,), dtype=torch.float32)
    offset = torch.zeros((1,), dtype=torch.float32)
    angle = torch.zeros((1,), dtype=torch.float32)
    extra = torch.zeros(extra_shape, dtype=torch.float32)

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (mic_idx, cab_idx, distance, offset, angle, extra),
        out,
        input_names=["mic_idx", "cab_idx", "distance", "offset", "angle", "extra"],
        output_names=["logmag"],
        dynamic_axes={
            "mic_idx": {0: "batch"},
            "cab_idx": {0: "batch"},
            "distance": {0: "batch"},
            "offset": {0: "batch"},
            "angle": {0: "batch"},
            "extra": {0: "batch"},
            "logmag": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    if metadata_out:
        meta = {
            "name": "ConeTrace",
            "release": "conetrace-godscab-v0.1",
            "checkpoint": str(ckpt_path),
            "onnx": str(out),
            "mics": ckpt["mics"],
            "cabs": ckpt["cabs"],
            "dist_norm": float(ckpt["dist_norm"]),
            "off_norm": float(ckpt["off_norm"]),
            "condition_presence": bool(ckpt.get("condition_presence", False)),
            "n_extra": n_extra,
            "presence": 3.0 if ckpt.get("condition_presence") and ckpt.get("presence") is None else ckpt.get("presence"),
            "sr": 48_000,
            "n_taps": 4096,
            "n_bins": 2049,
            "inputs": {
                "mic_idx": "int64[batch]",
                "cab_idx": "int64[batch]",
                "distance": "float32[batch], log1p(distance_mm) / dist_norm",
                "offset": "float32[batch], offset_mm / off_norm",
                "angle": "float32[batch], angle_deg / 90",
                "extra": f"float32[batch,{max(1, n_extra)}], presence as (presence - 1) / 4 when enabled",
            },
        }
        metadata_out.parent.mkdir(parents=True, exist_ok=True)
        metadata_out.write_text(json.dumps(meta, indent=2) + "\n")


def verify(ckpt_path: Path, onnx_path: Path) -> float:
    import onnxruntime as ort

    model, ckpt = load_model(ckpt_path)
    n_extra = int(ckpt.get("n_extra", 0))
    rng = np.random.default_rng(7)
    mic_idx = rng.integers(0, int(ckpt["n_mics"]), size=(8,), dtype=np.int64)
    cab_idx = np.zeros((8,), dtype=np.int64)
    distance_mm = rng.uniform(0, 50.8, size=(8,)).astype(np.float32)
    offset_mm = rng.uniform(0, 83, size=(8,)).astype(np.float32)
    angle_deg = np.zeros((8,), dtype=np.float32)
    extra = np.zeros((8, max(1, n_extra)), dtype=np.float32)
    if ckpt.get("condition_presence"):
        extra[:, 0] = (3.0 - 1.0) / 4.0

    inputs = {
        "mic_idx": mic_idx,
        "cab_idx": cab_idx,
        "distance": np.log1p(distance_mm) / float(ckpt["dist_norm"]),
        "offset": offset_mm / float(ckpt["off_norm"]),
        "angle": angle_deg / 90.0,
        "extra": extra,
    }
    with torch.no_grad():
        torch_out = model(
            torch.from_numpy(inputs["mic_idx"]),
            torch.from_numpy(inputs["cab_idx"]),
            torch.from_numpy(inputs["distance"]).float(),
            torch.from_numpy(inputs["offset"]).float(),
            torch.from_numpy(inputs["angle"]).float(),
            torch.from_numpy(inputs["extra"]).float(),
        ).numpy()
    ort_out = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    return float(np.max(np.abs(torch_out - ort_out)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=REPO_ROOT / "runs/real_pcond_h256/best.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs/assets/conetrace-godscab-v0.1.onnx")
    ap.add_argument("--metadata-out", type=Path, default=REPO_ROOT / "docs/assets/conetrace-godscab-v0.1.onnx.json")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    export_onnx(args.ckpt, args.out, args.metadata_out, opset=args.opset)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    if args.metadata_out:
        print(f"wrote {args.metadata_out}")
    if not args.no_verify:
        max_abs = verify(args.ckpt, args.out)
        print(f"ONNX Runtime max abs diff vs PyTorch: {max_abs:.6g}")


if __name__ == "__main__":
    main()
