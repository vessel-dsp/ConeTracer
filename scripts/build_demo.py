"""Build the zero-install HTML demo: model weights baked into a single demo.html.

Usage: python scripts/build_demo.py [--ckpt runs/demo/model.npz] [--out app/demo.html]

The demo runs the spectral decoder forward pass + min-phase reconstruction in
JavaScript, convolves a Karplus-Strong guitar riff through the generated IR with
WebAudio, and lets you drag the mic across the speaker. No server, no installs.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cabir.model import MLP  # noqa: E402
from cabir.synth import MIC_NAMES  # noqa: E402


def pack_weights(model: MLP) -> str:
    buf = []
    meta = {"sizes": list(model.sizes), "mics": MIC_NAMES, "layers": []}
    offset = 0
    for i in range(model.n_layers):
        for key in (f"W{i}", f"b{i}"):
            arr = model.params[key].astype(np.float32)
            buf.append(arr.tobytes())
            meta["layers"].append({"key": key, "shape": list(arr.shape), "offset": offset})
            offset += arr.size
    return json.dumps(meta), base64.b64encode(b"".join(buf)).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/demo/model.npz")
    ap.add_argument("--out", default="app/demo.html")
    args = ap.parse_args()

    model = MLP.load(args.ckpt)
    meta, weights_b64 = pack_weights(model)

    here = os.path.dirname(__file__)
    with open(os.path.join(here, "demo_template.html")) as f:
        html = f.read()
    html = html.replace("/*__META__*/", meta).replace("__WEIGHTS_B64__", weights_b64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
