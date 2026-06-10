"""Conditional spectral decoder — pure-numpy MLP with Adam.

f(conditioning) -> log-magnitude spectrum (N_BINS).
Conditioning: mic one-hot + log(distance) + offset + angle (all normalized).
Small enough (<1M params) to train on CPU and run in a browser (weights -> JSON).
"""
from __future__ import annotations

import json

import numpy as np

from . import N_BINS
from .synth import MIC_NAMES, Condition

COND_DIM = len(MIC_NAMES) + 3


def encode_condition(cond: Condition) -> np.ndarray:
    vec = np.zeros(COND_DIM)
    vec[MIC_NAMES.index(cond.mic)] = 1.0
    vec[len(MIC_NAMES) + 0] = (np.log(cond.distance_mm) - np.log(40.0)) / 1.5
    vec[len(MIC_NAMES) + 1] = cond.offset_mm / 130.0 * 2 - 1
    vec[len(MIC_NAMES) + 2] = cond.angle_deg / 45.0 * 2 - 1
    return vec


class MLP:
    def __init__(self, sizes=(COND_DIM, 256, 256, N_BINS), seed: int = 0):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.params = {}
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.params[f"W{i}"] = rng.standard_normal((a, b)) * np.sqrt(2.0 / a)
            self.params[f"b{i}"] = np.zeros(b)
        self.n_layers = len(sizes) - 1
        self._adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self._adam_t = 0

    def forward(self, x: np.ndarray, cache: dict | None = None) -> np.ndarray:
        h = x
        if cache is not None:
            cache["h0"] = h
        for i in range(self.n_layers):
            z = h @ self.params[f"W{i}"] + self.params[f"b{i}"]
            h = np.maximum(z, 0) if i < self.n_layers - 1 else z
            if cache is not None:
                cache[f"z{i}"], cache[f"h{i+1}"] = z, h
        return h

    def backward(self, cache: dict, d_out: np.ndarray) -> dict:
        grads = {}
        d = d_out
        for i in reversed(range(self.n_layers)):
            h_in = cache[f"h{i}"]
            grads[f"W{i}"] = h_in.T @ d / len(d)
            grads[f"b{i}"] = d.mean(axis=0)
            if i > 0:
                d = (d @ self.params[f"W{i}"].T) * (cache[f"z{i-1}"] > 0)
        return grads

    def adam_step(self, grads: dict, lr: float = 1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self._adam_t += 1
        for k, g in grads.items():
            self._adam_m[k] = b1 * self._adam_m[k] + (1 - b1) * g
            self._adam_v[k] = b2 * self._adam_v[k] + (1 - b2) * g**2
            m_hat = self._adam_m[k] / (1 - b1**self._adam_t)
            v_hat = self._adam_v[k] / (1 - b2**self._adam_t)
            self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def save(self, path: str):
        np.savez_compressed(path, sizes=np.array(self.sizes), **self.params)

    @classmethod
    def load(cls, path: str) -> "MLP":
        data = np.load(path)
        model = cls(sizes=tuple(int(s) for s in data["sizes"]))
        for k in model.params:
            model.params[k] = data[k]
        return model

    def export_json(self, path: str):
        """Browser-portable weights (used by the zero-install HTML demo)."""
        payload = {
            "sizes": list(self.sizes),
            "mics": MIC_NAMES,
            "layers": [
                {
                    "W": self.params[f"W{i}"].astype(np.float32).round(5).tolist(),
                    "b": self.params[f"b{i}"].astype(np.float32).round(5).tolist(),
                }
                for i in range(self.n_layers)
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f)
