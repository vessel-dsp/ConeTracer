"""Synthetic cabinet IR generator — physics-inspired stand-in for real IR packs.

Fabricates labeled (condition -> magnitude/IR) pairs so the full pipeline
(dataset -> model -> inference -> UI) can be built and validated with zero
downloads. Each effect mirrors a real capture phenomenon:

- speaker base response: bandpass + cone-breakup resonances + seeded modal ripple
- offset (cap -> edge): HF tilt + moving interference notches (path-difference comb)
- distance: proximity low-shelf (cardioid), 1/d level, floor-bounce comb when far
- angle: off-axis HF rolloff
- mic type: archetype EQ curves (incl. position-dependent proximity strength)

NOTE: mic characteristics here are BAKED INTO the generated IR, exactly like a
real capture: the dataset stores only the label (mic name), never the mic curve.
The model must learn each mic's coloration from data — synth.py is hidden
ground truth for evaluation, not an input to the model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import N_BINS, SR
from .dsp import apply_predelay, fade_tail, minphase_from_magnitude, normalize_energy

C_MM_S = 343_000.0  # speed of sound in mm/s

FREQS = np.linspace(0, SR / 2, N_BINS)

# Mic archetypes: (name, presence_peak_hz, presence_db, hf_cut_hz, proximity_strength)
MICS = {
    "dyn57": dict(presence_hz=5200, presence_db=5.0, hf_cut=14_000, prox=1.0, lf_cut=90),
    "dyn421": dict(presence_hz=4200, presence_db=2.5, hf_cut=16_000, prox=0.9, lf_cut=60),
    "ribbon121": dict(presence_hz=0, presence_db=0.0, hf_cut=8_500, prox=1.4, lf_cut=40),
    "cond414": dict(presence_hz=9000, presence_db=2.0, hf_cut=19_000, prox=0.5, lf_cut=30),
}
MIC_NAMES = list(MICS)


@dataclass(frozen=True)
class Condition:
    mic: str          # one of MIC_NAMES
    distance_mm: float  # 5..300 (grill to capsule)
    offset_mm: float    # 0 (dust cap) .. 130 (cone edge)
    angle_deg: float    # 0..45 off-axis
    cab_seed: int = 0   # which synthetic cabinet


def _db(x_db: np.ndarray | float) -> np.ndarray | float:
    return 10 ** (np.asarray(x_db) / 20)


def _peak(f0: float, gain_db: float, q: float) -> np.ndarray:
    """Magnitude of a resonant peak/dip."""
    if f0 <= 0:
        return np.ones_like(FREQS)
    x = FREQS / f0
    resp = 1 + (gain_db / 20) / np.sqrt((1 - x**2) ** 2 + (x / q) ** 2)
    return np.maximum(resp, 0.05)


def _shelf_low(f0: float, gain_db: float) -> np.ndarray:
    s = 1 / (1 + (FREQS / max(f0, 1)) ** 2)
    return _db(gain_db * s)


def _rolloff_high(f0: float, order: float) -> np.ndarray:
    return 1 / np.sqrt(1 + (FREQS / max(f0, 1)) ** (2 * order))


def _rolloff_low(f0: float, order: float = 2) -> np.ndarray:
    x = FREQS / max(f0, 1)
    return (x**order) / np.sqrt(1 + x ** (2 * order))


def cab_magnitude(cond: Condition) -> np.ndarray:
    """Ground-truth magnitude response for a condition (the 'real cab' the model must learn)."""
    rng = np.random.default_rng(cond.cab_seed)
    mic = MICS[cond.mic]

    # --- speaker/cab base: bandpass + breakup peaks + seeded modal fingerprint
    mag = _rolloff_low(75.0) * _rolloff_high(5200.0, 2.5)
    for f0, g, q in [(110, 4, 2.2), (1600, 3, 3.0), (2600, 4, 4.0), (3900, 3.5, 5.0)]:
        jitter = 1 + 0.08 * rng.standard_normal()
        mag *= _peak(f0 * jitter, g, q)
    # modal ripple fingerprint (smooth, fixed per cab)
    ripple = rng.standard_normal(24)
    phase = rng.uniform(0, 2 * np.pi, 24)
    log_f = np.log1p(FREQS / 50)
    fingerprint = sum(
        0.35 * ripple[k] * np.sin((k + 2) * log_f + phase[k]) for k in range(24)
    )
    mag *= _db(fingerprint)

    # --- offset: cap bright -> edge dark + interference notches
    off_norm = np.clip(cond.offset_mm / 130.0, 0, 1)
    mag *= _rolloff_high(8000 - 5500 * off_norm, 1.5) / _rolloff_high(8000, 1.5)
    path_diff_mm = np.sqrt(cond.distance_mm**2 + cond.offset_mm**2) - cond.distance_mm
    tau = path_diff_mm / C_MM_S
    comb_depth = 0.45 * off_norm / (1 + cond.distance_mm / 80)
    mag *= np.abs(1 + comb_depth * np.exp(-2j * np.pi * FREQS * tau))

    # --- distance: proximity, level, floor bounce
    prox_db = 7.0 * mic["prox"] * np.exp(-cond.distance_mm / 60.0)
    mag *= _shelf_low(250, prox_db)
    mag *= 60.0 / (60.0 + cond.distance_mm)  # level vs distance
    if cond.distance_mm > 100:
        bounce_tau = (2 * np.sqrt(600**2 + (cond.distance_mm / 2) ** 2)) / C_MM_S * 0.18
        depth = 0.18 * (cond.distance_mm - 100) / 300
        mag *= np.abs(1 + depth * np.exp(-2j * np.pi * FREQS * bounce_tau))

    # --- angle: off-axis HF rolloff
    ang = np.clip(cond.angle_deg / 45.0, 0, 1)
    mag *= _rolloff_high(16000 - 9500 * ang, 1.2) / _rolloff_high(16000, 1.2)

    # --- mic curve (BAKED IN, like every real capture)
    mag *= _rolloff_low(mic["lf_cut"], 1.5)
    mag *= _rolloff_high(mic["hf_cut"], 2.0)
    if mic["presence_hz"]:
        mag *= _peak(mic["presence_hz"], mic["presence_db"], 1.4)

    return np.maximum(mag, 1e-6)


def render_ir(cond: Condition) -> np.ndarray:
    """Full synthetic IR: min-phase from ground-truth magnitude + physical pre-delay."""
    mag = cab_magnitude(cond)
    ir = minphase_from_magnitude(mag)
    delay = int(round(cond.distance_mm / C_MM_S * SR))
    ir = apply_predelay(ir, delay)
    return normalize_energy(fade_tail(ir))


def sample_conditions(n: int, seed: int, cab_seed: int = 0) -> list[Condition]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append(
            Condition(
                mic=MIC_NAMES[rng.integers(len(MIC_NAMES))],
                distance_mm=float(np.exp(rng.uniform(np.log(5), np.log(300)))),
                offset_mm=float(rng.uniform(0, 130)),
                angle_deg=float(rng.uniform(0, 45)),
                cab_seed=cab_seed,
            )
        )
    return out
