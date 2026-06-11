"""DSP utilities: min-phase reconstruction, windowing, wav export."""
from __future__ import annotations

import numpy as np

from . import N_TAPS, SR


def minphase_from_magnitude(mag: np.ndarray, n_taps: int = N_TAPS) -> np.ndarray:
    """Minimum-phase IR from a magnitude response (rfft bins) via the cepstral method."""
    n_fft = 2 * (mag.shape[-1] - 1)
    log_mag = np.log(np.maximum(mag, 1e-9))
    # Even-symmetric full spectrum -> real cepstrum
    cep = np.fft.irfft(log_mag, n=n_fft)
    # Fold: keep c[0], double positive quefrencies, zero negative ones
    win = np.zeros(n_fft)
    win[0] = 1.0
    win[1 : n_fft // 2] = 2.0
    win[n_fft // 2] = 1.0
    min_phase_spectrum = np.exp(np.fft.rfft(cep * win, n=n_fft))
    ir = np.fft.irfft(min_phase_spectrum, n=n_fft)[:n_taps]
    return ir.astype(np.float64)


def apply_predelay(ir: np.ndarray, delay_samples: int) -> np.ndarray:
    if delay_samples <= 0:
        return ir
    out = np.zeros_like(ir)
    out[delay_samples:] = ir[: len(ir) - delay_samples]
    return out


def fade_tail(ir: np.ndarray, fade_frac: float = 0.25) -> np.ndarray:
    n_fade = int(len(ir) * fade_frac)
    out = ir.copy()
    out[-n_fade:] *= np.hanning(2 * n_fade)[n_fade:]
    return out


def normalize_energy(ir: np.ndarray) -> np.ndarray:
    return ir / max(np.sqrt(np.sum(ir**2)), 1e-12)


def normalize_peak(ir: np.ndarray, peak_db: float = -0.3) -> np.ndarray:
    peak = np.max(np.abs(ir))
    return ir * (10 ** (peak_db / 20) / max(peak, 1e-12))


def write_wav(path: str, ir: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    sf.write(path, normalize_peak(ir).astype(np.float32), sr, subtype="PCM_24")


def to_mono(x: np.ndarray) -> np.ndarray:
    """Collapse a (n,) or (n, channels) buffer to mono by averaging channels."""
    return x if x.ndim == 1 else x.mean(axis=1)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """High-quality resample to ``sr_out`` (no-op if already there)."""
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out)


def align_onset(ir: np.ndarray, thresh_db: float = -40.0, lead: int = 0) -> np.ndarray:
    """Trim leading silence/pre-delay: drop everything before the first sample
    whose level crosses ``thresh_db`` below peak, keeping ``lead`` samples of run-up."""
    peak = float(np.max(np.abs(ir)))
    if peak <= 0:
        return ir
    thr = peak * 10 ** (thresh_db / 20)
    onset = int(np.argmax(np.abs(ir) >= thr))
    return ir[max(0, onset - lead):]


def fit_length(ir: np.ndarray, n: int = N_TAPS) -> np.ndarray:
    """Trim or zero-pad to exactly ``n`` samples."""
    if len(ir) >= n:
        return ir[:n]
    out = np.zeros(n, dtype=ir.dtype)
    out[: len(ir)] = ir
    return out


def normalize_ir(ir: np.ndarray, *, minphase: bool = False) -> np.ndarray:
    """Full target normalization: window tail -> (optional min-phase) -> unit energy.

    Assumes the input is already mono, 48 kHz, onset-aligned and length-fit.
    With ``minphase=True`` the IR is replaced by the deterministic min-phase
    reconstruction of its own magnitude (the design's default training target;
    keep an ``is_minphase`` flag to A/B against the raw-phase version)."""
    ir = fade_tail(ir.astype(np.float64))
    if minphase:
        mag = np.abs(np.fft.rfft(ir, n=N_TAPS))
        ir = minphase_from_magnitude(mag)
    return normalize_energy(ir).astype(np.float32)


def log_spectral_distance(mag_a: np.ndarray, mag_b: np.ndarray) -> float:
    """RMS distance in dB between two magnitude responses (100 Hz - 12 kHz band)."""
    n_bins = mag_a.shape[-1]
    freqs = np.linspace(0, SR / 2, n_bins)
    band = (freqs >= 100) & (freqs <= 12_000)
    db_a = 20 * np.log10(np.maximum(mag_a[..., band], 1e-9))
    db_b = 20 * np.log10(np.maximum(mag_b[..., band], 1e-9))
    return float(np.sqrt(np.mean((db_a - db_b) ** 2)))
