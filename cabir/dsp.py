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


def log_spectral_distance(mag_a: np.ndarray, mag_b: np.ndarray) -> float:
    """RMS distance in dB between two magnitude responses (100 Hz - 12 kHz band)."""
    n_bins = mag_a.shape[-1]
    freqs = np.linspace(0, SR / 2, n_bins)
    band = (freqs >= 100) & (freqs <= 12_000)
    db_a = 20 * np.log10(np.maximum(mag_a[..., band], 1e-9))
    db_b = 20 * np.log10(np.maximum(mag_b[..., band], 1e-9))
    return float(np.sqrt(np.mean((db_a - db_b) ** 2)))
