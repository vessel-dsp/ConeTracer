"""Gradio test UI for real cab IR checkpoints.

Run:
    python app/gradio_app.py --ckpt runs/real_p3/best.pt
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from PIL import Image
from starlette.templating import Jinja2Templates

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir import N_TAPS, SR
from cabir.dsp import normalize_peak, resample_to, write_wav
from cabir.infer import generate_real_ir, load_real_checkpoint

DEFAULT_CKPT = REPO_ROOT / "runs" / "real_pcond_h256" / "best.pt"
DEFAULT_LABELS = REPO_ROOT / "data" / "parsed" / "labels.parquet"
DEFAULT_IRS = REPO_ROOT / "data" / "parsed" / "irs.npy"
DEFAULT_TEST_AUDIO = REPO_ROOT / "data" / "test_audio" / "inputs"
AUDIO_EXTS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg"}


class AppState:
    def __init__(self, ckpt_path: Path, labels_path: Path, irs_path: Path, test_audio_dir: Path = DEFAULT_TEST_AUDIO):
        self.ckpt_path = ckpt_path
        self.labels_path = labels_path
        self.irs_path = irs_path
        self.test_audio_dir = test_audio_dir
        self.ckpt, self.model = load_real_checkpoint(ckpt_path)
        self.labels = pd.read_parquet(labels_path)
        self.irs = np.load(irs_path, mmap_mode="r")
        self.test_audio = self._load_test_audio()

    @property
    def cabs(self) -> list[str]:
        return list(self.ckpt["cabs"])

    @property
    def mics(self) -> list[str]:
        return list(self.ckpt["mics"])

    def _load_test_audio(self) -> dict[str, str]:
        if not self.test_audio_dir.is_dir():
            return {}
        files = [
            p for p in self.test_audio_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        return {
            p.relative_to(self.test_audio_dir).as_posix(): str(p)
            for p in sorted(files, key=lambda x: x.relative_to(self.test_audio_dir).as_posix())
        }


STATE: AppState | None = None


def _patch_starlette_template_response() -> None:
    """Allow Gradio 3.x to run with Starlette's newer TemplateResponse signature."""
    original = Jinja2Templates.TemplateResponse
    if getattr(original, "_cabir_compat", False):
        return

    def compat(self, *args, **kwargs):
        if args and isinstance(args[0], str):
            name = args[0]
            context = args[1] if len(args) > 1 else kwargs.pop("context", None)
            request = (context or {}).get("request")
            return original(self, request, name, context, **kwargs)
        return original(self, *args, **kwargs)

    compat._cabir_compat = True
    Jinja2Templates.TemplateResponse = compat


def _nearest_real_ir(cab: str, mic: str, distance: float, offset: float, angle: float):
    assert STATE is not None
    df = STATE.labels
    mask = (
        (df["cab"] == cab)
        & (df["mic"] == mic)
        & (df["capture_type"] == "close")
        & (df["ts"] == bool(STATE.ckpt.get("ts", False)))
    )
    presence = _default_presence()
    if presence is not None:
        mask &= df["presence"] == float(presence)
    rows = df[mask].copy()
    if rows.empty:
        return None, None

    dist_norm = float(STATE.ckpt["dist_norm"])
    off_norm = float(STATE.ckpt["off_norm"])
    rows["_d"] = (
        (np.log1p(rows["distance_mm"].astype(float)) / dist_norm - math.log1p(distance) / dist_norm) ** 2
        + (rows["offset_mm"].astype(float) / off_norm - offset / off_norm) ** 2
        + (rows["angle_deg"].astype(float) / 90.0 - angle / 90.0) ** 2
    )
    row = rows.sort_values("_d").iloc[0]
    return np.asarray(STATE.irs[int(row["index"])], dtype=np.float64), row


def _default_presence() -> float | None:
    assert STATE is not None
    presence = STATE.ckpt.get("presence")
    return 3.0 if presence is None and STATE.ckpt.get("condition_presence") else presence


def _waveform_plot(model_ir: np.ndarray, real_ir: np.ndarray | None, real_label: str):
    t_ms = np.arange(len(model_ir)) / SR * 1000.0
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.plot(t_ms, model_ir, label="model", linewidth=1.2)
    if real_ir is not None:
        ax.plot(t_ms, real_ir, label=real_label, linewidth=0.9, alpha=0.75)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(model_ir) / SR * 1000.0)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _magnitude_plot(model_ir: np.ndarray, real_ir: np.ndarray | None, real_label: str):
    freqs = np.fft.rfftfreq(N_TAPS, 1.0 / SR)

    def db(ir):
        mag = np.abs(np.fft.rfft(ir, n=N_TAPS))
        return 20 * np.log10(np.maximum(mag, 1e-8))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.semilogx(freqs[1:], db(model_ir)[1:], label="model", linewidth=1.4)
    if real_ir is not None:
        ax.semilogx(freqs[1:], db(real_ir)[1:], label=real_label, linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(60, 20_000)
    ax.set_ylim(-75, 20)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def _empty_plot(message: str):
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    return fig


def _fig_to_image(fig) -> np.ndarray:
    tmp = tempfile.NamedTemporaryFile(prefix="cabir_plot_", suffix=".png", delete=False)
    tmp.close()
    fig.savefig(tmp.name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    image = np.asarray(Image.open(tmp.name).convert("RGB"))
    Path(tmp.name).unlink(missing_ok=True)
    return image


def _empty_image(message: str) -> np.ndarray:
    return _fig_to_image(_empty_plot(message))


def _spectrogram_compare(dry: np.ndarray | None, rendered: np.ndarray | None):
    if dry is None or rendered is None:
        return _empty_image("Choose test audio or upload a clip to show spectrograms.")

    from scipy.signal import stft

    n = min(len(dry), len(rendered), 20 * SR)
    dry = dry[:n]
    rendered = rendered[:n]
    if n < 2048:
        return _empty_image("Selected clip is too short for a spectrogram.")

    f, t, dry_stft = stft(dry, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
    _, _, rendered_stft = stft(rendered, fs=SR, nperseg=2048, noverlap=1536, boundary=None)
    dry_db = 20 * np.log10(np.maximum(np.abs(dry_stft), 1e-8))
    rendered_db = 20 * np.log10(np.maximum(np.abs(rendered_stft), 1e-8))
    diff_db = np.clip(rendered_db - dry_db, -18, 18)

    freq_mask = f <= 12_000
    f_khz = f[freq_mask] / 1000.0
    dry_db = dry_db[freq_mask]
    rendered_db = rendered_db[freq_mask]
    diff_db = diff_db[freq_mask]
    vmax = np.percentile(np.concatenate([dry_db.ravel(), rendered_db.ravel()]), 99)
    vmin = vmax - 80

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax, data, title in [
        (axes[0], dry_db, "Original"),
        (axes[1], rendered_db, "Rendered through model IR"),
    ]:
        im = ax.pcolormesh(t, f_khz, data, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_ylabel("kHz")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, pad=0.01, label="dB")

    im = axes[2].pcolormesh(t, f_khz, diff_db, shading="auto", cmap="coolwarm", vmin=-18, vmax=18)
    axes[2].set_ylabel("kHz")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_title("Rendered - original")
    fig.colorbar(im, ax=axes[2], pad=0.01, label="dB")
    fig.tight_layout()
    return _fig_to_image(fig)


def _mean_spectrum_db(audio: np.ndarray, n_fft: int = 4096, hop: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    if len(audio) < n_fft:
        padded = np.zeros(n_fft, dtype=np.float64)
        padded[: len(audio)] = audio
        audio = padded
    win = np.hanning(n_fft)
    frames = []
    for start in range(0, max(1, len(audio) - n_fft + 1), hop):
        frame = audio[start : start + n_fft]
        if len(frame) < n_fft:
            break
        frames.append(np.abs(np.fft.rfft(frame * win)))
    if not frames:
        frames = [np.abs(np.fft.rfft(audio[:n_fft] * win))]
    mag = np.mean(np.stack(frames), axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    return freqs, 20 * np.log10(np.maximum(mag, 1e-8))


def _smooth_curve(y: np.ndarray, bins: int = 25) -> np.ndarray:
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(y, kernel, mode="same")


def _spectrum_delta_image(dry: np.ndarray | None, rendered: np.ndarray | None, title: str):
    if dry is None or rendered is None:
        return _empty_image("Choose test audio or upload a clip to show spectral delta.")
    n = min(len(dry), len(rendered), 20 * SR)
    dry = dry[:n]
    rendered = rendered[:n]
    freqs, dry_db = _mean_spectrum_db(dry)
    _, rendered_db = _mean_spectrum_db(rendered)
    delta = np.clip(rendered_db - dry_db, -36, 36)
    smooth_delta = _smooth_curve(delta, bins=31)
    mask = (freqs >= 60) & (freqs <= 12_000)

    bands = [
        ("80-150", 80, 150),
        ("150-300", 150, 300),
        ("300-800", 300, 800),
        ("0.8-2k", 800, 2000),
        ("2-5k", 2000, 5000),
        ("5-10k", 5000, 10_000),
    ]
    band_vals = []
    for _, lo, hi in bands:
        band_mask = (freqs >= lo) & (freqs < hi)
        band_vals.append(float(np.mean(delta[band_mask])) if np.any(band_mask) else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].semilogx(freqs[mask], smooth_delta[mask], color="#2563eb", linewidth=2.0)
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[0].fill_between(freqs[mask], 0, smooth_delta[mask], where=smooth_delta[mask] >= 0,
                         color="#ef4444", alpha=0.25)
    axes[0].fill_between(freqs[mask], 0, smooth_delta[mask], where=smooth_delta[mask] < 0,
                         color="#2563eb", alpha=0.2)
    axes[0].set_xlim(60, 12_000)
    axes[0].set_ylim(-24, 24)
    axes[0].set_ylabel("dB")
    axes[0].set_title(title)
    axes[0].grid(True, which="both", alpha=0.2)

    labels = [b[0] for b in bands]
    colors = ["#ef4444" if v >= 0 else "#2563eb" for v in band_vals]
    axes[1].bar(labels, band_vals, color=colors, alpha=0.78)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].set_ylabel("avg dB")
    axes[1].set_ylim(-18, 18)
    axes[1].grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    return _fig_to_image(fig)


def _mag_db(ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(N_TAPS, 1.0 / SR)
    mag = np.abs(np.fft.rfft(ir, n=N_TAPS))
    return freqs, 20 * np.log10(np.maximum(mag, 1e-8))


def _gain_normalize_db(freqs: np.ndarray, db: np.ndarray, lo: float = 100, hi: float = 6000) -> np.ndarray:
    band = (freqs >= lo) & (freqs <= hi)
    ref = float(np.median(db[band])) if np.any(band) else float(np.median(db))
    return db - ref


def _model_vs_real_ir_image(model_ir: np.ndarray, real_ir: np.ndarray | None):
    if real_ir is None:
        return _empty_image("No nearest real IR found for this setting."), float("nan"), float("nan")

    freqs, model_db = _mag_db(model_ir)
    _, real_db = _mag_db(real_ir)
    model_norm = _gain_normalize_db(freqs, model_db)
    real_norm = _gain_normalize_db(freqs, real_db)
    delta = np.clip(model_norm - real_norm, -36, 36)
    smooth_delta = _smooth_curve(delta, bins=31)
    guitar_band = (freqs >= 100) & (freqs <= 6000)
    full_band = (freqs >= 100) & (freqs <= 12_000)
    lsd_guitar = float(np.sqrt(np.mean(delta[guitar_band] ** 2))) if np.any(guitar_band) else float("nan")
    lsd_full = float(np.sqrt(np.mean(delta[full_band] ** 2))) if np.any(full_band) else float("nan")

    mask = (freqs >= 60) & (freqs <= 12_000)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.2), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].semilogx(freqs[mask], model_norm[mask], label="model IR", linewidth=1.8)
    axes[0].semilogx(freqs[mask], real_norm[mask], label="nearest real IR", linewidth=1.4, alpha=0.8)
    axes[0].set_xlim(60, 12_000)
    axes[0].set_ylim(-60, 30)
    axes[0].set_ylabel("dB, gain-normalized")
    axes[0].set_title("Model IR vs nearest real IR")
    axes[0].grid(True, which="both", alpha=0.2)
    axes[0].legend(loc="lower left")

    axes[1].semilogx(freqs[mask], smooth_delta[mask], color="#2563eb", linewidth=2.0)
    axes[1].axhline(0, color="black", linewidth=0.8, alpha=0.45)
    axes[1].fill_between(freqs[mask], 0, smooth_delta[mask], where=smooth_delta[mask] >= 0,
                         color="#ef4444", alpha=0.25)
    axes[1].fill_between(freqs[mask], 0, smooth_delta[mask], where=smooth_delta[mask] < 0,
                         color="#2563eb", alpha=0.2)
    axes[1].set_xlim(60, 12_000)
    axes[1].set_ylim(-24, 24)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("model - real dB")
    axes[1].grid(True, which="both", alpha=0.2)
    fig.tight_layout()
    return _fig_to_image(fig), lsd_guitar, lsd_full


def _ir_error_metrics(model_ir: np.ndarray, real_ir: np.ndarray) -> tuple[float, float]:
    freqs, model_db = _mag_db(model_ir)
    _, real_db = _mag_db(real_ir)
    model_norm = _gain_normalize_db(freqs, model_db)
    real_norm = _gain_normalize_db(freqs, real_db)
    delta = model_norm - real_norm
    guitar_band = (freqs >= 100) & (freqs <= 6000)
    full_band = (freqs >= 100) & (freqs <= 12_000)
    lsd_guitar = float(np.sqrt(np.mean(delta[guitar_band] ** 2))) if np.any(guitar_band) else float("nan")
    lsd_full = float(np.sqrt(np.mean(delta[full_band] ** 2))) if np.any(full_band) else float("nan")
    return lsd_guitar, lsd_full


def _validation_rows(cab: str, mic: str) -> pd.DataFrame:
    assert STATE is not None
    df = STATE.labels
    mask = (
        (df["cab"] == cab)
        & (df["mic"] == mic)
        & (df["capture_type"] == "close")
        & (df["ts"] == bool(STATE.ckpt.get("ts", False)))
    )
    presence = _default_presence()
    if presence is not None:
        mask &= df["presence"] == float(presence)
    rows = df[mask].copy()
    if rows.empty:
        return pd.DataFrame(columns=["distance_mm", "offset_mm", "guitar_rms_db", "wide_rms_db", "file"])

    result_rows = []
    for _, row in rows.iterrows():
        model_ir = generate_real_ir(
            STATE.model, STATE.ckpt,
            cab=cab,
            mic=mic,
            distance_mm=float(row.distance_mm),
            offset_mm=float(row.offset_mm),
            angle_deg=float(row.angle_deg),
            presence=float(row.presence) if not np.isnan(row.presence) else _default_presence(),
        )
        real_ir = np.asarray(STATE.irs[int(row["index"])], dtype=np.float64)
        guitar_lsd, full_lsd = _ir_error_metrics(model_ir, real_ir)
        result_rows.append({
            "distance_mm": float(row.distance_mm),
            "offset_mm": float(row.offset_mm),
            "angle_deg": float(row.angle_deg),
            "guitar_rms_db": guitar_lsd,
            "wide_rms_db": full_lsd,
            "file": row.file,
        })
    return pd.DataFrame(result_rows)


def _validation_grid_image(results: pd.DataFrame, metric: str):
    if results.empty:
        return _empty_image("No captured positions found for this cab/mic.")

    pivot = results.pivot_table(
        index="distance_mm",
        columns="offset_mm",
        values=metric,
        aggfunc="mean",
    ).sort_index().sort_index(axis=1)

    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(data, cmap="viridis_r", vmin=0, vmax=max(6.0, np.nanmax(data)), aspect="auto")
    ax.set_title(f"Validation grid: {metric.replace('_', ' ')} (yellow/green = stronger, blue/purple = weaker)")
    ax.set_xlabel("Offset from dust-cap center (mm)")
    ax.set_ylabel("Distance from grille (mm)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{x:g}" for x in pivot.index])
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            if np.isfinite(data[y, x]):
                color = "black" if data[y, x] < 3 else "white"
                ax.text(x, y, f"{data[y, x]:.1f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("RMS dB error / confidence, lower is better")
    fig.tight_layout()
    return _fig_to_image(fig)


def _read_audio(path: str):
    audio, sr = sf.read(path, dtype="float64", always_2d=False)
    audio = audio.mean(axis=1) if audio.ndim == 2 else audio
    return resample_to(audio, sr, SR)


def _fft_convolve_same(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    from scipy.signal import fftconvolve

    return fftconvolve(audio, ir, mode="full")[: len(audio)]


def _convolve_audio(audio: np.ndarray, ir: np.ndarray) -> np.ndarray:
    wet = _fft_convolve_same(audio, ir)
    return normalize_peak(wet, peak_db=-1.0).astype(np.float32)


def _selected_audio_path(test_audio_name: str | None, upload_path: str | None) -> str | None:
    assert STATE is not None
    return upload_path or (STATE.test_audio.get(test_audio_name) if test_audio_name else None)


def _crossfade_chunks(chunks: list[np.ndarray], fade_len: int) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    out = chunks[0].astype(np.float64)
    for chunk in chunks[1:]:
        chunk = chunk.astype(np.float64)
        fade = min(fade_len, len(out), len(chunk))
        if fade <= 0:
            out = np.concatenate([out, chunk])
            continue
        fade_in = np.linspace(0.0, 1.0, fade, endpoint=False)
        fade_out = 1.0 - fade_in
        overlap = out[-fade:] * fade_out + chunk[:fade] * fade_in
        out = np.concatenate([out[:-fade], overlap, chunk[fade:]])
    return out


def render(
    cab: str,
    mic: str,
    distance: float,
    offset: float,
    angle: float,
    test_audio_name: str | None,
    upload_path: str | None,
):
    assert STATE is not None
    model_ir = generate_real_ir(
        STATE.model, STATE.ckpt,
        cab=cab, mic=mic, distance_mm=distance, offset_mm=offset, angle_deg=angle,
        presence=_default_presence(),
    )
    real_ir, row = _nearest_real_ir(cab, mic, distance, offset, angle)
    real_label = "nearest real"
    if row is not None:
        real_label = f"nearest real: {row.distance_mm:.1f}mm / {row.offset_mm:.1f}mm"

    wav = tempfile.NamedTemporaryFile(prefix="cabir_", suffix=".wav", delete=False)
    wav.close()
    write_wav(wav.name, model_ir)

    model_audio = None
    real_audio = None
    spec_plot = _empty_image("Choose test audio or upload a clip to show spectrograms.")
    delta_plot = _empty_image("Choose test audio or upload a clip to show spectral delta.")
    di_path = _selected_audio_path(test_audio_name, upload_path)
    if di_path:
        dry = _read_audio(di_path)
        model_wet = _convolve_audio(dry, model_ir)
        model_audio = (SR, model_wet)
        spec_plot = _spectrogram_compare(dry, model_wet)
        delta_plot = _spectrum_delta_image(dry, model_wet, "Rendered - original average spectrum")
        if real_ir is not None:
            real_audio = (SR, _convolve_audio(dry, real_ir))

    status = f"Generated {mic} at {distance:.1f}mm, offset {offset:.1f}mm, {angle:.0f} deg"
    if row is not None:
        status += f" | nearest real file: {row.file}"
    return (
        _waveform_plot(model_ir, real_ir, real_label),
        _magnitude_plot(model_ir, real_ir, real_label),
        wav.name,
        model_audio,
        real_audio,
        spec_plot,
        delta_plot,
        status,
    )


def render_sweep(
    cab: str,
    mic: str,
    distance: float,
    offset: float,
    angle: float,
    test_audio_name: str | None,
    upload_path: str | None,
    start_offset: float,
    end_offset: float,
    seconds: float,
    steps: int,
):
    assert STATE is not None
    di_path = _selected_audio_path(test_audio_name, upload_path)
    if not di_path:
        empty = _empty_image("Choose a test clip or upload DI audio first.")
        return None, None, empty, empty, "Choose a test clip or upload DI audio first."

    dry = _read_audio(di_path)
    max_len = max(1, int(seconds * SR))
    dry = dry[:max_len]
    if len(dry) < SR // 2:
        empty = _empty_image("Selected clip is too short for a sweep.")
        return None, None, empty, empty, "Selected clip is too short for a sweep."

    steps = max(2, int(steps))
    segment_len = max(1, math.ceil(len(dry) / steps))
    offsets = np.linspace(start_offset, end_offset, steps)
    chunks = []
    for i, sweep_offset in enumerate(offsets):
        seg = dry[i * segment_len : min(len(dry), (i + 1) * segment_len)]
        if len(seg) == 0:
            continue
        ir = generate_real_ir(
            STATE.model, STATE.ckpt,
            cab=cab, mic=mic, distance_mm=distance, offset_mm=float(sweep_offset), angle_deg=angle,
            presence=_default_presence(),
        )
        chunks.append(_fft_convolve_same(seg, ir))

    fade_len = min(int(0.05 * SR), max(1, segment_len // 4))
    sweep = normalize_peak(_crossfade_chunks(chunks, fade_len), peak_db=-1.0).astype(np.float32)
    static_ir = generate_real_ir(
        STATE.model, STATE.ckpt,
        cab=cab, mic=mic, distance_mm=distance, offset_mm=offset, angle_deg=angle,
        presence=_default_presence(),
    )
    static = _convolve_audio(dry, static_ir)[: len(sweep)]
    spec_plot = _spectrogram_compare(dry[: len(sweep)], sweep)
    delta_plot = _spectrum_delta_image(dry[: len(sweep)], sweep, "Mic sweep - original average spectrum")
    status = (
        f"Sweep rendered: {mic}, {distance:.1f}mm, "
        f"offset {start_offset:.1f}->{end_offset:.1f}mm over {len(dry) / SR:.1f}s"
    )
    return (SR, sweep), (SR, static), spec_plot, delta_plot, status


def validate_model(
    cab: str,
    mic: str,
    distance: float,
    offset: float,
    angle: float,
    test_audio_name: str | None,
    upload_path: str | None,
):
    assert STATE is not None
    model_ir = generate_real_ir(
        STATE.model, STATE.ckpt,
        cab=cab, mic=mic, distance_mm=distance, offset_mm=offset, angle_deg=angle,
        presence=_default_presence(),
    )
    real_ir, row = _nearest_real_ir(cab, mic, distance, offset, angle)
    ir_plot, lsd_guitar, lsd_full = _model_vs_real_ir_image(model_ir, real_ir)

    audio_delta = _empty_image("Choose test audio or upload a clip for rendered-audio delta.")
    model_audio = None
    real_audio = None
    diff_audio = None
    if real_ir is not None:
        di_path = _selected_audio_path(test_audio_name, upload_path)
        if di_path:
            dry = _read_audio(di_path)
            model_wet = _convolve_audio(dry, model_ir)
            real_wet = _convolve_audio(dry, real_ir)
            n = min(len(model_wet), len(real_wet))
            model_wet = model_wet[:n]
            real_wet = real_wet[:n]
            diff = normalize_peak(model_wet - real_wet, peak_db=-1.0).astype(np.float32)
            audio_delta = _spectrum_delta_image(real_wet, model_wet, "ConeTrace-rendered - nearest-real-rendered audio")
            model_audio = (SR, model_wet)
            real_audio = (SR, real_wet)
            diff_audio = (SR, diff)

    status = (
        f"IR shape error: {lsd_guitar:.2f} dB RMS (100 Hz-6 kHz), "
        f"{lsd_full:.2f} dB RMS (100 Hz-12 kHz)."
    )
    if row is not None:
        status += f" Nearest real: {row.file}"
    else:
        status += " No nearest real IR found."
    return ir_plot, audio_delta, model_audio, real_audio, diff_audio, status


def validate_grid(cab: str, mic: str, metric_label: str):
    metric = "wide_rms_db" if metric_label.startswith("Wide") else "guitar_rms_db"
    results = _validation_rows(cab, mic)
    if results.empty:
        return (
            _empty_image("No captured positions found for this cab/mic."),
            pd.DataFrame(),
            "No captured positions found for this cab/mic.",
        )

    heatmap = _validation_grid_image(results, metric)
    mean_guitar = float(results["guitar_rms_db"].mean())
    worst_guitar = float(results["guitar_rms_db"].max())
    mean_wide = float(results["wide_rms_db"].mean())
    worst_wide = float(results["wide_rms_db"].max())
    summary = (
        f"{len(results)} captured positions | "
        f"100 Hz-6 kHz mean {mean_guitar:.2f} dB, worst {worst_guitar:.2f} dB | "
        f"100 Hz-12 kHz mean {mean_wide:.2f} dB, worst {worst_wide:.2f} dB"
    )
    table = results.sort_values("guitar_rms_db", ascending=False).copy()
    table["distance_mm"] = table["distance_mm"].round(1)
    table["offset_mm"] = table["offset_mm"].round(1)
    table["angle_deg"] = table["angle_deg"].round(1)
    table["guitar_rms_db"] = table["guitar_rms_db"].round(2)
    table["wide_rms_db"] = table["wide_rms_db"].round(2)
    return heatmap, table, summary


def build_app(state: AppState) -> gr.Blocks:
    global STATE
    STATE = state
    default_cab = state.cabs[0]
    default_mic = "sm57" if "sm57" in state.mics else state.mics[0]
    test_audio_choices = [""] + list(state.test_audio.keys())
    default_test_audio = (
        "jazz-hop-guitar.wav"
        if "jazz-hop-guitar.wav" in state.test_audio
        else (test_audio_choices[1] if len(test_audio_choices) > 1 else "")
    )

    with gr.Blocks(title="ConeTrace") as demo:
        gr.Markdown(
            "# ConeTrace\n"
            "Continuous mic-position cabinet IR modeling."
        )
        with gr.Row():
            with gr.Column(scale=1):
                cab = gr.Dropdown(state.cabs, value=default_cab, label="Cab")
                mic = gr.Dropdown(state.mics, value=default_mic, label="Mic")
                distance = gr.Slider(0, 80, value=25, step=0.1, label="Distance from grille (mm)")
                offset = gr.Slider(0, 83, value=32, step=0.1, label="Offset from dust-cap center (mm)")
                angle = gr.Slider(0, 90, value=0, step=1, label="Off-axis angle (deg)")
                test_audio = gr.Dropdown(test_audio_choices, value=default_test_audio, label="Test audio")
                di = gr.Audio(source="upload", type="filepath", label="Upload DI clip")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.Tab("Normal render"):
                        render_btn = gr.Button("Render", variant="primary")
                        status = gr.Textbox(label="Status")
                        wav = gr.File(label="Generated IR WAV")
                        wave = gr.Plot(label="Impulse response")
                        mag = gr.Plot(label="Magnitude response")
                        model_audio = gr.Audio(label="ConeTrace IR audition", type="numpy")
                        real_audio = gr.Audio(label="Nearest real IR audition", type="numpy")
                        delta = gr.Image(label="Average spectrum delta", type="numpy")
                        spectrogram = gr.Image(label="Original vs rendered spectrogram", type="numpy")

                    with gr.Tab("Mic sweep"):
                        with gr.Row():
                            sweep_start = gr.Slider(0, 83, value=0, step=0.1, label="Start offset (mm)")
                            sweep_end = gr.Slider(0, 83, value=83, step=0.1, label="End offset (mm)")
                        with gr.Row():
                            sweep_seconds = gr.Slider(2, 20, value=8, step=0.5, label="Sweep length (s)")
                            sweep_steps = gr.Slider(3, 24, value=12, step=1, label="IR steps")
                        sweep_btn = gr.Button("Render Sweep", variant="primary")
                        sweep_status = gr.Textbox(label="Sweep status")
                        sweep_audio = gr.Audio(label="Mic-move sweep", type="numpy")
                        static_audio = gr.Audio(label="Static reference", type="numpy")
                        sweep_delta = gr.Image(label="Mic sweep average spectrum delta", type="numpy")
                        sweep_spectrogram = gr.Image(label="Original vs mic-move sweep spectrogram", type="numpy")

                    with gr.Tab("Model vs real"):
                        validate_btn = gr.Button("Validate Against Nearest Real", variant="primary")
                        validation_status = gr.Textbox(label="Validation status")
                        ir_validation = gr.Image(label="ConeTrace IR vs nearest real IR", type="numpy")
                        audio_validation = gr.Image(label="ConeTrace-rendered vs nearest-real-rendered spectrum delta", type="numpy")
                        validation_model_audio = gr.Audio(label="ConeTrace-rendered audio", type="numpy")
                        validation_real_audio = gr.Audio(label="Nearest-real-rendered audio", type="numpy")
                        validation_diff_audio = gr.Audio(label="Difference / null audio", type="numpy")

                    with gr.Tab("Validation grid"):
                        grid_metric = gr.Radio(
                            ["Guitar band 100 Hz-6 kHz", "Wide band 100 Hz-12 kHz"],
                            value="Guitar band 100 Hz-6 kHz",
                            label="Metric",
                        )
                        grid_btn = gr.Button("Run Grid Validation", variant="primary")
                        grid_status = gr.Textbox(label="Grid summary")
                        grid_heatmap = gr.Image(label="Position-grid error heatmap", type="numpy")
                        grid_table = gr.Dataframe(label="Worst positions first")

        inputs = [cab, mic, distance, offset, angle, test_audio, di]
        outputs = [wave, mag, wav, model_audio, real_audio, spectrogram, delta, status]
        render_btn.click(render, inputs=inputs, outputs=outputs)
        demo.load(render, inputs=inputs, outputs=outputs)
        sweep_btn.click(
            render_sweep,
            inputs=[
                cab, mic, distance, offset, angle, test_audio, di,
                sweep_start, sweep_end, sweep_seconds, sweep_steps,
            ],
            outputs=[sweep_audio, static_audio, sweep_spectrogram, sweep_delta, sweep_status],
        )
        validate_btn.click(
            validate_model,
            inputs=[cab, mic, distance, offset, angle, test_audio, di],
            outputs=[
                ir_validation,
                audio_validation,
                validation_model_audio,
                validation_real_audio,
                validation_diff_audio,
                validation_status,
            ],
        )
        grid_btn.click(
            validate_grid,
            inputs=[cab, mic, grid_metric],
            outputs=[grid_heatmap, grid_table, grid_status],
        )
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--irs", type=Path, default=DEFAULT_IRS)
    ap.add_argument("--test-audio", type=Path, default=DEFAULT_TEST_AUDIO)
    ap.add_argument("--server-name", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    args = ap.parse_args()

    _patch_starlette_template_response()
    state = AppState(args.ckpt, args.labels, args.irs, args.test_audio)
    app = build_app(state)
    app.launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
