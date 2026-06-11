"""PyTorch Dataset over parsed labels.parquet + irs.npy (M2).

Usage:
    ds = IRDataset(cab="Mesa Oversized Rectifier 4x12", mics=["sm57","sm7b","c414"])
    loader = DataLoader(ds, batch_size=32, shuffle=True)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import N_TAPS

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data/parsed/labels.parquet"
DEFAULT_IRS = REPO_ROOT / "data/parsed/irs.npy"


class IRDataset(Dataset):
    """Log-magnitude spectra dataset for the conditional spectral decoder.

    Each item: categorical conditioning (mic, cab) + continuous conditioning
    (distance, offset, angle) + target log-magnitude spectrum (N_TAPS/2+1 bins).

    Normalization constants (dist_norm, off_norm) are computed from the full
    filtered dataset so train and val splits share the same scale.
    """

    N_BINS: int = N_TAPS // 2 + 1  # 2049

    def __init__(
        self,
        parquet: Path = DEFAULT_PARQUET,
        irs: Path = DEFAULT_IRS,
        *,
        cab: str | None = None,
        mics: list[str] | None = None,
        ts: bool | None = False,        # None = include all ts values
        presence: float | None = None,  # None = include all presence values
        capture_type: str | None = "close",
        val_frac: float = 0.2,
        split: str = "train",           # "train" | "val" | "all"
        seed: int = 42,
    ):
        df = pd.read_parquet(parquet)

        if cab:
            df = df[df.cab == cab]
        if mics:
            df = df[df.mic.isin(mics)]
        if ts is not None:
            df = df[df.ts == ts]
        if presence is not None:
            df = df[df.presence.isna() | (df.presence == presence)]
        if capture_type is not None:
            df = df[df.capture_type == capture_type]
        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                "IRDataset: no rows after filtering. "
                f"cab={cab!r} mics={mics!r} ts={ts} presence={presence} capture_type={capture_type!r}"
            )

        # Vocabulary (from full filtered set — consistent across splits)
        self.mics: list[str] = sorted(df.mic.unique().tolist())
        self.cabs: list[str] = sorted(df.cab.unique().tolist())
        self.mic2idx: dict[str, int] = {m: i for i, m in enumerate(self.mics)}
        self.cab2idx: dict[str, int] = {c: i for i, c in enumerate(self.cabs)}

        # Normalization references (computed from full set before split)
        self.dist_norm: float = math.log1p(300.0)   # log(1+300mm) ≈ log(1+room distance)
        max_off = float(df["offset_mm"].max())
        self.off_norm: float = max_off if (not np.isnan(max_off) and max_off > 0) else 1.0

        # Train / val split — hold out ENTIRE POSITIONS per mic so val samples
        # represent genuinely unseen (distance, offset) coordinates, not just
        # unseen presence variants of training positions.
        if split == "all":
            keep = df.index.tolist()
        else:
            rng = np.random.default_rng(seed)
            val_rows, train_rows = [], []
            for mic in self.mics:
                mic_df = df[df.mic == mic]
                # group by unique (distance_mm, offset_mm) position key
                pos_groups = mic_df.groupby(["distance_mm", "offset_mm"], sort=False)
                pos_keys = list(pos_groups.groups.keys())
                rng.shuffle(pos_keys)
                n_val = max(1, round(len(pos_keys) * val_frac))
                val_pos = set(pos_keys[:n_val])
                for key, idx_list in pos_groups.groups.items():
                    (val_rows if key in val_pos else train_rows).extend(idx_list.tolist())
            keep = val_rows if split == "val" else train_rows

        self.df = df.loc[keep].reset_index(drop=True)
        ir_array = np.load(irs, mmap_mode="r")
        self._irs = np.asarray(ir_array[self.df["index"].values], dtype=np.float32)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        ir = self._irs[idx]   # (N_TAPS,) float32

        # Target: log-magnitude spectrum
        spec = np.abs(np.fft.rfft(ir))                      # (N_BINS,)
        logmag = np.log(spec + 1e-8).astype(np.float32)    # (N_BINS,)

        dist_mm = float(row.distance_mm) if not np.isnan(row.distance_mm) else 25.0
        off_mm = float(row.offset_mm) if not np.isnan(row.offset_mm) else 0.0
        presence = float(row.presence) if not np.isnan(row.presence) else 3.0

        return {
            "mic_idx":  torch.tensor(self.mic2idx[row.mic], dtype=torch.long),
            "cab_idx":  torch.tensor(self.cab2idx[row.cab], dtype=torch.long),
            "distance": torch.tensor(math.log1p(dist_mm) / self.dist_norm, dtype=torch.float32),
            "offset":   torch.tensor(off_mm / self.off_norm, dtype=torch.float32),
            "angle":    torch.tensor(float(row.angle_deg) / 90.0, dtype=torch.float32),
            "presence": torch.tensor((presence - 1.0) / 4.0, dtype=torch.float32),
            "logmag":   torch.from_numpy(logmag),
            "ir":       torch.from_numpy(ir.copy()),
        }
