"""End-to-end self-test for the IR ingest pipeline — synthetic fixtures, no
downloads. Fabricates folders matching the REAL filename grammars (God's Cab
from its manual; Redwirez from documented naming), runs `cabir.ingest`, and
asserts labels / capture types / path-filtering / quarantine / stored IRs.

    python scripts/selftest_ingest.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from cabir import N_TAPS, SR  # noqa: E402
from cabir.ingest import ingest  # noqa: E402
from cabir.labels import read_parquet  # noqa: E402

# (relative path under raw, source sample rate, channels)
FIXTURES = [
    # --- God's Cab: real grammar; only the 48 folder should ingest ---
    ("Gods_Cab_1.4/48/SM57/NO-TS/57_1_inch_cap_pres_1.wav", 44100, 1),       # close
    ("Gods_Cab_1.4/48/SM57/NO-TS/57_grill_edge_pres_3.wav", 44100, 1),       # grill=0
    ("Gods_Cab_1.4/48/SM57/NO-TS/57_1_inch_cone_far_pres_2.wav", 44100, 1),  # cone_far
    ("Gods_Cab_1.4/48/SM57/TS/57_TS_2_inch_cone_near_pres_5.wav", 44100, 1), # TS + cone_near
    ("Gods_Cab_1.4/48/C414/NO-TS/C414_1_foot_pres_1.wav", 44100, 1),         # distant
    ("Gods_Cab_1.4/48/U87/NO-TS/U87_2_feet_pres_3.wav", 44100, 1),           # distant
    ("Gods_Cab_1.4/48/NT5/Mono/NO-TS/NT5_left_dead_room_pres_1.wav", 44100, 1),    # room
    ("Gods_Cab_1.4/48/NT5/Stereo/NO-TS/NT5_stereo_live_room_pres_4.wav", 44100, 2),  # room (stereo)
    # duplicate sample-rate copies -> must be SKIPPED by path filter
    ("Gods_Cab_1.4/44.1/SM57/NO-TS/57_1_inch_cap_pres_1.wav", 44100, 1),
    ("Gods_Cab_1.4/96/SM57/NO-TS/57_1_inch_cap_pres_1.wav", 96000, 1),
    # deprecated legacy mic -> must be SKIPPED by path_exclude
    ("Gods_Cab_1.4/48/1.0_Legacy_IRs/M3/NO-TS/M3_room_1.wav", 44100, 1),
    # --- Redwirez: documented grammar via the generic parser ---
    ("redwirez_1960a/SM57/SM57 Cap 0in.wav", 48000, 1),
    ("redwirez_1960a/R121/R121 Cone-Edge 4in OffAxis.wav", 96000, 2),
    ("redwirez_1960a/SM57/SM57 Mystery Take.wav", 48000, 1),  # -> quarantine
]


def _fake_ir(sr: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(0.04 * sr)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
    ir = np.concatenate([np.zeros(int(0.002 * sr)), ir])  # leading silence to trim
    ir /= np.max(np.abs(ir))
    return ir if channels == 1 else np.column_stack([ir, ir])


def _row(df, substr):
    return df[df["file"].str.replace("\\", "/").str.contains(substr, regex=False)].iloc[0]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw, out = tmp / "raw", tmp / "parsed"
        for rel, sr, ch in FIXTURES:
            p = raw / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(p), _fake_ir(sr, ch), sr, subtype="PCM_24")
        # a Fractal .syx dump (wrong extension) and a manual must be ignored
        (raw / "Gods_Cab_1.4" / "AXE-FX" / "SM57").mkdir(parents=True)
        (raw / "Gods_Cab_1.4" / "AXE-FX" / "SM57" / "57.syx").write_bytes(b"\xf0\x00\xf7")
        (raw / "Gods_Cab_1.4" / "Gods_Cab_Manual.pdf").write_bytes(b"%PDF-1.4\n")

        summary = ingest(raw, REPO_ROOT / "data" / "parsers", out, minphase=False)

        print("\n--- assertions ---")
        # 8 God's Cab (48, non-legacy) + 2 Redwirez = 10 parsed; 1 quarantine;
        # 3 skipped (44.1, 96, legacy). The .syx isn't audio so never counted.
        assert summary["parsed"] == 10, f"parsed={summary['parsed']}"
        assert summary["quarantined"] == 1, f"quarantined={summary['quarantined']}"
        assert summary["skipped"] == 3, f"skipped={summary['skipped']}"
        assert not summary["unknown_packs"], summary["unknown_packs"]

        df = read_parquet(out / "labels.parquet")
        assert len(df) == 10 and df["index"].tolist() == list(range(10))
        assert set(df["pack"]) == {"godscab", "redwirez_1960a"}

        # God's Cab close-mic: distance/offset/presence/ts from the real grammar
        cap = _row(df, "57_1_inch_cap_pres_1")
        assert cap["mic"] == "sm57" and cap["cab"] == "Mesa Oversized Rectifier 4x12"
        assert abs(cap["distance_mm"] - 25.4) < 0.01 and cap["offset_mm"] == 0.0
        assert cap["capture_type"] == "close" and not cap["ts"] and cap["presence"] == 1.0

        grill = _row(df, "57_grill_edge_pres_3")
        assert grill["distance_mm"] == 0.0 and grill["offset_mm"] == 32.0 and grill["presence"] == 3.0

        ts = _row(df, "57_TS_2_inch_cone_near")
        assert ts["ts"] and abs(ts["distance_mm"] - 50.8) < 0.01
        assert ts["offset_mm"] == 57.0 and ts["presence"] == 5.0

        far = _row(df, "57_1_inch_cone_far")
        assert far["offset_mm"] == 83.0 and far["capture_type"] == "close"

        # distant capture: feet distance, no lateral offset
        dist = _row(df, "C414_1_foot")
        assert dist["capture_type"] == "distant" and abs(dist["distance_mm"] - 304.8) < 0.01
        assert math.isnan(dist["offset_mm"])

        # room capture: no distance, no offset
        room = _row(df, "NT5_left_dead_room")
        assert room["capture_type"] == "room"
        assert math.isnan(room["distance_mm"]) and math.isnan(room["offset_mm"])

        # Redwirez via generic parser: off-axis + cone-edge, presence absent
        r = _row(df, "R121 Cone-Edge 4in OffAxis")
        assert r["mic"] == "r121" and r["axis"] == "off" and r["angle_deg"] == 45.0
        assert r["offset_mm"] == 110.0 and abs(r["distance_mm"] - 4 * 25.4) < 0.01
        assert r["capture_type"] == "close" and math.isnan(r["presence"])

        # stored IRs: shape, dtype, finite, unit energy, resampled to 48k
        irs = np.load(out / "irs.npy")
        assert irs.shape == (10, N_TAPS) and irs.dtype == np.float32, (irs.shape, irs.dtype)
        assert np.isfinite(irs).all()
        energies = np.sqrt((irs.astype(np.float64) ** 2).sum(axis=1))
        assert np.allclose(energies, 1.0, atol=1e-3), energies
        assert (df["sr"] == SR).all() and (df["n_samples"] == N_TAPS).all()

        cov = (out / "coverage.md").read_text()
        assert "Mesa Oversized Rectifier 4x12" in cov and "capture type" in cov

    print("PASS — real-grammar ingest verified (10 parsed, 1 quarantined, 3 skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
