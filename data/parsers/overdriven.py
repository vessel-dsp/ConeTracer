"""Overdriven.fr free guitar cab IRs — all SSP2-series packs.

One parser handles all 11 packs.  Grammar:
    OD-{CAB}-{MIC}-P{OFFSET_MM}-{DIST_MM}[-SUFFIX].wav
inside {PACK}/SSP2/{MIC_DIR}/

Offset (P00…P25): mm from dust-cap centre.
Distance: mm from speaker grille.
capture_type: close ≤ 50 mm, distant > 50 mm.

Terms (readme.txt): "free for musical/video creations, any commercial or
non-commercial purpose."  Derivative works (model weights + generated IRs)
are permitted.

Skipped via path_exclude (silently, not quarantined):
  sp1-preview     blend/preview mix track, not a raw capture
  -ua610          different preamp chain (UA610 ≠ SSP2)
  -eq.            post-EQ applied, not raw
  -cut1 / -cut2   filter variants with unclear cutoff parameters
  -p1.            inverted-polarity copy (-P0 / no suffix = canonical)

Kept:  plain, -P0, -L (left speaker of a 2x12, confirmed mono),
       -LC / -LC45 / -LC75 (standard HPF on ribbon mics).
"""
from __future__ import annotations

import math
import re

from cabir.labels import Label, PackConfig

NAN = math.nan

CONFIG = PackConfig(
    pack="overdriven",
    cab="",       # set per-file from pack folder lookup
    speaker="",   # set per-file from pack folder lookup
    distance_ref="grille",
    mics={},      # resolved per-file from MIC_DIR name
    positions={}, # resolved from P[NN] token
    signatures=["overdriven-"],
    path_include=["ssp2/"],
    path_exclude=["sp1-preview", "-ua610", "-eq.", "-cut1", "-cut2", "-p1."],
)

# (folder_substring, pack_code, cab, speaker) — first match wins; all keys are unique
_PACK_META: list[tuple[str, str, str, str]] = [
    ("de-112-t75",  "od_e112_t75",   "DE-112 1x12 ported",     "T-75"),
    ("e112-bvv",    "od_e112_bvv",   "E112 1x12 ported",       "M. BV30V"),
    ("e112-cv",     "od_e112_cv",    "DE-112 1x12",            "Celestion V-Type"),
    ("e112-gm",     "od_e112_gm",    "DE-112 1x12 ported",     "Grey Mojo"),
    ("e112-k100",   "od_e112_k100",  "German 112 1x12 ported", "K-100"),
    ("e112-p50e",   "od_e112_p50e",  "E112 1x12 ported",       "P50E"),
    ("e112-vt",     "od_e112_vt",    "DE-112 1x12",            "V-Type"),
    ("fb-vet30",    "od_fb_vet30",   "FATBOY-212",             "WGS VET30"),
    ("m212-gb55",   "od_m212_gb55",  "M-212 2x12",             "GB 55"),
    ("m212-p50",    "od_m212_p50",   "M212 2x12",              "E. P50"),
    ("m212-vint",   "od_m212_vint",  "M212 2x12",              "G12 Vintage"),
]

# MIC_DIR name → canonical mic label (overdriven code kept for unknowns)
_MIC_MAP: dict[str, str] = {
    "DYN-57":    "sm57",
    "DYN-57-P":  "sm57",
    "DYN-421":   "md421",
    "DYN-906":   "e906",
    "DYN-835-P": "e835",
    "DYN-58":    "sm58",
    "DYN-201":   "m201",
    "DYN-US-6":  "dyn_us6",
    "DYN-US-8":  "dyn_us8",
    "RBN-160":   "m160",
    "RBN-160-P": "m160",
    "RBN-US-1":  "r121",
    "RBN-CN-1":  "rbn_cn1",
    "RBN-CN-2":  "rbn_cn2",
    "CND-AU-1":  "cnd_au1",
    "CND-M3":    "cnd_m3",
    "CND-M3-P":  "cnd_m3",
    "CND-JP-1":  "cnd_jp1",
    "CND-2020":  "at2020",
}

_POS_RE = re.compile(r"(?i)-P(\d{2})-(\d+)")


def parse(rel_path: str, cfg: PackConfig = CONFIG) -> Label | None:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        return None

    mic_dir = parts[-2]
    filename = parts[-1].rsplit(".", 1)[0]
    pack_folder = next((p for p in parts if p.lower().startswith("overdriven-")), "")

    mic = _MIC_MAP.get(mic_dir)
    if not mic:
        return None

    folder_lower = pack_folder.lower()
    meta = next(
        ((code, cab, spk) for key, code, cab, spk in _PACK_META if key in folder_lower),
        None,
    )
    if meta is None:
        return None
    pack_code, cab, speaker = meta

    m = _POS_RE.search(filename)
    if not m:
        return None
    offset_mm = float(m.group(1))
    distance_mm = float(m.group(2))

    return Label(
        pack=pack_code,
        cab=cab,
        speaker=speaker,
        mic=mic,
        distance_mm=distance_mm,
        distance_ref="grille",
        offset_mm=offset_mm,
        angle_deg=0.0,
        axis="on",
        capture_type="close" if distance_mm <= 50 else "distant",
        ts=False,
        presence=NAN,
        file=rel_path,
    )
