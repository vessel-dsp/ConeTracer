"""Parser config: Redwirez Marshall 1960A free pack (G12M-25 Greenbacks).

Filenames encode mic + lateral position + distance, e.g.::

    1960A/SM57/SM57 Cap 0in.wav
    1960A/R121/R121 Cone-Edge 4in OffAxis.wav

⚠ VERIFY ON REAL FILES: the mic alias set and the position→offset_mm mapping
below are from Redwirez's *documented* naming; confirm the exact tokens against
the downloaded pack (run `python scripts/selftest_ingest.py` style spot-check on
a real folder) and adjust `mics` / `positions` here — nothing else changes.
The lateral offsets are NOMINAL mm and a candidate for per-pack calibration
(design §1 Risk: distance/position conventions differ across packs).
"""
from cabir.labels import PackConfig

CONFIG = PackConfig(
    pack="redwirez_1960a",
    cab="Marshall 1960A 4x12",
    speaker="Celestion G12M-25 Greenback",
    distance_ref="grille",
    signatures=["1960", "redwirez", "marshall"],
    # The free pack's documented 17-mic set (filename alias -> canonical name).
    mics={
        "sm57": "sm57", "57": "sm57",
        "sm7": "sm7", "sm7b": "sm7",
        "md421": "md421", "421": "md421", "md409": "md409", "409": "md409",
        "r121": "r121", "121": "r121", "royer": "r121",
        "km84": "km84", "u47": "u47", "u67": "u67", "u87": "u87",
        "c414": "c414", "414": "c414",
        "m160": "m160", "4038": "coles_4038", "re20": "re20",
        "pr30": "pr30", "d6": "d6", "i5": "audix_i5",
    },
    # position token -> nominal lateral offset in mm (dust-cap center = 0)
    positions={
        "cap": 0.0, "center": 0.0, "dustcap": 0.0, "dust cap": 0.0,
        "cap edge": 32.0, "capedge": 32.0, "cap-edge": 32.0,
        "cone": 65.0,
        "cone edge": 110.0, "coneedge": 110.0, "cone-edge": 110.0,
        "edge": 120.0,
    },
)
