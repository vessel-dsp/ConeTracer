"""Parser: God's Cab v1.4 (Signals Audio / Wilkinson) — Mesa Oversized Rectifier
4x12, Celestion V30. Grammar is from the bundled Gods_Cab_Manual.pdf.

Filename: ``<mic>[_TS]_<distance>_<position>_pres_<N>.wav`` (close mics), or
``<mic>[_TS]_<feet>_pres_<N>`` (distant), or ``<mic>[_TS]_[stereo|left|right]_<dead|live>_room_pres_<N>``.

Per the manual:
  - distance is measured from the grill cloth: ``grill`` = touching (0 mm),
    else ``N_inch`` (close) or ``N_foot/feet`` (distant).
  - position on the cone: ``cap`` (dust-cap center), ``edge`` (dust-cap edge),
    ``cone_near``, ``cone_far`` (~1" further out each).
  - ``TS`` = a Tube Screamer (TS9) was in the chain — a mid-boost EQ, NOT cab
    physics → recorded as the ``ts`` flag, not blended into the position grid.
  - ``pres_N`` (N=1..5) = the 6505+ power-amp presence (brightness) setting baked
    into the IR → recorded as ``presence`` (it changes magnitude but isn't a mic
    position). The dataset keeps all variants; the loader filters/conditions.

We ingest only the 48 kHz folder (44.1/96 are resampled duplicates — manual
§Sampling Rates) and skip the Axe-FX ``.syx`` dumps and the deprecated
``1.0_Legacy_IRs`` (M3/C02 mics). Offsets are NOMINAL radial mm from the manual
diagram (cap=0, then ~1" steps) — candidates for calibration.
"""
from cabir.labels import (
    MM_PER_FOOT,
    MM_PER_INCH,
    NAN,
    Label,
    PackConfig,
)

CONFIG = PackConfig(
    pack="godscab",
    cab="Mesa Oversized Rectifier 4x12",
    speaker="Celestion V30",
    distance_ref="grille",
    signatures=["god", "gods_cab", "gods cab"],
    path_include=["/48/"],            # one sample rate; 44.1 & 96 are duplicates
    path_exclude=["axe-fx", "legacy"],  # .syx hardware dumps + deprecated M3/C02 mics
    mics={"57": "sm57", "sm7b": "sm7b", "c414": "c414",
          "u87": "u87", "nt5": "nt5", "md421": "md421"},
    positions={"cap": 0.0, "edge": 32.0, "cone_near": 57.0, "cone_far": 83.0},
)

_MICS = {"57": "sm57", "sm7b": "sm7b", "c414": "c414",
         "u87": "u87", "nt5": "nt5", "md421": "md421"}
# nominal radial offset (mm) from dust-cap center, per the manual's position photo
_OFFSET = {"cap": 0.0, "edge": 32.0, "cone_near": 57.0, "cone_far": 83.0}
_UNIT_MM = {"inch": MM_PER_INCH, "inches": MM_PER_INCH, "foot": MM_PER_FOOT, "feet": MM_PER_FOOT}


def _take_distance(toks: list[str]) -> tuple[float, str, bool] | None:
    """Return (distance_mm, capture_kind, found). ``grill`` -> 0; N inch/foot otherwise."""
    if "grill" in toks or "grille" in toks:
        return 0.0, "close", True
    for i in range(len(toks) - 1):
        if toks[i].replace(".", "", 1).isdigit() and toks[i + 1] in _UNIT_MM:
            mm = float(toks[i]) * _UNIT_MM[toks[i + 1]]
            kind = "distant" if toks[i + 1] in ("foot", "feet") else "close"
            return mm, kind, True
    return None


def parse(rel_path: str, cfg: PackConfig = CONFIG) -> Label | None:
    toks = rel_path.split("/")[-1].rsplit(".", 1)[0].lower().split("_")

    mic = next((_MICS[t] for t in toks if t in _MICS), None)
    if mic is None:                                   # e.g. legacy M3/C02 -> quarantine
        return None

    ts = "ts" in toks
    presence = NAN
    if "pres" in toks:
        i = toks.index("pres")
        if i + 1 < len(toks) and toks[i + 1].isdigit():
            presence = float(toks[i + 1])

    if "room" in toks:                                # room/ambience capture
        return Label(cfg.pack, cfg.cab, cfg.speaker, mic, NAN, cfg.distance_ref,
                     NAN, 0.0, "on", "room", ts, presence, rel_path)

    dist = _take_distance(toks)
    if dist is None:
        return None
    distance_mm, kind, _ = dist

    pos = next((p for p in ("cone_near", "cone_far") if p.replace("_", " ") in
                rel_path.lower().replace("_", " ")), None)
    if pos is None:
        pos = next((p for p in ("cap", "edge") if p in toks), None)
    offset = _OFFSET[pos] if pos else NAN

    if kind == "close" and pos is None:               # close distance but no position token
        return None
    capture_type = "close" if (kind == "close" and pos) else "distant"
    return Label(cfg.pack, cfg.cab, cfg.speaker, mic, round(distance_mm, 2),
                 cfg.distance_ref, offset, 0.0, "on", capture_type, ts, presence, rel_path)
