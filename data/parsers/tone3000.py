"""Parser for manually approved TONE3000 cab IR imports.

This parser is intentionally narrow. TONE3000 metadata is community-authored
and inconsistent, so only tone IDs with reviewed naming grammar are parsed.

Approved grammar for tone 45023:
    <model_id>_V30 <speaker_pos> 4FB 4x12 SM57 <offset>in <distance>in [OA30] <preamp>.wav

Interpretation from the tone description and model-name samples:
  - first inch token: radial offset from dust-cap center
  - second inch token: mic distance from grille
  - OA30: 30 degrees off axis; absent = on axis

Raw audio remains local-only under data/raw/tone3000 and must not be
redistributed.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from cabir.labels import MM_PER_INCH, Label, PackConfig

NAN = math.nan

CONFIG = PackConfig(
    pack="tone3000",
    cab="",
    speaker="",
    distance_ref="grille",
    signatures=["tone3000", "45023"],
    path_include=[],
    path_exclude=[],
    mics={},
    positions={},
)

_TONE_META = {
    "45023": {
        "pack": "tone3000_45023",
        "cab": "Mesa Boogie 4FB Traditional Straight 4x12",
        "speaker": "Celestion Vintage 30",
        "mics": {"SM57": "sm57"},
    },
}

_NAME_RE = re.compile(
    r"^(?:(?P<model_id>\d+)_)?"
    r"(?P<speaker>\S+)\s+"
    r"(?P<speaker_pos>LL|LR|UL|UR)\s+"
    r"(?P<cab_code>4FB)\s+4x12\s+"
    r"(?P<mic>[A-Za-z0-9]+)\s+"
    r"(?P<offset>\d+(?:\.\d+)?)in\s+"
    r"(?P<distance>\d+(?:\.\d+)?)in"
    r"(?:\s+OA(?P<off_axis>\d+(?:\.\d+)?))?"
    r"(?:\s+(?P<chain>.*))?$",
    re.I,
)


def parse(rel_path: str, cfg: PackConfig = CONFIG) -> Label | None:
    parts = rel_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        return None

    tone_id = next((part for part in parts if part in _TONE_META), None)
    if tone_id is None:
        return None
    meta = _TONE_META[tone_id]

    stem = Path(parts[-1]).stem
    match = _NAME_RE.match(stem)
    if not match:
        return None

    mic_token = match.group("mic").upper()
    mic = meta["mics"].get(mic_token)
    if mic is None:
        return None

    offset_mm = round(float(match.group("offset")) * MM_PER_INCH, 2)
    distance_mm = round(float(match.group("distance")) * MM_PER_INCH, 2)
    angle = float(match.group("off_axis") or 0.0)

    return Label(
        pack=meta["pack"],
        cab=meta["cab"],
        speaker=meta["speaker"],
        mic=mic,
        distance_mm=distance_mm,
        distance_ref=cfg.distance_ref,
        offset_mm=offset_mm,
        angle_deg=angle,
        axis="off" if angle else "on",
        capture_type="close",
        ts=False,
        presence=NAN,
        file=rel_path,
    )
