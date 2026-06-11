"""Label schema + filename-parsing framework for cab IR packs (Workstream 1).

A pack's parser lives in ``data/parsers/`` and exposes a ``CONFIG`` (a
:class:`PackConfig` with the cab/speaker constants, filename vocabulary, and
path filters). Simple packs rely on the generic :func:`parse_path`; packs with
a richer grammar (e.g. God's Cab presence/Tube-Screamer/room captures) also
export a custom ``parse(rel_path, cfg)`` function.

Ingest is **lossless**: every documented label dimension is recorded (mic,
distance, position/offset, angle, plus `capture_type`, `ts`, `presence`) so the
downstream torch Dataset can filter or condition on them. Anything a parser
can't fully resolve returns ``None`` and is *quarantined*, never guessed.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Stored-IR contract (everything is normalized to this before training).
from . import N_TAPS, SR  # noqa: F401  (re-exported for parser convenience)

MM_PER_INCH = 25.4
MM_PER_FOOT = 304.8
NAN = math.nan

# Parquet column order: physical label, then capture metadata, then audio provenance.
LABEL_COLUMNS = [
    "index",
    "pack",
    "cab",
    "speaker",
    "mic",
    "distance_mm",
    "distance_ref",   # what distance is measured FROM: "grille" | "cap"
    "offset_mm",      # lateral position, dust-cap center = 0; NaN for distant/room
    "angle_deg",
    "axis",           # "on" | "off"
    "capture_type",   # "close" | "distant" | "room"
    "ts",             # bool: a tone-shaping pedal (Tube Screamer) was in the chain
    "presence",       # power-amp presence index baked into the IR (NaN if pack has none)
    "sr",             # stored sample rate (always 48000)
    "n_samples",      # stored length (always 4096)
    "src_sr",         # original file sample rate (provenance)
    "is_minphase",    # True if the stored IR is the min-phase reconstruction
    "file",           # path relative to data/raw (provenance / round-trip)
]


@dataclass
class Label:
    """Physical + capture label derived purely from a filename/path."""

    pack: str
    cab: str
    speaker: str
    mic: str
    distance_mm: float
    distance_ref: str          # "grille" | "cap"
    offset_mm: float           # NaN when not a cap→edge position (distant/room)
    angle_deg: float
    axis: str                  # "on" | "off"
    capture_type: str          # "close" | "distant" | "room"
    ts: bool
    presence: float            # NaN when the pack has no presence dimension
    file: str                  # relative path under data/raw

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class PackConfig:
    """Pack constants + filename vocabulary + which raw paths to ingest.

    ``mics`` / ``positions`` keys are matched case-insensitively as whole
    space-delimited tokens, longest-match-wins. ``path_include`` (all must be
    present) and ``path_exclude`` (any present → skip) filter the raw tree
    *before* parsing, so duplicate sample-rate copies, hardware-format dumps,
    and deprecated folders are skipped rather than quarantined.
    """

    pack: str
    cab: str
    speaker: str
    distance_ref: str                       # convention this pack measures distance from
    mics: dict[str, str]                    # filename alias -> canonical mic name
    positions: dict[str, float]             # filename token -> offset_mm (cap=0)
    signatures: list[str] = field(default_factory=list)   # substrings that ID the pack folder
    path_include: list[str] = field(default_factory=list)  # rel-path must contain ALL of these
    path_exclude: list[str] = field(default_factory=list)  # rel-path containing ANY of these is skipped
    default_off_axis_angle: float = 45.0

    def matches(self, folder_name: str) -> bool:
        name = folder_name.lower()
        return any(sig in name for sig in self.signatures)

    def path_ok(self, rel_path: str) -> bool:
        p = rel_path.lower().replace("\\", "/")
        if any(x in p for x in self.path_exclude):
            return False
        return all(inc in p for inc in self.path_include)


# --- token helpers (shared by generic and custom parsers) -------------------

_SEP = re.compile(r"[\\/_\-\s]+")
_DOT_NOT_DECIMAL = re.compile(r"(?<!\d)\.|\.(?!\d)")  # dots except those inside a number
_DISTANCE = re.compile(r"(\d+(?:\.\d+)?)\s*(inches|inch|in|\"|feet|foot|ft|mm|cm)\b")
_ANGLE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:degrees|degree|degs|deg|°)\b")


def normalize(rel_path: str) -> str:
    """Lowercase, drop the extension, and turn separators into spaces so the path
    becomes one space-delimited token stream. Decimal points are preserved (so
    ``0.5in`` stays intact) while extension/separator dots become spaces."""
    stem = rel_path[: rel_path.rfind(".")] if "." in Path(rel_path).name else rel_path
    s = _DOT_NOT_DECIMAL.sub(" ", stem.lower())
    return f" {_SEP.sub(' ', s).strip()} "


def longest_token_match(norm: str, vocab: dict) -> tuple[str, object] | None:
    """Return the (alias, value) for the longest vocab key present as a token run."""
    best: tuple[str, object] | None = None
    for alias, value in vocab.items():
        if f" {alias} " in norm and (best is None or len(alias) > len(best[0])):
            best = (alias, value)
    return best


def distance_mm(norm: str) -> float | None:
    m = _DISTANCE.search(norm)
    if not m:
        return None
    value, unit = float(m.group(1)), m.group(2)
    if unit in ('"', "in", "inch", "inches"):
        return value * MM_PER_INCH
    if unit in ("ft", "foot", "feet"):
        return value * MM_PER_FOOT
    if unit == "cm":
        return value * 10.0
    return value  # mm


def parse_path(rel_path: str, cfg: PackConfig) -> Label | None:
    """Generic filename -> Label for close-mic packs (mic + distance + position).
    Returns ``None`` if any of those three is missing."""
    norm = normalize(rel_path)

    mic = longest_token_match(norm, cfg.mics)
    pos = longest_token_match(norm, cfg.positions)
    dist = distance_mm(norm)
    if mic is None or pos is None or dist is None:
        return None

    # axis / angle: trust only an explicit degree marker or on/off keyword so
    # speaker model numbers (57, 421, 906...) are never misread as angles.
    angle_m = _ANGLE.search(norm)
    explicit_off = " off " in norm or " offaxis " in norm
    explicit_on = " on " in norm or " onaxis " in norm
    if angle_m:
        angle = float(angle_m.group(1))
        axis = "on" if angle == 0 and not explicit_off else "off"
    elif explicit_off:
        angle, axis = cfg.default_off_axis_angle, "off"
    else:
        angle, axis = 0.0, "on"
    if explicit_on and not angle_m:
        angle, axis = 0.0, "on"

    return Label(
        pack=cfg.pack, cab=cfg.cab, speaker=cfg.speaker, mic=mic[1],
        distance_mm=round(dist, 2), distance_ref=cfg.distance_ref,
        offset_mm=float(pos[1]), angle_deg=angle, axis=axis,
        capture_type="close", ts=False, presence=NAN, file=rel_path,
    )


def missing_fields(rel_path: str, cfg: PackConfig) -> list[str]:
    """Which required fields a filename failed to yield ([] if parseable by the
    generic parser). Drives the quarantine report so unknown tokens are easy to add."""
    norm = normalize(rel_path)
    miss = []
    if longest_token_match(norm, cfg.mics) is None:
        miss.append("mic")
    if longest_token_match(norm, cfg.positions) is None:
        miss.append("position")
    if distance_mm(norm) is None:
        miss.append("distance")
    return miss


# --- parquet IO -------------------------------------------------------------

def write_parquet(rows: list[dict], path: str | Path):
    import pandas as pd

    df = pd.DataFrame(rows, columns=LABEL_COLUMNS)
    df.to_parquet(path, engine="pyarrow", index=False)
    return df


def read_parquet(path: str | Path):
    import pandas as pd

    return pd.read_parquet(path, engine="pyarrow")


# --- coverage report --------------------------------------------------------

def coverage_report(df) -> str:
    """Markdown report of position-grid coverage per (cab, mic) — tells us where
    interpolation is data-supported vs extrapolation (design §1, Risk: sparse grids)."""
    import pandas as pd

    lines = [
        "# Coverage report",
        "",
        f"Total IRs: **{len(df)}**  ·  packs: {', '.join(sorted(df['pack'].unique())) or '—'}",
        "",
    ]
    if df.empty:
        lines.append("_No labeled IRs yet — drop packs in `data/raw/` and re-run `python -m cabir.ingest`._")
        return "\n".join(lines)

    by_type = df["capture_type"].value_counts().to_dict()
    ts_n = int(df["ts"].sum())
    lines += [
        f"By capture type: {by_type}  ·  Tube-Screamer IRs: {ts_n}  ·  "
        f"plain: {len(df) - ts_n}",
        "",
        "> For the design's 4-D conditioning (mic, distance, offset, angle), a single "
        "(mic, distance, offset) can map to multiple targets that differ only by "
        "`presence`/`ts`. Filter to one (`ts=False`, one `presence`) in the dataset "
        "loader, or add them as conditioning inputs.",
        "",
    ]

    for (cab, mic), g in df.groupby(["cab", "mic"]):
        close = g[g["capture_type"] == "close"]
        lines += [
            f"## {cab} — {mic}  ({len(g)} IRs)",
            "",
            f"- capture types: {g['capture_type'].value_counts().to_dict()}",
            f"- presence values: {sorted(g['presence'].dropna().unique().tolist())}"
            f"  ·  ts: {g['ts'].value_counts().to_dict()}",
        ]
        if not close.empty:
            dists = sorted(close["distance_mm"].round(1).unique())
            offs = sorted(close["offset_mm"].dropna().round(1).unique())
            lines += [
                f"- close-mic distances (mm, from {close['distance_ref'].iloc[0]}): {dists}",
                f"- close-mic offsets (mm, cap=0): {offs}",
                "",
                "close-mic grid — distance ↓ \\ offset → (IR count):",
                "",
                pd.crosstab(close["distance_mm"].round(1), close["offset_mm"].round(1)).to_markdown(),
            ]
        lines.append("")
    return "\n".join(lines)
