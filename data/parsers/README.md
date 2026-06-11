# IR pack parsers

One module per pack. Each exposes a `CONFIG` of type `cabir.labels.PackConfig`
and, optionally, a custom `parse(rel_path, cfg) -> Label | None`. The ingest
driver (`python -m cabir.ingest`) discovers every `*.py` here, detects which pack
a folder under `data/raw/` is (via `CONFIG.signatures`), applies the config's
path filters, then labels each remaining file — using the custom `parse()` if the
module defines one, else the generic `cabir.labels.parse_path`. Files that don't
yield a complete label are **quarantined**, never guessed.

- **Declarative pack** (e.g. `redwirez_1960a.py`): just a `CONFIG` with `mics`
  (alias → canonical name) and `positions` (token → `offset_mm`, cap = 0). The
  generic `parse_path` matches mic + distance + position tokens.
- **Custom-grammar pack** (e.g. `godscab.py`): `CONFIG` for the constants/filters
  plus a `parse()` that handles pack-specific tokens (presence, Tube-Screamer,
  grill/inch/foot distances, close/distant/room capture types).

`CONFIG.path_include` / `path_exclude` filter the raw tree *before* parsing — use
them to keep one sample rate and drop duplicate-rate copies, hardware dumps
(`.syx`), and deprecated folders (these are skipped, not quarantined).

## Add a parser for a new pack

1. Create `data/parsers/<pack>.py` with `CONFIG = PackConfig(...)` (set
   `signatures`, `path_include`/`path_exclude`, `mics`, `positions`).
2. If the filename grammar is richer than mic+distance+position, add a
   `parse(rel_path, cfg)` returning a `cabir.labels.Label` (or `None`).
3. Drop the raw pack in `data/raw/<pack>/`, run `python -m cabir.ingest`, and
   check `data/parsed/quarantine/<pack>.txt` — anything there is a token the
   parser doesn't know yet. Add it and re-run.

`offset_mm` values are **nominal** and a candidate for per-pack calibration
(distance/position conventions differ across packs — see design §1 Risk).
`distance_ref` records what the distance is measured from so two packs are never
blended into one position grid without alignment.
