# data/raw — drop IR packs here

Raw IR packs are **training inputs only** and are gitignored (license). Download
each pack and unzip it into its own folder here, then run the ingest driver:

```bash
python -m cabir.ingest          # -> data/parsed/{labels.parquet, irs.npy, coverage.md}
```

The driver matches a folder to a parser by name signatures (see
`data/parsers/`), so keep the pack name in the folder. Layout:

```
data/raw/
├── redwirez_1960a/     # Redwirez Marshall 1960A free pack (unzipped)
├── godscab/            # God's Cab v1.4 (unzipped)
└── <your_pack>/        # add a parser in data/parsers/ to ingest it
```

## Where to get the packs

| Folder | Pack | Source | License — CHECK BEFORE TRAINING |
|---|---|---|---|
| `Gods_Cab_1.4` | God's Cab v1.4 (Mesa OS Rectifier 4×12, V30) — **659 IRs** | Signals Audio / Wilkinson Audio (free, donationware) | free; weights-only output, never re-ship raw |
| `Overdriven-*-SSP2-v1.0` | 11 overdriven.fr SSP2-series packs (11 cabs, up to 5 mics) — **500 IRs** | overdriven.fr (free, direct download) | "free for musical/video creations, any commercial or non-commercial purpose" ✓ derivative works permitted |
| `redwirez_1960a` | Marshall 1960A free pack (G12M-25) | redwirez.com (free, email signup) | free w/ signup; verify redistribution terms |
| `ownhammer_*` | OwnHammer free packs (Cheapie / Rock-Box / 112) | ownhammer.com | free; terms per pack |
| `tone3000_*` | TONE3000 community IRs | tone3000.com/api | API ToS; labels are loose — parse descriptions |

**Note on duplicate sample rates / formats:** packs often ship the same IRs at
44.1/48/96 kHz plus hardware-specific dumps (e.g. God's Cab Axe-FX `.syx`). The
parser's `path_include`/`path_exclude` keep **only the 48 kHz audio** and skip
the rest — no manual pruning needed. `coverage.md` reports what was skipped.

**Not a position-grid pack:** `Simpulse_Freddy` (also from Wilkinson) is a small
set of pre-mixed IRs whose filenames (`Simpulse_ONE_TWO`, …) encode no mic /
distance / on-cone position, so it can't be labeled into the `(cab, mic,
distance, offset, angle)` schema and is reported as an unknown pack. Leave it out
of training, or treat it as a separate "mixes" asset.

**License gate (design §1, Risk):** before training on a pack, confirm its
written terms permit derivative works. We distribute model weights + generated
IRs only, never the raw captures. Flag any pack whose terms forbid derivatives.

## TONE3000 discovery

TONE3000 is not a normal unzip-and-parse pack. Start with metadata-only
discovery:

```bash
TONE3000_ACCESS_TOKEN=... python scripts/tone3000_discover.py --limit 100
python scripts/tone3000_make_audit.py
TONE3000_ACCESS_TOKEN=... python scripts/tone3000_enrich_audit.py
TONE3000_ACCESS_TOKEN=... python scripts/tone3000_import_approved.py --dry-run
```

This writes ignored local files under `data/tone3000/` and estimates how many
IR-like records have enough mic/distance/offset text to be labelable. The
enrichment step records tone/model metadata and model-name samples only; it
does not download model files. The guarded importer only downloads rows after
manual approval (`usable_for_grid=yes`, `license_tos_ok=yes`,
`download_models_ok=yes`) and defaults to dry-run mode. Do not train on
TONE3000 data until API access, Terms, and label quality are reviewed.

The current approved local probe is tone `45023` only. It is parsed by
`data/parsers/tone3000.py`; generated probe outputs under
`data/parsed_tone3000_probe/` are ignored.

## After ingest

- `data/parsed/coverage.md` — where each (cab, mic) grid is dense vs sparse.
- `data/parsed/quarantine/<pack>.txt` — files whose filenames didn't yield a
  complete label, with the missing field noted. Add the unknown token to that
  pack's parser config and re-run; nothing is ever label-guessed.
