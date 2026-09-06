#!/usr/bin/env python
"""
Materialise the Dickerson-zoo label set for Auditor check 5 (D5, resolved
2026-09-03): write data/dickerson_zoo/names.csv from the 108 factor column
headers of data/development/dickerson_factor_returns.parquet.

Provenance chain (2026-09-03 + citations_verified.md §2): the
parquet was pulled by scripts/download_dickerson_factors.py (2026-09-01); its
column headers are taken VERBATIM from the ``single_sort_exc_all.csv`` member of
the source zip, lowercased at ingest. This script drops the ``year_month`` time
column and writes the remaining labels UNCHANGED (no further normalisation --
the matcher, agents/auditor/checks/mt_flag.py, casefolds both sides at compare
time instead). Deterministic: same parquet -> byte-identical csv.

The csv lives under /data/ (machine-local, gitignored with the licensed tree);
THIS generator + the parquet's recorded sha are the reproducibility anchor.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PARQUET = _REPO_ROOT / "data" / "development" / "dickerson_factor_returns.parquet"
_OUT = _REPO_ROOT / "data" / "dickerson_zoo" / "names.csv"
_TIME_COLUMN = "year_month"


def main() -> int:
    import pandas as pd

    if not _PARQUET.exists():
        print(f"[dickerson-names] source parquet missing: {_PARQUET}", file=sys.stderr)
        return 1
    sha = hashlib.sha256(_PARQUET.read_bytes()).hexdigest()
    columns = list(pd.read_parquet(_PARQUET).columns)
    if _TIME_COLUMN not in columns:
        print(f"[dickerson-names] expected time column {_TIME_COLUMN!r} absent; "
              f"refusing (wrong source file?)", file=sys.stderr)
        return 1
    names = [c for c in columns if c != _TIME_COLUMN]
    if len(names) != 108:
        print(f"[dickerson-names] expected 108 labels, found {len(names)}; "
              "refusing (the zoo count is part of the registered claim)", file=sys.stderr)
        return 1

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dickerson zoo labels (Auditor check 5, D5) -- GENERATED, do not hand-edit.",
        f"# source: {_PARQUET.relative_to(_REPO_ROOT)} (sha256 {sha})",
        "# regenerate: ./.venv/bin/python scripts/build_dickerson_names.py",
        "name",
        *names,
    ]
    _OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[dickerson-names] wrote {len(names)} labels -> {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
