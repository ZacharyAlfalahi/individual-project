"""
Download the DRR-2026 monthly long-short factor series (openbondassetpricing.com)
→ data/development/dickerson_factor_returns.parquet, truncated at the development
boundary BEFORE anything touches disk.

Source artifact (CONFIRM-ON-LOAD outcome recorded 2026-09-01, scope_changes.md):
  "108 Bond Factors — Dickerson, Robotti & Rossetti (2026)" → Bond Level Single
  Sort → 108 Wide Format Factors (single_sort_public.zip), member
  single_sort_exc_all.csv: all-bonds decile sorts, long-short P10-P1,
  value-weighted, EXCESS returns in DECIMAL units, month-end `date` column,
  sample 1973-02..2024-12. Single-sort matches the anchor spec §3.2/§9 (DRR-2026
  Table 2 Panel A is single-sort, not within-firm). Sign-corrected columns carry
  a trailing `*` (e.g. `str*` = -1 x the raw winners-minus-losers series);
  names are stored verbatim. The DRR-2023 replicated BBW factor series (MKTB/
  DRF/CRF/LRF) is NOT in this distribution — recorded, not worked around.

Window discipline: rows after December of (holdout_start_year - 1) are dropped
in memory at parse time — holdout-era rows are discarded unread and are never
written to disk. The stored artifact is therefore development-only by
construction, like baa_aaa_spread.parquet.

Mirrors scripts/download_baa_aaa_spread.py (curl + atomic parquet write).
DEVIATION (recorded in scope_changes.md): the endpoint URL is held as a module
constant, not a thresholds.yaml key, to keep this additive instrument zero-touch
on the frozen thresholds file (FROZEN_THRESHOLDS_SHA256 re-pin discipline).
DEVIATION from the BAA template: curl runs WITHOUT text=True and WITH -L —
the payload is a binary zip and the host redirects; do not "fix" either back.

Output columns:
  year_month  — str, e.g. "2015-06"
  <factor>    — float, long-short excess return in decimal units, names
                verbatim from the source header (108 columns incl. `str*`,
                `mom6_1`)

Usage:
  python scripts/download_dickerson_factors.py
"""

import io
import os
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_FILE = REPO_ROOT / "data" / "development" / "dickerson_factor_returns.parquet"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# openbondassetpricing.com → Data → Corporate Bond Factor Data → 108 Bond
# Factors (Dickerson, Robotti & Rossetti 2026) → Bond Level Single Sort →
# 108 Wide Format Factors. Accessed 2026-09-01.
DICKERSON_FACTORS_URL = (
    "https://openbondassetpricing.com/wp-content/uploads/2026/01/single_sort_public.zip"
)
# The zip ships six CSVs (exc/dur x all/ig/nig) + a README; the anchor-relevant
# one is the all-bonds decile-sort excess-return file (README: "single-sort,
# All bonds (decile sorts)"). Exact member name — never a heuristic pick.
ZIP_MEMBER = "single_sort_exc_all.csv"

_DATE_COLUMN_CANDIDATES = ("date", "yyyymm", "datadate", "month")


def fetch_bytes(url: str) -> bytes:
    """The only network function. Returns the raw payload bytes."""
    print(f"Downloading: {url}")
    # -f: fail (nonzero exit) on HTTP 4xx/5xx so an error page never reaches the
    # parser; -S: still show the error under -s; -L: follow redirects (WordPress
    # uploads); --retry 3: tolerate transient blips. NO text=True — binary zip.
    result = subprocess.run(
        ["curl", "-fsSL", "--retry", "3", "--max-time", "120", url],
        capture_output=True,
        check=True,
    )
    if not result.stdout:
        raise ValueError(f"empty payload from {url}")
    return result.stdout


def _parse_csv_bytes(raw: bytes) -> pd.DataFrame:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), na_values=["", "NA", "NaN"])


def _parse_member(name: str, raw: bytes) -> pd.DataFrame:
    if name.endswith(".csv"):
        return _parse_csv_bytes(raw)
    if name.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(raw))
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(raw))
    raise ValueError(f"unsupported member type: {name}")


def parse_payload(raw: bytes) -> pd.DataFrame:
    """Format discovery + parse + column normalisation. Pure (no I/O)."""
    if raw[:4] == b"PK\x03\x04":
        zf = zipfile.ZipFile(io.BytesIO(raw))
        names = zf.namelist()
        if "[Content_Types].xml" in names:
            # An xlsx is itself a zip — disambiguate before member selection.
            df = pd.read_excel(io.BytesIO(raw))
        elif ZIP_MEMBER in names:
            df = _parse_member(ZIP_MEMBER, zf.read(ZIP_MEMBER))
        else:
            candidates = [
                n for n in names
                if n.endswith((".csv", ".xlsx", ".parquet"))
                and not n.startswith(("__MACOSX/", "."))
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"expected member {ZIP_MEMBER!r} or exactly one data member; "
                    f"zip contains {names}"
                )
            df = _parse_member(candidates[0], zf.read(candidates[0]))
    else:
        df = _parse_csv_bytes(raw)

    if df.shape[1] < 2:
        snippet = raw[:200].decode("utf-8", errors="replace")
        raise ValueError(f"payload parsed to <2 columns — not the factor file: {snippet!r}")

    df.columns = [str(c).strip().lower() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise ValueError(f"duplicate column names after lowercasing: {sorted(df.columns)}")
    df = df.loc[:, [c for c in df.columns if not c.startswith("unnamed")]]
    print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
    return df


def _to_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """Find the date column, convert to a `year_month` string, put it first."""
    date_col = next((c for c in _DATE_COLUMN_CANDIDATES if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"no date column among {_DATE_COLUMN_CANDIDATES} in {list(df.columns)}"
        )
    s = df[date_col]
    if pd.api.types.is_numeric_dtype(s):
        # yyyymm integers/floats, e.g. 200408 / 200408.0
        parsed = pd.to_datetime(
            s.astype("Int64").astype(str), format="%Y%m", errors="coerce"
        )
    else:
        parsed = pd.to_datetime(s, errors="coerce")
    if parsed.isna().all():
        raise ValueError(f"date column {date_col!r} did not parse")
    keep = parsed.notna()
    out = df.loc[keep].drop(columns=[date_col]).copy()
    out.insert(0, "year_month", parsed[keep].dt.to_period("M").astype(str).values)
    return out.reset_index(drop=True)


def _dev_boundary_period() -> pd.Period:
    """Last development month = December of (holdout_start_year - 1). The
    development artefact must NOT contain holdout-era rows (window discipline);
    they are dropped in memory, unread, before the parquet is written."""
    with open(THRESHOLDS_FILE) as f:
        hsy = int(yaml.safe_load(f)["trace_cleaning"]["holdout_start_year"])
    return pd.Period(f"{hsy - 1}-12", freq="M")


def truncate_to_dev(df: pd.DataFrame, boundary: pd.Period) -> pd.DataFrame:
    """Drop rows after the development boundary. Pure; the tripwire target."""
    periods = pd.PeriodIndex(df["year_month"], freq="M")
    n_dropped = int((periods > boundary).sum())
    out = df[periods <= boundary]
    out = out.drop_duplicates(subset=["year_month"], keep="first")
    n_dupes = len(df) - n_dropped - len(out)
    if n_dupes:
        print(f"  Deduplicated {n_dupes} repeated month(s)")
    out = out.sort_values("year_month").reset_index(drop=True)
    if out.empty:
        raise ValueError(
            f"zero rows at/before the dev boundary {boundary} — stale or wrong source?"
        )
    assert (pd.PeriodIndex(out["year_month"], freq="M") <= boundary).all()
    print(f"  Discarded {n_dropped} post-{boundary} row(s) unread")
    return out


def build_parquet(raw: bytes) -> int:
    df = _to_year_month(parse_payload(raw))
    out = truncate_to_dev(df, _dev_boundary_period())
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE} ({len(out):,} rows)")
    print(f"  Date range: {out['year_month'].min()} to {out['year_month'].max()}")
    return len(out)


def main() -> None:
    n = build_parquet(fetch_bytes(DICKERSON_FACTORS_URL))
    print(f"\nDone. {n:,} monthly observations → {OUT_FILE}")


if __name__ == "__main__":
    main()
