"""
Download the macro conditioning series for the extension templates T1/T2 (vix, term_spread)
from FRED → data/development/{vix,term_spread}.parquet.

Series:
  vix          — VIXCLS (CBOE Volatility Index, daily) aggregated to a monthly mean.
  term_spread  — GS10 - GS3M (10y minus 3m constant-maturity Treasury yields, monthly, pp).

WARMING (F6). Full history is kept below the dev-boundary cap: VIX from 1990, GS3M from 1981,
GS10 from 1953 — all reach >= 1997-07, i.e. >= 60 months before the 2002-07 panel start, so the
templates' expanding_window_past_only(min 60) threshold is fully warm at the 2004-08 inference
start and effective T stays 209 for every template. This retains pre-sample macro data
(1997-2001) PURELY to warm a threshold — not a holdout violation (holdout is forward,
2022-01..2025-09 per SC-SCI-12, correcting SC-SCI-10; originally 2022-2024 at the tag);
see docs/data/registers/scope_changes.md F6. The upper cap is the dev boundary (no holdout-era
rows in a development artefact); the holdout macro slice is built in the one-shot holdout frozen inventory.

Mirrors scripts/download_rf_rate.py / download_baa_aaa_spread.py (public FRED CSV via curl, no
API key). URLs live in docs/thresholds.yaml (monthly_panel.*_url).

Usage:
  python scripts/download_macro_conditioning.py
"""

import io
import os
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "development"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def _cfg() -> dict:
    with open(THRESHOLDS_FILE) as f:
        return yaml.safe_load(f)["monthly_panel"]


def _dev_boundary() -> pd.Period:
    with open(THRESHOLDS_FILE) as f:
        hsy = int(yaml.safe_load(f)["trace_cleaning"]["holdout_start_year"])
    return pd.Period(f"{hsy - 1}-12", freq="M")


def _fred(url: str) -> pd.DataFrame:
    print(f"Downloading: {url}")
    # -f fails on HTTP error (no error page into read_csv); -S shows it under -s; --retry 3.
    result = subprocess.run(
        ["curl", "-fsS", "--retry", "3", "--max-time", "60", url],
        capture_output=True, text=True, check=True,
    )
    if not result.stdout.lstrip().startswith("observation_date"):
        raise ValueError(f"unexpected FRED response (no observation_date header) from {url}")
    return pd.read_csv(io.StringIO(result.stdout), na_values=[".", ""])


def _monthly(url: str, name: str, *, aggregate: str) -> pd.Series:
    df = _fred(url)
    col = [c for c in df.columns if c != "observation_date"][0]
    df = df.rename(columns={"observation_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", col])
    df["ym"] = df["date"].dt.to_period("M")
    grp = df.groupby("ym")[col]
    s = (grp.mean() if aggregate == "mean" else grp.last()).astype(float)
    return s.rename(name)


def _write(series: pd.Series, value_col: str, out_name: str) -> int:
    s = series[series.index <= _dev_boundary()]                    # cap at dev boundary
    out = pd.DataFrame({"year_month": s.index.astype(str), value_col: s.values})
    if out.empty:
        raise ValueError(f"{out_name}: zero rows after dev-boundary cap — check the URL")
    path = OUT_DIR / out_name
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), str(tmp))
    os.replace(tmp, path)
    print(f"  Written: {path} ({len(out):,} rows, {out['year_month'].min()}..{out['year_month'].max()})")
    return len(out)


def main() -> None:
    cfg = _cfg()
    # VIX — daily VIXCLS aggregated to a monthly mean.
    vix = _monthly(cfg["vix_url"], "vix", aggregate="mean")
    _write(vix, "vix", "vix.parquet")
    # term_spread = GS10 - GS3M (monthly constant-maturity yields, percentage points).
    gs10 = _monthly(cfg["term_spread_gs10_url"], "gs10", aggregate="last")
    gs3m = _monthly(cfg["term_spread_gs3m_url"], "gs3m", aggregate="last")
    term = (gs10 - gs3m).dropna().rename("term_spread")
    _write(term, "term_spread", "term_spread.parquet")
    print("\nDone.")


if __name__ == "__main__":
    main()
