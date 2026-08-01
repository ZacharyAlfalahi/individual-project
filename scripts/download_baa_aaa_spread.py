"""
Download the Moody's BAA-AAA credit spread from FRED → data/development/baa_aaa_spread.parquet.

Series (both monthly, annualised %):
  BAA — Moody's Seasoned Baa Corporate Bond Yield
  AAA — Moody's Seasoned Aaa Corporate Bond Yield
Spread = BAA - AAA (percentage points). This is the extension_1 credit-cycle regime variable
(docs/extension_1_config.yaml); its DEVELOPMENT-window median is the pre-registered split.

Mirrors scripts/download_rf_rate.py (public FRED CSV via curl, no API key). URLs live in
docs/thresholds.yaml (monthly_panel.baa_yield_url / .aaa_yield_url) — no hard-coded endpoint.

Output columns:
  year_month         — str, e.g. "2015-06"
  baa_aaa_spread_pp  — float, BAA yield - AAA yield in percentage points

Usage:
  python scripts/download_baa_aaa_spread.py
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
OUT_FILE = REPO_ROOT / "data" / "development" / "baa_aaa_spread.parquet"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        return yaml.safe_load(f)["monthly_panel"]


def download_fred_series(url: str) -> pd.DataFrame:
    print(f"Downloading: {url}")
    # -f: fail (nonzero exit) on HTTP 4xx/5xx so an error page never flows into read_csv;
    # -S: still show the error under -s; --retry 3: tolerate transient network blips.
    result = subprocess.run(
        ["curl", "-fsS", "--retry", "3", "--max-time", "60", url],
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.lstrip().startswith("observation_date"):
        raise ValueError(f"unexpected FRED response (no observation_date header) from {url}")
    return pd.read_csv(io.StringIO(result.stdout), na_values=[".", ""])


def _monthly_series(url: str, name: str) -> pd.Series:
    df = download_fred_series(url)
    series_col = [c for c in df.columns if c != "observation_date"][0]
    df = df.rename(columns={"observation_date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", series_col])
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    return df.set_index("year_month")[series_col].rename(name).astype(float)


def _dev_boundary_period(root: Path) -> pd.Period:
    """Last development month = December of (holdout_start_year - 1). The development artefact
    must NOT contain holdout-era (>= holdout_start_year) macro values (window discipline); the
    holdout BAA-AAA slice is built separately inside the one-shot holdout frozen inventory."""
    with open(THRESHOLDS_FILE) as f:
        hsy = int(yaml.safe_load(f)["trace_cleaning"]["holdout_start_year"])
    return pd.Period(f"{hsy - 1}-12", freq="M")


def build_spread_parquet(cfg: dict) -> int:
    baa = _monthly_series(cfg["baa_yield_url"], "baa")
    aaa = _monthly_series(cfg["aaa_yield_url"], "aaa")
    joined = pd.concat([baa, aaa], axis=1).dropna()
    if joined.empty:
        raise ValueError("BAA/AAA join produced zero overlapping months — check the URLs")
    joined["baa_aaa_spread_pp"] = joined["baa"] - joined["aaa"]

    out = joined.reset_index()[["year_month", "baa_aaa_spread_pp"]]
    # Cap at the development boundary — never let holdout-era rows into a dev artefact.
    dev_end = _dev_boundary_period(REPO_ROOT)
    out = out[pd.PeriodIndex(out["year_month"], freq="M") <= dev_end].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"BAA-AAA: zero rows at/before the dev boundary {dev_end} — check the pull")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE} ({len(out):,} rows)")
    print(f"  Date range: {out['year_month'].min()} to {out['year_month'].max()}")
    return len(out)


def main() -> None:
    cfg = load_config()
    n = build_spread_parquet(cfg)
    print(f"\nDone. {n:,} monthly observations → {OUT_FILE}")


if __name__ == "__main__":
    main()
