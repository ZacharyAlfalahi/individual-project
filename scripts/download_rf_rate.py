"""
Download risk-free rate from FRED and save to data/development/rf_rate.parquet.

Series: TB3MS — 3-Month Treasury Bill: Secondary Market Rate (monthly, annualised %).
Source: Federal Reserve H.15 release via FRED public CSV endpoint (no API key required).

Output columns:
  year_month  — pandas Period[M] (e.g. 2015-06)
  rf_monthly  — monthly return in decimal (annualised % / 12 / 100)

Usage:
  python scripts/download_rf_rate.py
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
OUT_FILE = REPO_ROOT / "data" / "development" / "rf_rate.parquet"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["monthly_panel"]


def download_fred_series(url: str) -> pd.DataFrame:
    print(f"Downloading: {url}")
    result = subprocess.run(
        # -f fails on HTTP 4xx/5xx (no error page into read_csv); -S shows it under -s; --retry 3.
        ["curl", "-fsS", "--retry", "3", "--max-time", "60", url],
        capture_output=True,
        text=True,
        check=True,
    )
    if not result.stdout.lstrip().startswith("observation_date"):
        raise ValueError(f"unexpected FRED response (no observation_date header) from {url}")
    return pd.read_csv(io.StringIO(result.stdout), na_values=[".", ""])


def build_rf_parquet(cfg: dict) -> int:
    url = cfg["rf_rate_url"]
    df = download_fred_series(url)

    # FRED CSV columns: observation_date, <series_id>
    series_col = [c for c in df.columns if c != "observation_date"][0]
    df = df.rename(columns={"observation_date": "date", series_col: "rate_pct_annual"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "rate_pct_annual"])

    # Convert annualised % to monthly decimal: (rate / 100) / 12
    df["rf_monthly"] = df["rate_pct_annual"] / 100.0 / 12.0
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

    out = df[["year_month", "rf_monthly"]].reset_index(drop=True)

    if len(out) == 0:
        raise ValueError("FRED download returned zero rows — check URL or response content")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE} ({len(out):,} rows)")
    print(f"  Date range: {out['year_month'].min()} to {out['year_month'].max()}")
    return len(out)


def main():
    cfg = load_config()
    n = build_rf_parquet(cfg)
    print(f"\nDone. {n:,} monthly observations → {OUT_FILE}")


if __name__ == "__main__":
    main()
