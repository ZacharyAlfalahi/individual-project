"""
Build monthly corporate bond return panel from cleaned TRACE transaction data.

Pipeline:
  1. Load data/development/trace_clean.parquet via Polars lazy scan (276M rows — avoids OOM)
  2. Filter sub_prdct (CORP + blank/NULL for pre-2012 coverage; drop CHRC, ELN)
  3. Filter institutional trades (entrd_vol_qt >= min_vol_qt)
  4. Aggregate to VWAP price per bond-month
  5. Compute holding-period return: ret = (P_t - P_{t-1}) / P_{t-1}
  6. Merge risk-free rate from data/development/rf_rate.parquet
  7. Compute excess return: xret = ret - rf_monthly
  8. Write data/development/monthly_panel.parquet + monthly_panel_report.json

All thresholds come from docs/thresholds.yaml (nothing hard-coded here).

Return methodology note:
  Uses rptd_pr (clean/dirty price) without accrued interest adjustment. This is
  the standard approximation in TRACE-based papers (Bai, Bali, Wen 2019). Duration-
  matched excess returns require maturity data not available in the current schema
  and are left as an open decision pending Librarian output on target papers.

Usage:
  python scripts/build_monthly_panel.py

Requires: data/development/trace_clean.parquet  (preprocess_trace.py)
          data/development/rf_rate.parquet        (download_rf_rate.py)
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACE_FILE = REPO_ROOT / "data" / "development" / "trace_clean.parquet"
RF_FILE = REPO_ROOT / "data" / "development" / "rf_rate.parquet"
OUT_FILE = REPO_ROOT / "data" / "development" / "monthly_panel.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "monthly_panel_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["monthly_panel"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def build_panel(cfg: dict) -> dict:
    sub_prdct_keep = set(cfg["sub_prdct_keep"])   # {"CORP", ""}
    min_vol = cfg["min_vol_qt"]

    print("Scanning trace_clean.parquet with Polars (lazy)...")

    sub_filter = (
        pl.col("sub_prdct").is_null() |
        pl.col("sub_prdct").is_in(list(sub_prdct_keep))
    )

    lazy = (
        pl.scan_parquet(str(TRACE_FILE))
        .filter(pl.col("bond_id").is_not_null())
        .filter(pl.col("entrd_vol_qt") >= min_vol)
        .filter(sub_filter)
        .with_columns(
            pl.col("trd_exctn_dt").dt.strftime("%Y-%m").alias("year_month"),
            (pl.col("rptd_pr") * pl.col("entrd_vol_qt")).alias("price_x_vol"),
        )
        .group_by(["bond_id", "year_month"])
        .agg(
            pl.col("price_x_vol").sum().alias("price_x_vol_sum"),
            pl.col("entrd_vol_qt").sum().alias("total_vol"),
            pl.col("rptd_pr").count().alias("n_trades"),
            pl.col("sub_prdct").first().alias("sub_prdct"),
        )
        .with_columns(
            (pl.col("price_x_vol_sum") / pl.col("total_vol")).alias("price_eom"),
        )
        .drop("price_x_vol_sum")
    )

    print("  Collecting aggregation result...")
    agg_pl = lazy.collect()
    agg = agg_pl.to_pandas()
    bond_months = len(agg)
    print(f"  {bond_months:,} bond-month observations")

    if bond_months == 0:
        agg["ret"] = pd.Series(dtype=float)
        agg["xret"] = pd.Series(dtype=float)
        agg["rf_monthly"] = pd.Series(dtype=float)
        ret_non_null = 0
        xret_non_null = 0
        missing_rf = 0
    else:
        agg = agg.sort_values(["bond_id", "year_month"])

        print("Computing holding-period returns...")
        agg["price_lag"] = agg.groupby("bond_id")["price_eom"].shift(1)
        agg["ret"] = (agg["price_eom"] - agg["price_lag"]) / agg["price_lag"]
        # Null out returns where lag is from a non-consecutive prior month
        agg["year_month_pd"] = pd.PeriodIndex(agg["year_month"], freq="M")
        agg["prev_ym"] = agg.groupby("bond_id")["year_month_pd"].shift(1)
        gap = (agg["year_month_pd"] - agg["prev_ym"]).map(
            lambda x: x.n if pd.notna(x) else float("nan")
        )
        agg.loc[(gap != 1) | gap.isna(), "ret"] = float("nan")
        agg = agg.drop(columns=["price_lag", "year_month_pd", "prev_ym"])

        ret_non_null = agg["ret"].notna().sum()
        print(f"  {ret_non_null:,} non-null returns ({ret_non_null/bond_months:.1%} of bond-months)")

        print("Merging risk-free rate...")
        rf = pd.read_parquet(RF_FILE)
        agg = agg.merge(rf, on="year_month", how="left")
        missing_rf = agg["rf_monthly"].isna().sum()
        if missing_rf > 0:
            print(f"  WARNING: {missing_rf:,} bond-months have no matching rf rate")

        agg["xret"] = agg["ret"] - agg["rf_monthly"]
        xret_non_null = agg["xret"].notna().sum()

    out_cols = ["bond_id", "year_month", "price_eom", "ret", "xret",
                "n_trades", "total_vol", "sub_prdct", "rf_monthly"]
    panel = agg[out_cols].reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")

    counts = {
        "bond_month_observations": bond_months,
        "ret_non_null": int(ret_non_null),
        "xret_non_null": int(xret_non_null),
        "unique_bonds": int(panel["bond_id"].nunique()),
        "date_range_start": panel["year_month"].min(),
        "date_range_end": panel["year_month"].max(),
        "missing_rf_months": int(missing_rf),
    }
    return counts


def write_report(counts: dict, cfg: dict) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "counts": counts,
        "return_methodology": (
            "VWAP price per bond-month; ret = (P_t - P_{t-1}) / P_{t-1}; "
            "xret = ret - rf_monthly (TB3MS / 12 / 100); "
            "no accrued interest adjustment (standard TRACE approximation, BBW 2019)"
        ),
        "note": (
            "Per-filter transaction row counts not reported here (Polars lazy path); "
            "see data/development/cleaning_report.json for TRACE-level counts."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    for required in [TRACE_FILE, RF_FILE]:
        if not required.exists():
            print(f"ERROR: Required file not found: {required}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config()
    print(f"Config: sub_prdct_keep={cfg['sub_prdct_keep']}, "
          f"min_vol_qt={cfg['min_vol_qt']:,}, method={cfg['price_agg_method']}")

    counts = build_panel(cfg)
    write_report(counts, cfg)

    print(f"\nDone.")
    print(f"  {counts['bond_month_observations']:,} bond-month observations")
    print(f"  {counts['unique_bonds']:,} unique bonds")
    print(f"  {counts['date_range_start']} – {counts['date_range_end']}")
    print(f"  {counts['xret_non_null']:,} excess return observations")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
