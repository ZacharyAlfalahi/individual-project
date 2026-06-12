"""
Daily aggregation layer — emits a (cusip, date)-keyed daily panel for ONE
column family at a time (raw or corrected). Retains per-day min and max
trade prices alongside VWAP — these are non-deferrable per A7.4 of the
registry amendments (Filter 4's input is intraday range = (max-min)/mean).

Stage order (normative per A7.2 of the registry amendments):
  decimal-shift → bounce-back → VWAP→daily → distressed filters 1-4
This script is "VWAP→daily" for whichever family.

Trade-level filters applied here mirror the existing monthly_panel build so
the distressed filters operate on the same trade pool that ends up in the
maximal monthly panel:
  - sub_prdct ∈ {"CORP", ""} (CORP plus pre-2012 NULL records)
  - entrd_vol_qt ≥ min_vol_qt (BBW 2019 institutional-size filter)
  - cusip_id not null/blank (CUSIP-keying)

Output columns:
  cusip_id (str), trd_exctn_dt (date), price_vwap (float),
  total_vol (float), n_trades (int), min_price (float), max_price (float)

Inputs and outputs are family-determined by the --family CLI flag:
  --family raw  → reads trace_clean_raw.parquet  → trace_daily_raw.parquet
  --family corr → reads trace_clean_corr.parquet → trace_daily_corr.parquet

Both families run through this same script (the boolean-branch principle):
the only difference between the two daily outputs is the input price data
upstream — never any logic divergence in this script.

Usage:
  python scripts/build_daily_panel.py --family raw
  python scripts/build_daily_panel.py --family corr
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

FAMILY_PATHS = {
    "raw": {
        "input":  REPO_ROOT / "data" / "development" / "trace_clean_raw.parquet",
        "output": REPO_ROOT / "data" / "development" / "trace_daily_raw.parquet",
        "holdout_input":  REPO_ROOT / "data" / "holdout" / "trace_clean_raw.parquet",
        "holdout_output": REPO_ROOT / "data" / "holdout" / "trace_daily_raw.parquet",
        "report": REPO_ROOT / "data" / "development" / "daily_panel_raw_report.json",
    },
    "corr": {
        "input":  REPO_ROOT / "data" / "development" / "trace_clean_corr.parquet",
        "output": REPO_ROOT / "data" / "development" / "trace_daily_corr.parquet",
        "holdout_input":  REPO_ROOT / "data" / "holdout" / "trace_clean_corr.parquet",
        "holdout_output": REPO_ROOT / "data" / "holdout" / "trace_daily_corr.parquet",
        "report": REPO_ROOT / "data" / "development" / "daily_panel_corr_report.json",
    },
}

OUTPUT_SCHEMA = pa.schema([
    pa.field("cusip_id",      pa.string()),
    pa.field("trd_exctn_dt",  pa.timestamp("us")),
    pa.field("price_vwap",    pa.float64()),
    pa.field("total_vol",     pa.float64()),
    pa.field("n_trades",      pa.int64()),
    pa.field("min_price",     pa.float64()),
    pa.field("max_price",     pa.float64()),
])


def load_config() -> dict:
    """Reuses monthly_panel filters (sub_prdct, min_vol_qt) at the daily stage
    so the distressed filters and downstream monthly build see the same
    trade pool."""
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["monthly_panel"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def aggregate_daily(input_path: Path, output_path: Path, cfg: dict) -> dict:
    """Aggregate per-trade rows to (cusip_id, trd_exctn_dt). Returns counts."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    sub_prdct_keep = list(cfg["sub_prdct_keep"])
    min_vol = float(cfg["min_vol_qt"])

    sub_filter = (
        pl.col("sub_prdct").is_null()
        | pl.col("sub_prdct").is_in(sub_prdct_keep)
    )

    print(f"  Aggregating {input_path.name} → {output_path.name} ...")
    lazy = (
        pl.scan_parquet(str(input_path))
        # Trade-level filters — match the existing monthly_panel pre-aggregation
        .filter(pl.col("cusip_id").is_not_null() & (pl.col("cusip_id") != ""))
        .filter(pl.col("entrd_vol_qt") >= min_vol)
        .filter(sub_filter)
        .with_columns(
            (pl.col("rptd_pr") * pl.col("entrd_vol_qt")).alias("price_x_vol"),
        )
        .group_by(["cusip_id", "trd_exctn_dt"])
        .agg(
            pl.col("price_x_vol").sum().alias("price_x_vol_sum"),
            pl.col("entrd_vol_qt").sum().alias("total_vol"),
            pl.col("rptd_pr").count().alias("n_trades"),
            pl.col("rptd_pr").min().alias("min_price"),
            pl.col("rptd_pr").max().alias("max_price"),
        )
        .with_columns(
            (pl.col("price_x_vol_sum") / pl.col("total_vol")).alias("price_vwap"),
        )
        .drop("price_x_vol_sum")
        .select([
            "cusip_id", "trd_exctn_dt", "price_vwap", "total_vol",
            "n_trades", "min_price", "max_price",
        ])
    )

    print("  Collecting aggregation...")
    df = lazy.collect()
    n_rows = df.height
    n_cusips = df["cusip_id"].n_unique()
    print(f"  {n_rows:,} cusip-day rows; {n_cusips:,} unique cusips")

    table = df.to_arrow().cast(OUTPUT_SCHEMA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".parquet.tmp")
    if tmp.exists():
        tmp.unlink()
    pq.write_table(table, str(tmp))

    # Verify row count
    actual = pq.read_metadata(str(tmp)).num_rows
    if actual != n_rows:
        tmp.unlink()
        raise AssertionError(f"Daily parquet row count {actual:,} != {n_rows:,}")
    os.replace(tmp, output_path)

    return {
        "rows": n_rows,
        "unique_cusips": n_cusips,
    }


def write_report(family: str, dev_counts: dict, hold_counts: "dict | None",
                 cfg: dict, paths: dict) -> None:
    """hold_counts is None during development — no holdout statistic is
    computed or surfaced into this (development-side) report."""
    report_path = paths["report"]
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": {
            "sub_prdct_keep": cfg["sub_prdct_keep"],
            "min_vol_qt": cfg["min_vol_qt"],
        },
        "stage": f"daily_aggregation_{family}",
        "registry_role": (
            f"Daily layer for the {family} column family. "
            "Per A7.4 the daily cache retains per-day min and max prices "
            "alongside VWAP — Filter 4 (intraday inconsistency) requires the "
            "intraday range. Bounce-back has already been applied for the "
            "corr family; the raw family is post-Dick-Nielsen only."
        ),
        "holdout_processed": hold_counts is not None,
        "rows_dev": dev_counts,
        "inputs": {
            "dev": str(paths["input"].relative_to(REPO_ROOT)),
        },
        "outputs": {
            "dev": str(paths["output"].relative_to(REPO_ROOT)),
        },
    }
    if hold_counts is not None:
        report["rows_holdout"] = hold_counts
        report["inputs"]["holdout"] = str(paths["holdout_input"].relative_to(REPO_ROOT))
        report["outputs"]["holdout"] = str(paths["holdout_output"].relative_to(REPO_ROOT))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_path)
    print(f"  Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--family", choices=["raw", "corr"], required=True,
        help="Column family to build daily layer for.",
    )
    parser.add_argument(
        "--holdout", action="store_true",
        help="Also process the holdout partition. NEVER pass during "
             "development; reserved for the single post-freeze pipeline "
             "run in weeks 13-14 (inviolable data rule).",
    )
    args = parser.parse_args()

    paths = FAMILY_PATHS[args.family]
    if not paths["input"].exists():
        print(f"ERROR: Required input not found: {paths['input']}", file=sys.stderr)
        if args.family == "raw":
            print("Run scripts/preprocess_trace.py first.", file=sys.stderr)
        else:
            print("Run preprocess_trace.py → apply_decimal_shift.py → bounce_back_filter.py first.",
                  file=sys.stderr)
        sys.exit(1)

    cfg = load_config()
    print(f"Family: {args.family}")
    print(f"  sub_prdct_keep={cfg['sub_prdct_keep']}, min_vol_qt={cfg['min_vol_qt']:,}")

    print("Processing development partition...")
    dev_counts = aggregate_daily(paths["input"], paths["output"], cfg)

    hold_counts = None
    if args.holdout:
        print("Processing holdout partition (post-freeze run)...")
        hold_counts = aggregate_daily(paths["holdout_input"], paths["holdout_output"], cfg)
    else:
        print("Holdout partition NOT processed (development mode; "
              "pass --holdout for the post-freeze run).")

    write_report(args.family, dev_counts, hold_counts, cfg, paths)

    print("\nDone.")
    print(f"  Dev: {dev_counts['rows']:,} rows → {paths['output']}")
    if hold_counts is not None:
        print(f"  Hold: {hold_counts['rows']:,} rows → {paths['holdout_output']}")


if __name__ == "__main__":
    main()
