"""
Build the maximal CUSIP-keyed monthly bond panel — dual column families.

Reads the two daily-layer parquets emitted by the registry pipeline:

  data/development/trace_daily_raw.parquet                 (raw family)
  data/development/trace_daily_corr_filtered.parquet       (corrected family, post-distressed)

and emits the maximal monthly panel:

  data/development/monthly_panel_maximal.parquet

Output columns (keyed on cusip_id × year_month):

  cusip                      str    primary key
  date                       date   month-end timestamp
  size                       float  placeholder constant (FISD-gated)

  price_eom_raw              float  Σ(daily_vwap × daily_vol) / Σ(daily_vol), raw
  price_eom_corr             float  same, corr family
  ret_raw                    float  (P_t − P_{t−1}) / P_{t−1}, raw, adjacency-checked
  ret_corr                   float  same, corr family
  xret_raw                   float  ret_raw − rf_monthly
  xret_corr                  float  ret_corr − rf_monthly
  n_trades_raw, n_trades_corr int   summed daily n_trades
  total_vol_raw, total_vol_corr float summed daily total_vol
  last_trade_date_raw        date   max(trd_exctn_dt) on raw daily
  last_trade_date_corr       date   max(trd_exctn_dt) on corr-FILTERED daily
                                    (per A5: post-distressed-filter, not pre)

  rf_monthly                 float  risk-free rate (TB3MS / 12 / 100)
  exit_reason                str    NaN-everywhere placeholder (FISD-gated)

Two methodology principles encoded:
  (1) Adjacency rule applied PER FAMILY. raw and corr have potentially
      divergent NaN patterns — bounce-back and distressed filters drop only
      from corr — so a missing month in one family doesn't propagate to
      the other. This is the spec's "do not 'repair' divergent NaN
      patterns" rule.
  (2) last_trade_date_corr computed from the POST-distressed-filter daily
      artifact, per A5. Using pre-filter daily would manufacture phantom
      freshness and corrupt Phase 2's stale_price mask.

The correction toggles are view-time operations in Phase 2's view layer, not
here. This script unconditionally emits both families; the toggles
select between them at engine-feed time.

Usage:
  python scripts/build_monthly_panel.py
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
RAW_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_raw.parquet"
CORR_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_corr_filtered.parquet"
RF_FILE = REPO_ROOT / "data" / "development" / "rf_rate.parquet"
OUT_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "monthly_panel_maximal_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# size column is a placeholder until FISD's amount_outstanding lands. Same
# value for both families — size is a bond characteristic, not a price, so
# it's not family-indexed under A9.
SIZE_PLACEHOLDER = 1.0


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["monthly_panel"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def aggregate_family(daily_path: Path, family: str) -> pd.DataFrame:
    """Aggregate one family's daily layer → monthly panel with last_trade_date.

    Output columns:
      cusip_id, year_month,
      price_eom_<family>, total_vol_<family>, n_trades_<family>,
      last_trade_date_<family>
    """
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily input not found: {daily_path}")

    print(f"  Aggregating {family} daily layer ({daily_path.name}) ...")
    lazy = (
        pl.scan_parquet(str(daily_path))
        .with_columns(
            pl.col("trd_exctn_dt").dt.strftime("%Y-%m").alias("year_month"),
            (pl.col("price_vwap") * pl.col("total_vol")).alias("price_x_vol"),
        )
        .group_by(["cusip_id", "year_month"])
        .agg(
            pl.col("price_x_vol").sum().alias("price_x_vol_sum"),
            pl.col("total_vol").sum().alias(f"total_vol_{family}"),
            pl.col("n_trades").sum().alias(f"n_trades_{family}"),
            pl.col("trd_exctn_dt").max().alias(f"last_trade_date_{family}"),
        )
        .with_columns(
            (pl.col("price_x_vol_sum") / pl.col(f"total_vol_{family}"))
                .alias(f"price_eom_{family}"),
        )
        .drop("price_x_vol_sum")
    )
    df = lazy.collect().to_pandas()
    print(f"  {family}: {len(df):,} cusip-month rows, "
          f"{df['cusip_id'].nunique():,} cusips")
    return df


def compute_returns_inplace(panel: pd.DataFrame, family: str) -> None:
    """Compute ret_<family> with the adjacency rule.

    Applied AFTER the outer-join across families, so the function sees the
    full panel of (cusip, year_month) rows (including rows that exist for
    one family but not the other). Returns are computed PER FAMILY: each
    family has its own price series with its own NaN pattern. The adjacency
    rule is enforced within each family's own series.
    """
    price_col = f"price_eom_{family}"
    panel.sort_values(["cusip_id", "year_month"], inplace=True)

    panel[f"price_lag_{family}"] = panel.groupby("cusip_id")[price_col].shift(1)
    panel[f"ret_{family}"] = (
        (panel[price_col] - panel[f"price_lag_{family}"])
        / panel[f"price_lag_{family}"]
    )
    # Adjacency: ret = NaN unless prior month is the immediately adjacent
    # calendar month. Without this, a year-long gap would silently produce
    # a one-month-style return.
    ym_period = pd.PeriodIndex(panel["year_month"], freq="M")
    prev_ym = panel.groupby("cusip_id")[f"price_lag_{family}"].transform(
        lambda s: ym_period.to_series().reset_index(drop=True).reindex(s.index).shift(1).iloc[:len(s)]
    )
    # Simpler: shift year_month itself per group
    panel["_ym_pd"] = ym_period
    panel[f"_prev_ym_{family}"] = panel.groupby("cusip_id")["_ym_pd"].shift(1)
    gap = (panel["_ym_pd"] - panel[f"_prev_ym_{family}"]).map(
        lambda x: x.n if pd.notna(x) else float("nan")
    )
    panel.loc[(gap != 1) | gap.isna(), f"ret_{family}"] = float("nan")
    panel.drop(columns=[f"price_lag_{family}", f"_prev_ym_{family}"], inplace=True)


def build_panel(cfg: dict) -> dict:
    if not RF_FILE.exists():
        raise FileNotFoundError(f"RF rate file not found: {RF_FILE}")

    raw_df = aggregate_family(RAW_DAILY, "raw")
    corr_df = aggregate_family(CORR_DAILY, "corr")

    print("Outer-joining families on (cusip_id, year_month) ...")
    panel = pd.merge(
        raw_df, corr_df, on=["cusip_id", "year_month"], how="outer"
    ).sort_values(["cusip_id", "year_month"]).reset_index(drop=True)
    print(f"  Joined: {len(panel):,} cusip-month rows, "
          f"{panel['cusip_id'].nunique():,} unique cusips")

    print("Computing per-family returns with adjacency rule ...")
    # Use a simpler in-line adjacency implementation; the helper above tried
    # to be too clever. Adjacency is per-family because NaN patterns diverge.
    panel.sort_values(["cusip_id", "year_month"], inplace=True)
    panel["_ym_pd"] = pd.PeriodIndex(panel["year_month"], freq="M")
    for family in ("raw", "corr"):
        price_col = f"price_eom_{family}"
        # Per-cusip lag; only the rows with non-NaN price in this family will
        # produce a non-NaN lag chain (a NaN price in this family results in
        # downstream NaN ret regardless of the other family).
        panel[f"_lag_{family}"] = panel.groupby("cusip_id")[price_col].shift(1)
        panel[f"_prev_ym_{family}"] = panel.groupby("cusip_id")["_ym_pd"].shift(1)
        panel[f"ret_{family}"] = (
            (panel[price_col] - panel[f"_lag_{family}"])
            / panel[f"_lag_{family}"]
        )
        gap = (panel["_ym_pd"] - panel[f"_prev_ym_{family}"]).map(
            lambda x: x.n if pd.notna(x) else float("nan")
        )
        panel.loc[(gap != 1) | gap.isna(), f"ret_{family}"] = float("nan")
        panel.drop(columns=[f"_lag_{family}", f"_prev_ym_{family}"], inplace=True)
    panel.drop(columns=["_ym_pd"], inplace=True)

    print("Merging risk-free rate ...")
    rf = pd.read_parquet(RF_FILE)
    panel = panel.merge(rf, on="year_month", how="left")
    missing_rf = panel["rf_monthly"].isna().sum()
    if missing_rf:
        print(f"  WARNING: {missing_rf:,} cusip-months have no matching rf rate")

    panel["xret_raw"] = panel["ret_raw"] - panel["rf_monthly"]
    panel["xret_corr"] = panel["ret_corr"] - panel["rf_monthly"]

    # Engine-contract auxiliary columns
    panel["cusip"] = panel["cusip_id"]
    panel["date"] = (
        pd.PeriodIndex(panel["year_month"], freq="M")
            .to_timestamp(how="end").normalize()
        + pd.offsets.MonthEnd(0)
    )
    panel["size"] = SIZE_PLACEHOLDER

    # FISD-gated columns: emitted now as NaN-everywhere placeholders so the
    # view layer has the column to operate on (it'll stay NaN until FISD
    # lands). Same A1 / A6 stance the registry takes.
    panel["exit_reason"] = pd.NA

    out_cols = [
        "cusip", "date", "size",
        "price_eom_raw", "price_eom_corr",
        "ret_raw", "ret_corr",
        "xret_raw", "xret_corr",
        "n_trades_raw", "n_trades_corr",
        "total_vol_raw", "total_vol_corr",
        "last_trade_date_raw", "last_trade_date_corr",
        "rf_monthly", "exit_reason",
    ]
    panel = panel[out_cols].reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    # Stash registry-aware metadata so downstream readers / Phase 2 views
    # can detect the panel kind without having to open every column.
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"panel_kind":  b"maximal",
        b"primary_key": b"cusip",
        b"families":    b"raw,corr",
        b"size_policy": f"placeholder_const_{SIZE_PLACEHOLDER}".encode("utf-8"),
        b"survivorship_policy": b"exit_reason_nan_pre_FISD",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")
    print(
        f"  WARN: size column is a placeholder constant ({SIZE_PLACEHOLDER}); "
        f"pass weighting='equal' in the rulebook until FISD's "
        f"amount_outstanding lands.",
        file=sys.stderr,
    )

    counts = {
        "cusip_month_observations": int(len(panel)),
        "unique_cusips": int(panel["cusip"].nunique()),
        "date_range_start": str(panel["date"].min().date()) if len(panel) else None,
        "date_range_end": str(panel["date"].max().date()) if len(panel) else None,
        "non_null_ret_raw": int(panel["ret_raw"].notna().sum()),
        "non_null_ret_corr": int(panel["ret_corr"].notna().sum()),
        "non_null_xret_raw": int(panel["xret_raw"].notna().sum()),
        "non_null_xret_corr": int(panel["xret_corr"].notna().sum()),
        "non_null_price_eom_raw": int(panel["price_eom_raw"].notna().sum()),
        "non_null_price_eom_corr": int(panel["price_eom_corr"].notna().sum()),
        "missing_rf_months": int(missing_rf),
        "size_column_policy": (
            f"placeholder constant {SIZE_PLACEHOLDER} (FISD-gated; use "
            f"weighting='equal' in rulebook)"
        ),
    }
    return counts


def write_report(counts: dict, cfg: dict) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": cfg,
        "counts": counts,
        "panel_kind": "maximal",
        "primary_key": "cusip",
        "families": ["raw", "corr"],
        "stage": "maximal_monthly_panel",
        "registry_role": (
            "Maximal monthly panel — dual column families (raw + corr) with "
            "shared metadata columns. Phase 2's view layer materialises "
            "uncorrected and corrected ENDPOINT views by selecting columns "
            "from this panel; this script never materialises those endpoints."
        ),
        "return_methodology": (
            "Daily-VWAP weighted Σ → monthly VWAP per family; "
            "ret = (P_t − P_{t-1}) / P_{t-1} with adjacency rule (NaN unless "
            "prior month is immediately adjacent) applied PER FAMILY; "
            "xret = ret − rf_monthly (TB3MS / 12 / 100)."
        ),
        "note": (
            "Per A1.5 raw and corr have divergent NaN patterns — bounce-back "
            "and distressed filters drop only from corr. Do not 'repair' "
            "this. last_trade_date_corr comes from the POST-distressed-filter "
            "daily artifact per A5; using pre-filter would corrupt the future "
            "stale_price mask."
        ),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    for required in [RAW_DAILY, CORR_DAILY, RF_FILE]:
        if not required.exists():
            print(f"ERROR: Required input not found: {required}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config()
    print(f"Config: rf_rate_series={cfg['rf_rate_series']}")
    counts = build_panel(cfg)
    write_report(counts, cfg)

    print(f"\nDone.")
    print(f"  {counts['cusip_month_observations']:,} cusip-month observations")
    print(f"  {counts['unique_cusips']:,} unique cusips")
    print(f"  {counts['date_range_start']} – {counts['date_range_end']}")
    print(f"  ret_raw non-null:  {counts['non_null_ret_raw']:,}")
    print(f"  ret_corr non-null: {counts['non_null_ret_corr']:,}")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
