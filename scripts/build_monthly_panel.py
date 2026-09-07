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
  size                       float  FISD offering_amt (value-weighting input)
  universe_eligible          bool   FISD universe flag (restriction applied at view)
  rating                     float  monthly as-of credit rating (1=AAA … 22=D)
  investment_grade           bool   rating ≤ IG threshold (nullable)
  maturity                   date   FISD maturity
  time_to_maturity           float  (maturity − date) in years

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
  exit_reason                str    matured | defaulted | defeased (survivorship);
                                    NA when never terminal. Calls undateable in FISD.

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
sys.path.insert(0, str(REPO_ROOT))
from shared.licensed_inputs import require_licensed_input  # noqa: E402
RAW_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_raw.parquet"
CORR_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_corr_filtered.parquet"
RF_FILE = REPO_ROOT / "data" / "development" / "rf_rate.parquet"
OUT_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "monthly_panel_maximal_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# FISD reference artefacts (scripts/build_fisd_reference.py). Characteristics
# (size, rating, maturity, universe, exit_reason) attach here once as shared,
# family-agnostic columns — they are bond facts, not prices, so they are NOT
# family-indexed under A9.
FISD_STATIC = REPO_ROOT / "data" / "development" / "fisd" / "fisd_reference_static.parquet"
FISD_RATINGS = REPO_ROOT / "data" / "development" / "fisd" / "fisd_ratings_monthly.parquet"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["monthly_panel"]


def load_fisd_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("fisd")
    if block is None:
        raise KeyError("thresholds.yaml missing the `fisd:` block (required for the merge)")
    return block


def merge_fisd(panel: pd.DataFrame, fisd_cfg: dict) -> pd.DataFrame:
    """Attach FISD characteristics as shared (family-agnostic) columns.

    Merges fisd_reference_static on `cusip` and fisd_ratings_monthly on
    (cusip, date), then sets:
      - size               = the configured size proxy (offering_amt)
      - universe_eligible  = FISD universe flag (unmatched → False)
      - rating, investment_grade = monthly as-of credit rating
      - maturity, time_to_maturity (years; negative past maturity)
      - exit_reason        = matured | defaulted | defeased, tagged on every
                             month from the EARLIEST exit date onward (so the
                             uncorrected view drops the post-exit rows and the
                             corrected view keeps them). NA when never terminal.
    """
    size_proxy = fisd_cfg["amount_outstanding"]["size_proxy"]

    static = pd.read_parquet(
        require_licensed_input(FISD_STATIC, "FISD static table"),
        columns=["cusip", "universe_eligible", size_proxy,
                 "maturity", "default_date", "defeased_date"],
    )
    static["cusip"] = static["cusip"].astype(str)
    ratings = pd.read_parquet(
        require_licensed_input(FISD_RATINGS, "FISD monthly ratings"),
        columns=["cusip", "date", "rating_numeric", "investment_grade"],
    ).rename(columns={"rating_numeric": "rating"})
    ratings["cusip"] = ratings["cusip"].astype(str)

    panel = panel.copy()
    panel["cusip"] = panel["cusip"].astype(str)
    panel = panel.merge(static, on="cusip", how="left")
    panel = panel.merge(ratings, on=["cusip", "date"], how="left")

    panel["size"] = panel[size_proxy]
    panel["universe_eligible"] = panel["universe_eligible"].fillna(False).astype(bool)
    panel["time_to_maturity"] = (panel["maturity"] - panel["date"]).dt.days / 365.25

    exit_date = pd.Series(pd.NaT, index=panel.index, dtype="datetime64[ns]")
    exit_type = pd.Series(pd.NA, index=panel.index, dtype="object")
    for label, col in (("defaulted", "default_date"),
                       ("defeased", "defeased_date"),
                       ("matured", "maturity")):
        d = panel[col]
        better = d.notna() & (exit_date.isna() | (d < exit_date))
        exit_date = exit_date.mask(better, d)
        exit_type = exit_type.mask(better, label)
    terminal = panel["date"] >= exit_date          # NaT comparison → False
    panel["exit_reason"] = exit_type.where(terminal, other=pd.NA)
    return panel


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
    # In-line per-family adjacency: returns are computed per family because the
    # raw and corr price series have divergent NaN patterns (A1.5).
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
    # validate="m:1": rf must be unique per year_month. A duplicated rf month
    # would silently fan out the panel (m:m → multiplied cusip-month rows);
    # this raises MergeError instead.
    panel = panel.merge(rf, on="year_month", how="left", validate="m:1")
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
    # Attach FISD characteristics as shared, family-agnostic columns in one
    # merge: size (= offering_amt), universe_eligible, rating, investment_grade,
    # maturity, time_to_maturity, and the survivorship exit_reason. Both
    # endpoint views inherit these identically (views.py _SHARED_PANEL_COLUMNS).
    fisd_cfg = load_fisd_config()
    panel = merge_fisd(panel, fisd_cfg)

    out_cols = [
        "cusip", "date", "size",
        "universe_eligible", "rating", "investment_grade",
        "maturity", "time_to_maturity",
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
        b"size_policy": b"fisd_offering_amt",
        b"survivorship_policy": b"exit_reason_from_FISD_maturity_default_defeased;calls_undateable",
        b"universe_policy": b"universe_eligible_flag;restriction_applied_at_view",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")

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
        "size_column_policy": "fisd offering_amt (value-weighting live)",
        "fisd_coverage": {
            "size_non_null": int(panel["size"].notna().sum()),
            "universe_eligible_rows": int(panel["universe_eligible"].sum()),
            "rating_non_null": int(panel["rating"].notna().sum()),
            "exit_reason_tagged_rows": int(panel["exit_reason"].notna().sum()),
        },
        "survivorship": _survivorship_report(panel, fisd_cfg),
    }
    return counts


def _survivorship_report(panel: pd.DataFrame, fisd_cfg: dict) -> dict:
    """Exit-reason breakdown (eligible) + the default-return picture.

    Makes the kept-distress-row return assumption explicit: of the distress
    (default) bond-months the corrected view keeps, how many carry a real
    (non-NaN) corrected-family return vs NaN (no trade → no crater). A bond that
    stops trading at default contributes NaN, so the survivorship gap is a LOWER
    bound under the TRACE-derived (no-recovery-overlay) convention.
    """
    elig = panel[panel["universe_eligible"]]
    distress = set(fisd_cfg.get("survivorship", {}).get("distress_exits", ["defaulted"]))
    by_reason = elig["exit_reason"].value_counts(dropna=True).to_dict()
    d = elig[elig["exit_reason"].isin(distress)]
    return {
        "eligible_terminal_by_reason": {str(k): int(v) for k, v in by_reason.items()},
        "distress_exits": sorted(distress),
        "distress_rows": int(len(d)),
        "distress_rows_with_corr_return": int(d["ret_corr"].notna().sum()),
        "distress_rows_nan_corr_return": int(d["ret_corr"].isna().sum()),
        "return_convention": "trace_derived_last_price; no_recovery_overlay (gap is a lower bound)",
    }


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

    print("\nDone.")
    print(f"  {counts['cusip_month_observations']:,} cusip-month observations")
    print(f"  {counts['unique_cusips']:,} unique cusips")
    print(f"  {counts['date_range_start']} – {counts['date_range_end']}")
    print(f"  ret_raw non-null:  {counts['non_null_ret_raw']:,}")
    print(f"  ret_corr non-null: {counts['non_null_ret_corr']:,}")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
