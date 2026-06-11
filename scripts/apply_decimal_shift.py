"""
WRDS-MMN decimal-shift correction stage (corrected branch only).

Reads `trace_clean_raw.parquet` (the post-Dick-Nielsen output of
preprocess_trace.py — see registry spec Section 2 for the basic-cleaning
boundary) and emits `trace_clean_decimal_shifted.parquet`, applying:

  1. **Price plausibility filter** — drop rows with `rptd_pr <= price_floor`
     or `rptd_pr > pre_correction_ceiling`. Relocated from preprocess_trace.py
     per A1.7 of the amendments: under `meas_err = OFF` the raw family must
     preserve wildly implausible prices bit-exact (the price filter is itself
     part of `meas_err` since it's a value-range check, not parsing).
  2. **Decimal-shift correction** — Dickerson, Robotti, Rossetti (2025)
     WRDS-MMN rule: prices in (300, 3000] → /10; (3000, 30000] → /100;
     elsewhere keep as-is. Drops unresolvables.
  3. **Persist `decimal_shift_applied` flag** — boolean column, True for
     rows where /10 or /100 was applied. Persisted (not apply-and-discard)
     so the spot-check trail per A6/A7 verification step 4 can recover
     which trades were shifted.

This script runs only when emitting the corrected column family. The raw
family bypasses it entirely (per A1.1). Bounce-back is the next downstream
stage; distressed daily filters live in the daily-layer pipeline.

Stage order (normative per A7.2):
  decimal-shift → bounce-back → VWAP→daily → distressed filters 1-4

Usage:
  python scripts/apply_decimal_shift.py
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_IN = REPO_ROOT / "data" / "development" / "trace_clean_raw.parquet"
HOLD_IN = REPO_ROOT / "data" / "holdout" / "trace_clean_raw.parquet"
DEV_OUT = REPO_ROOT / "data" / "development" / "trace_clean_decimal_shifted.parquet"
HOLD_OUT = REPO_ROOT / "data" / "holdout" / "trace_clean_decimal_shifted.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "decimal_shift_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Output schema = input schema + decimal_shift_applied flag column.
OUTPUT_SCHEMA = pa.schema([
    pa.field("bond_id",                pa.string()),
    pa.field("cusip_id",               pa.string()),
    pa.field("company_symbol",         pa.string()),
    pa.field("trd_exctn_dt",           pa.timestamp("us")),
    pa.field("trd_exctn_tm",           pa.string()),
    pa.field("rptd_pr",                pa.float64()),
    pa.field("entrd_vol_qt",           pa.float64()),
    pa.field("sub_prdct",              pa.string()),
    pa.field("rpt_side_cd",            pa.string()),
    pa.field("trdg_mkt_cd",            pa.string()),
    pa.field("trd_mod_3",              pa.string()),
    pa.field("bloomberg_identifier",   pa.string()),
    pa.field("scrty_type_cd",          pa.string()),
    pa.field("decimal_shift_applied",  pa.bool_()),
])


def load_thresholds() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    return cfg["trace_cleaning"]


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def apply_decimal_shift_vec(prices: np.ndarray, floor: float, ceiling: float):
    """
    Vectorized decimal-shift + price-plausibility on a 1-D price array.

    Returns (corrected_prices, shift_applied_flag, in_range_mask):
      - corrected_prices: float64 array with corrected values; NaN where
        no shift restores a valid price in (floor, ceiling].
      - shift_applied_flag: bool array, True where /10 or /100 was applied
        (False for in-range trades and dropped trades).
      - in_range_mask: bool array, True for rows the caller should keep.

    A row is kept iff its FINAL price is in (floor, ceiling]. Rows with
    pre-shift prices ≤ floor (e.g. 0) or > pre_correction_ceiling (30000)
    have no shift that restores them → dropped.
    """
    p = prices.astype(float)

    in_range_no_shift = (p > floor) & (p <= ceiling)
    in_range_div10 = (p / 10 > floor) & (p / 10 <= ceiling) & ~in_range_no_shift
    in_range_div100 = (
        (p / 100 > floor) & (p / 100 <= ceiling)
        & ~in_range_no_shift & ~in_range_div10
    )

    corrected = np.where(
        in_range_no_shift, p,
        np.where(in_range_div10, p / 10,
                 np.where(in_range_div100, p / 100, np.nan))
    )

    shift_applied = in_range_div10 | in_range_div100
    in_range_mask = in_range_no_shift | in_range_div10 | in_range_div100

    return corrected, shift_applied, in_range_mask


def process_partition(input_path: Path, output_path: Path, cfg: dict) -> dict:
    floor = float(cfg["price_floor"])
    pre_ceiling = float(cfg["pre_correction_ceiling"])
    ceiling = float(cfg["price_ceiling"])

    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    counts = {
        "input_rows": 0,
        "dropped_pre_ceiling": 0,        # prices > pre_correction_ceiling
        "dropped_floor": 0,              # prices ≤ floor (zero / negative)
        "kept_no_shift": 0,
        "kept_shift_div10": 0,
        "kept_shift_div100": 0,
        "output_rows": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".parquet.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    pf = pq.ParquetFile(str(input_path))
    n_rows = pf.metadata.num_rows
    counts["input_rows"] = n_rows
    n_batches_est = max(1, (n_rows + 199_999) // 200_000)

    try:
        for batch in tqdm(
            pf.iter_batches(batch_size=200_000),
            total=n_batches_est,
            desc=output_path.parent.name,
            unit="batch",
        ):
            if batch.num_rows == 0:
                continue
            prices = batch.column("rptd_pr").to_numpy(zero_copy_only=False)

            # First gate: prices > pre_correction_ceiling are unrecoverable per
            # Dick-Nielsen 2009 p.528. Drop and count.
            within_pre_ceiling = prices <= pre_ceiling
            counts["dropped_pre_ceiling"] += int((~within_pre_ceiling).sum())

            # Apply decimal-shift to the surviving subset.
            sub_prices = prices[within_pre_ceiling]
            corrected, shift_applied, in_range = apply_decimal_shift_vec(
                sub_prices, floor, ceiling
            )

            # Rows below floor / negative that no shift recovers are dropped here.
            counts["dropped_floor"] += int((~in_range).sum())
            counts["kept_no_shift"] += int(in_range.sum() - shift_applied.sum())
            counts["kept_shift_div10"] += int(((corrected * 10) == sub_prices)[in_range].sum())
            counts["kept_shift_div100"] += int(((corrected * 100) == sub_prices)[in_range].sum())

            # Rebuild a full-length mask aligning with the original batch order.
            keep_full = np.zeros(batch.num_rows, dtype=bool)
            keep_full[within_pre_ceiling] = in_range

            # Corrected prices + flag, aligned to the batch.
            corrected_full = prices.copy()
            corrected_full[within_pre_ceiling] = corrected
            shift_full = np.zeros(batch.num_rows, dtype=bool)
            shift_full[within_pre_ceiling] = shift_applied

            # Build the output batch: take the original columns, swap rptd_pr
            # for corrected_full, append decimal_shift_applied, then filter on
            # keep_full.
            cols = {field.name: batch.column(field.name) for field in batch.schema}
            cols["rptd_pr"] = pa.array(corrected_full, type=pa.float64())
            cols["decimal_shift_applied"] = pa.array(shift_full, type=pa.bool_())
            out_batch = pa.RecordBatch.from_pydict(cols)
            out_table = pa.Table.from_batches([out_batch])
            # Apply keep mask
            out_table = out_table.filter(pa.array(keep_full))
            if out_table.num_rows == 0:
                continue
            # Cast to OUTPUT_SCHEMA — explicit ordering and dtype.
            out_table = out_table.select([f.name for f in OUTPUT_SCHEMA])
            out_table = out_table.cast(OUTPUT_SCHEMA)

            if writer is None:
                writer = pq.ParquetWriter(str(tmp_path), OUTPUT_SCHEMA)
            writer.write_table(out_table)
            counts["output_rows"] += out_table.num_rows
    finally:
        if writer is not None:
            writer.close()

    if tmp_path.exists():
        actual = pq.read_metadata(str(tmp_path)).num_rows
        if actual != counts["output_rows"]:
            tmp_path.unlink()
            raise AssertionError(
                f"Tmp parquet row count {actual:,} != counter {counts['output_rows']:,}"
            )
        os.replace(tmp_path, output_path)

    return counts


def write_report(dev_counts: dict, hold_counts: dict, cfg: dict) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": {
            "price_floor": cfg["price_floor"],
            "price_ceiling": cfg["price_ceiling"],
            "pre_correction_ceiling": cfg["pre_correction_ceiling"],
        },
        "stage": "decimal_shift_corrected_branch",
        "registry_role": (
            "meas_err = ON, stage 1: price plausibility + WRDS-MMN decimal-shift. "
            "Bounce-back is stage 2; distressed daily filters 1-4 are stage 3 "
            "(daily-layer). Raw family bypasses this script per A1."
        ),
        "rows_dev": dev_counts,
        "rows_holdout": hold_counts,
        "inputs": {
            "dev": str(DEV_IN.relative_to(REPO_ROOT)),
            "holdout": str(HOLD_IN.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "dev": str(DEV_OUT.relative_to(REPO_ROOT)),
            "holdout": str(HOLD_OUT.relative_to(REPO_ROOT)),
        },
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report written: {REPORT_OUT}")


def main():
    if not DEV_IN.exists():
        print(f"ERROR: Required input not found: {DEV_IN}", file=sys.stderr)
        print("Run scripts/preprocess_trace.py first.", file=sys.stderr)
        sys.exit(1)

    cfg = load_thresholds()
    print(f"Thresholds loaded from {THRESHOLDS_FILE}")
    print(f"  price_floor={cfg['price_floor']}, price_ceiling={cfg['price_ceiling']}, "
          f"pre_correction_ceiling={cfg['pre_correction_ceiling']}")
    print("Stage: meas_err = ON — price plausibility + WRDS-MMN decimal-shift")

    print("Processing development partition...")
    dev_counts = process_partition(DEV_IN, DEV_OUT, cfg)
    print(
        f"  dev: in={dev_counts['input_rows']:,} "
        f"out={dev_counts['output_rows']:,} "
        f"dropped_floor={dev_counts['dropped_floor']:,} "
        f"dropped_pre_ceiling={dev_counts['dropped_pre_ceiling']:,} "
        f"shifted_div10={dev_counts['kept_shift_div10']:,} "
        f"shifted_div100={dev_counts['kept_shift_div100']:,}"
    )

    print("Processing holdout partition (mechanical; no statistics surfaced)...")
    hold_counts = process_partition(HOLD_IN, HOLD_OUT, cfg)

    write_report(dev_counts, hold_counts, cfg)

    print("\nDone.")
    print(f"  Development: {dev_counts['output_rows']:,} rows → {DEV_OUT}")
    print(f"  Holdout:     {hold_counts['output_rows']:,} rows → {HOLD_OUT}")


if __name__ == "__main__":
    main()
