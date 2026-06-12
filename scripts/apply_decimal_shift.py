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
  python scripts/apply_decimal_shift.py             # development partition only
  python scripts/apply_decimal_shift.py --holdout   # ALSO process holdout —
      reserved for the single post-freeze pipeline run (weeks 13-14).
      NEVER pass during development (inviolable data rule).
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional

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


class DecimalShiftResult(NamedTuple):
    """Vectorized decimal-shift output. The three keep-categories
    (no-shift, div10, div100) are mutually exclusive and their union is
    in_range_mask — the audit counts are taken directly from these masks
    rather than reverse-engineered from float arithmetic."""
    corrected: np.ndarray      # float64; NaN where no shift restores validity
    shift_applied: np.ndarray  # bool; div10 | div100
    in_range_mask: np.ndarray  # bool; rows the caller should keep
    div10_mask: np.ndarray     # bool; /10 applied
    div100_mask: np.ndarray    # bool; /100 applied


def apply_decimal_shift_vec(
    prices: np.ndarray, floor: float, ceiling: float
) -> DecimalShiftResult:
    """
    Vectorized decimal-shift + price-plausibility on a 1-D price array.

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

    return DecimalShiftResult(
        corrected=corrected,
        shift_applied=shift_applied,
        in_range_mask=in_range_mask,
        div10_mask=in_range_div10,
        div100_mask=in_range_div100,
    )


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
            res = apply_decimal_shift_vec(sub_prices, floor, ceiling)
            corrected, shift_applied, in_range = (
                res.corrected, res.shift_applied, res.in_range_mask
            )

            # Rows below floor / negative that no shift recovers are dropped here.
            # Category counts come straight from the masks (mutually exclusive,
            # union == in_range) — never reconstructed via float equality.
            counts["dropped_floor"] += int((~in_range).sum())
            counts["kept_no_shift"] += int((in_range & ~shift_applied).sum())
            counts["kept_shift_div10"] += int(res.div10_mask.sum())
            counts["kept_shift_div100"] += int(res.div100_mask.sum())

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

    category_sum = (
        counts["kept_no_shift"]
        + counts["kept_shift_div10"]
        + counts["kept_shift_div100"]
    )
    if category_sum != counts["output_rows"]:
        raise AssertionError(
            f"Shift-count decomposition broken: no_shift + div10 + div100 = "
            f"{category_sum:,} != output_rows {counts['output_rows']:,}"
        )

    return counts


def write_report(
    dev_counts: dict, hold_counts: Optional[dict], cfg: dict
) -> None:
    """hold_counts is None during development — no holdout statistic is
    computed or surfaced into this (development-side) report."""
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
        "holdout_processed": hold_counts is not None,
        "rows_dev": dev_counts,
        "inputs": {
            "dev": str(DEV_IN.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "dev": str(DEV_OUT.relative_to(REPO_ROOT)),
        },
    }
    if hold_counts is not None:
        report["rows_holdout"] = hold_counts
        report["inputs"]["holdout"] = str(HOLD_IN.relative_to(REPO_ROOT))
        report["outputs"]["holdout"] = str(HOLD_OUT.relative_to(REPO_ROOT))
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report written: {REPORT_OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--holdout", action="store_true",
        help="Also process the holdout partition. NEVER pass during "
             "development; reserved for the single post-freeze pipeline "
             "run in weeks 13-14 (inviolable data rule).",
    )
    args = parser.parse_args()

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

    hold_counts = None
    if args.holdout:
        print("Processing holdout partition (post-freeze run)...")
        hold_counts = process_partition(HOLD_IN, HOLD_OUT, cfg)
    else:
        print("Holdout partition NOT processed (development mode; "
              "pass --holdout for the post-freeze run).")

    write_report(dev_counts, hold_counts, cfg)

    print("\nDone.")
    print(f"  Development: {dev_counts['output_rows']:,} rows → {DEV_OUT}")
    if hold_counts is not None:
        print(f"  Holdout:     {hold_counts['output_rows']:,} rows → {HOLD_OUT}")


if __name__ == "__main__":
    main()
