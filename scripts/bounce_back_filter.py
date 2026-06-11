"""
DRR (Dickerson, Robotti, Rossetti 2026) bounce-back filter — corrected
branch, stage 2 of the meas_err pipeline.

Reads `trace_clean_decimal_shifted.parquet` (output of apply_decimal_shift.py),
applies the per-bond bounce-back filter from DRR Table A.2, and writes
`trace_clean_corr.parquet`. Dropped trades are persisted to a companion
artifact `bounceback_dropped_<partition>.parquet` for the audit trail
(spot-checkable via the verification step 4 in the Phase 1 plan).

Stage order (normative per A7.2 of the registry amendments):
  decimal-shift → bounce-back → VWAP→daily → distressed filters 1-4

Per-bond filter (DRR Table A.2): rolling trailing median of the last `window`
unique prices serves as the anchor; a trade more than `threshold_abs` from the
anchor is dropped only if at least one of the next `lookahead` trades shows a
recovery to within (back_to_anchor_tol + candidate_slack_abs) of the median.
A par-snap heuristic redirects the flagging anchor to par for near-par bonds;
cooldown applies only after a par-block drop (matching DRR Table A.2's "rows
skipped after flagging par blocks"). See docs/bounce_back_filter_spec.md.

All parameters live in docs/thresholds.yaml under the `bounce_back_filter` key.
This script never hard-codes thresholds.

Holdout processing is a mechanical transformation with parameters fixed in
thresholds.yaml. No statistic computed on holdout is read, surfaced, or used
to inform any decision.

Usage:
  python scripts/bounce_back_filter.py
"""

import hashlib
import json
import os
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import duckdb
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import polars as pl
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_IN = REPO_ROOT / "data" / "development" / "trace_clean_decimal_shifted.parquet"
HOLD_IN = REPO_ROOT / "data" / "holdout" / "trace_clean_decimal_shifted.parquet"
DEV_OUT = REPO_ROOT / "data" / "development" / "trace_clean_corr.parquet"
HOLD_OUT = REPO_ROOT / "data" / "holdout" / "trace_clean_corr.parquet"
DEV_DROPPED = REPO_ROOT / "data" / "development" / "bounceback_dropped_dev.parquet"
HOLD_DROPPED = REPO_ROOT / "data" / "holdout" / "bounceback_dropped_hold.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "bounce_back_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Carries the input schema (incl. decimal_shift_applied) through unchanged.
# trace_clean_corr.parquet has the SAME schema as the decimal-shifted input;
# bounceback_dropped artefacts have the same schema for the dropped trades.
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

REQUIRED_PARAM_KEYS = (
    "threshold_abs", "lookahead", "window",
    "back_to_anchor_tol", "candidate_slack_abs",
    "par_cooldown_after_flag",
    "par_spike_heuristic", "par_level", "par_band", "par_min_run",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_bounce_back_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    if "bounce_back_filter" not in cfg:
        raise KeyError(
            f"`bounce_back_filter` section missing from {THRESHOLDS_FILE}. "
            "See docs/bounce_back_filter_spec.md Section 5."
        )
    params = cfg["bounce_back_filter"]
    missing = [k for k in REQUIRED_PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"bounce_back_filter missing required keys: {missing}")
    return params


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Pure per-bond filter
# ---------------------------------------------------------------------------

def _push_unique(dq: deque, p: float) -> None:
    if not dq or dq[-1] != p:
        dq.append(p)


def _near_par_run(dq: deque, par_level: float, par_band: float, par_min_run: int) -> bool:
    if len(dq) < par_min_run:
        return False
    return all(abs(x - par_level) <= par_band for x in list(dq)[-par_min_run:])


def _apply_bounce_back_loop(prices, params: dict):
    """Sequential per-bond bounce-back filter (DRR Table A.2).

    Returns (keep_mask, n_dropped). See bounce_back_filter_spec.md Section 3.
    """
    threshold_abs = float(params["threshold_abs"])
    lookahead = int(params["lookahead"])
    window = int(params["window"])
    back_to_anchor_tol = float(params["back_to_anchor_tol"])
    candidate_slack_abs = float(params["candidate_slack_abs"])
    par_cooldown = int(params["par_cooldown_after_flag"])
    par_on = bool(params["par_spike_heuristic"])
    par_level = float(params["par_level"])
    par_band = float(params["par_band"])
    par_min_run = int(params["par_min_run"])

    recovery_tol = back_to_anchor_tol + candidate_slack_abs

    n = len(prices)
    keep = [True] * n
    trailing: deque = deque(maxlen=window)
    cooldown_remaining = 0
    n_dropped = 0

    for i, p in enumerate(prices):
        if cooldown_remaining > 0:
            _push_unique(trailing, p)
            cooldown_remaining -= 1
            continue

        if len(trailing) < 2:
            _push_unique(trailing, p)
            continue

        median_anchor = median(trailing)
        par_snap_active = par_on and _near_par_run(trailing, par_level, par_band, par_min_run)
        flag_anchor = par_level if par_snap_active else median_anchor

        if abs(p - flag_anchor) <= threshold_abs:
            _push_unique(trailing, p)
            continue

        hi = min(i + 1 + lookahead, n)
        recovered = any(abs(prices[j] - median_anchor) <= recovery_tol for j in range(i + 1, hi))

        if recovered:
            keep[i] = False
            n_dropped += 1
            if par_snap_active:
                cooldown_remaining = par_cooldown
        else:
            _push_unique(trailing, p)

    return keep, n_dropped


# ---------------------------------------------------------------------------
# Partition orchestration
# ---------------------------------------------------------------------------

def process_partition(input_path: Path, output_path: Path, dropped_path: Path,
                      params: dict, output_schema: pa.Schema = OUTPUT_SCHEMA) -> dict:
    """Read input parquet, apply per-bond bounce-back filter, write kept rows
    to output_path and dropped rows to dropped_path (companion artifact).

    Returns dict with: input_rows, kept_rows, dropped_bounce_back.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    sorted_tmp = input_path.with_name(input_path.name + ".sorted.tmp")
    out_tmp = output_path.with_suffix(".parquet.tmp")
    dropped_tmp = dropped_path.with_suffix(".parquet.tmp")
    for stale in (sorted_tmp, out_tmp, dropped_tmp):
        if stale.exists():
            stale.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: stream-sort to intermediate parquet via DuckDB (bounded RAM via
    # disk spill). Polars sink_parquet does not externalise sort at this scale.
    print(f"  Stream-sorting {input_path.name} → {sorted_tmp.name} via DuckDB ...")
    in_lit = "'" + str(input_path).replace("'", "''") + "'"
    out_lit = "'" + str(sorted_tmp).replace("'", "''") + "'"
    with duckdb.connect() as con:
        con.execute(
            f"""
            COPY (
                SELECT * FROM read_parquet({in_lit})
                ORDER BY bond_id, trd_exctn_dt, trd_exctn_tm NULLS LAST
            ) TO {out_lit} (FORMAT 'parquet')
            """
        )

    input_rows = pq.read_metadata(str(sorted_tmp)).num_rows
    print(f"  Sorted {input_rows:,} rows; iterating bond-by-bond...")

    kept_writer = None
    dropped_writer = None
    kept_rows = 0
    dropped_rows = 0
    bond_segments: list = []
    current_bond_id = None

    def _flush_current_bond():
        nonlocal kept_rows, dropped_rows, kept_writer, dropped_writer
        if not bond_segments:
            return
        sub_table = pa.Table.from_batches(bond_segments)
        sub_df = pl.from_arrow(sub_table)
        prices = sub_df.get_column("rptd_pr").to_list()
        mask, n_bb = _apply_bounce_back_loop(prices, params)

        # Write KEPT rows
        if any(mask):
            kept_sub = sub_df if all(mask) else sub_df.filter(pl.Series(mask))
            kept_table = kept_sub.to_arrow().cast(output_schema)
            if kept_writer is None:
                kept_writer = pq.ParquetWriter(str(out_tmp), output_schema)
            kept_writer.write_table(kept_table)
            kept_rows += kept_sub.height

        # Write DROPPED rows to companion artifact for audit trail
        if n_bb > 0:
            dropped_mask = [not k for k in mask]
            dropped_sub = sub_df.filter(pl.Series(dropped_mask))
            dropped_table = dropped_sub.to_arrow().cast(output_schema)
            if dropped_writer is None:
                dropped_writer = pq.ParquetWriter(str(dropped_tmp), output_schema)
            dropped_writer.write_table(dropped_table)
            dropped_rows += dropped_sub.height

    try:
        pf = pq.ParquetFile(str(sorted_tmp))
        n_batches_est = max(1, (input_rows + 199_999) // 200_000)
        for batch in tqdm(pf.iter_batches(batch_size=200_000),
                          total=n_batches_est,
                          desc=output_path.parent.name,
                          unit="batch"):
            if batch.num_rows == 0:
                continue
            # Partition-boundary invariant: prices must be positive, finite,
            # non-null at this stage (decimal-shift already screened).
            price_col = batch.column("rptd_pr")
            if (price_col.null_count > 0
                    or not pc.all(pc.is_finite(price_col)).as_py()):
                raise AssertionError(
                    f"Non-finite price found in {input_path.name} batch — "
                    "decimal-shift stage contract violated."
                )
            bond_col = batch.column("bond_id").to_pylist()
            run_starts = [0]
            for i in range(1, len(bond_col)):
                if bond_col[i] != bond_col[i - 1]:
                    run_starts.append(i)
            run_starts.append(len(bond_col))

            for j in range(len(run_starts) - 1):
                s, e = run_starts[j], run_starts[j + 1]
                bid = bond_col[s]
                seg = batch.slice(s, e - s)
                if current_bond_id is None or bid == current_bond_id:
                    bond_segments.append(seg)
                    current_bond_id = bid
                else:
                    _flush_current_bond()
                    bond_segments = [seg]
                    current_bond_id = bid

        _flush_current_bond()
    finally:
        if kept_writer is not None:
            kept_writer.close()
        if dropped_writer is not None:
            dropped_writer.close()
        if sorted_tmp.exists():
            try:
                sorted_tmp.unlink()
            except OSError:
                pass

    # Verify and rename kept output
    if out_tmp.exists():
        actual = pq.read_metadata(str(out_tmp)).num_rows
        if actual != kept_rows:
            out_tmp.unlink()
            raise AssertionError(
                f"Kept tmp parquet row count {actual:,} != counter {kept_rows:,}"
            )
        written_schema = pq.read_schema(str(out_tmp))
        if not written_schema.equals(output_schema):
            out_tmp.unlink()
            raise AssertionError(
                f"Schema drift in kept tmp parquet for {output_path.name}"
            )
        os.replace(out_tmp, output_path)
    elif kept_rows != 0:
        raise AssertionError(
            f"No kept tmp parquet written but kept_rows={kept_rows}"
        )
    else:
        raise AssertionError(
            f"Bounce-back removed every row in {input_path}; investigate before re-running."
        )

    # Verify and rename dropped companion artifact (if any)
    if dropped_tmp.exists():
        actual = pq.read_metadata(str(dropped_tmp)).num_rows
        if actual != dropped_rows:
            dropped_tmp.unlink()
            raise AssertionError(
                f"Dropped tmp parquet row count {actual:,} != counter {dropped_rows:,}"
            )
        os.replace(dropped_tmp, dropped_path)
    elif dropped_rows != 0:
        raise AssertionError(
            f"No dropped tmp parquet written but dropped_rows={dropped_rows}"
        )

    if input_rows - kept_rows != dropped_rows:
        raise AssertionError(
            f"Row arithmetic broken for {output_path.name}: "
            f"input={input_rows:,} kept={kept_rows:,} dropped={dropped_rows:,}"
        )

    return {
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "dropped_bounce_back": dropped_rows,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(dev_stats: dict, hold_stats: dict, thresholds_sha: str) -> None:
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha,
        "stage": "bounce_back_corrected_branch",
        "registry_role": (
            "meas_err = ON, stage 2: DRR Table A.2 per-bond bounce-back filter. "
            "Stage 3 (distressed daily filters 1-4) runs on the daily layer "
            "after VWAP aggregation. Raw family bypasses this script per A1."
        ),
        "rows_dev": dev_stats,
        "rows_holdout": hold_stats,
        "inputs": {
            "dev": str(DEV_IN.relative_to(REPO_ROOT)),
            "holdout": str(HOLD_IN.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "dev_kept": str(DEV_OUT.relative_to(REPO_ROOT)),
            "dev_dropped": str(DEV_DROPPED.relative_to(REPO_ROOT)),
            "holdout_kept": str(HOLD_OUT.relative_to(REPO_ROOT)),
            "holdout_dropped": str(HOLD_DROPPED.relative_to(REPO_ROOT)),
        },
        "audit_trail_note": (
            "Companion 'dropped' parquets persist the trades the bounce-back "
            "filter removed. These satisfy the Phase 1 verification step 4 "
            "(spot-check a known bounce-back trade present in trace_clean_raw "
            "and dropped from trace_clean_corr)."
        ),
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    params = load_bounce_back_config()
    sha = thresholds_sha256()
    print(f"Loaded bounce_back_filter config (thresholds sha256: {sha[:12]}...)")
    print("Stage: meas_err = ON — DRR Table A.2 per-bond bounce-back filter")

    print("Processing development partition...")
    dev_stats = process_partition(DEV_IN, DEV_OUT, DEV_DROPPED, params)
    print(
        f"  dev: input={dev_stats['input_rows']:,} kept={dev_stats['kept_rows']:,} "
        f"dropped={dev_stats['dropped_bounce_back']:,}"
    )

    print("Processing holdout partition (mechanical; no statistics surfaced)...")
    hold_stats = process_partition(HOLD_IN, HOLD_OUT, HOLD_DROPPED, params)
    print(
        f"  holdout: input={hold_stats['input_rows']:,} kept={hold_stats['kept_rows']:,} "
        f"dropped={hold_stats['dropped_bounce_back']:,}"
    )

    write_report(dev_stats, hold_stats, sha)
    print(f"Updated {REPORT_OUT}")
    print("Done.")


if __name__ == "__main__":
    main()
