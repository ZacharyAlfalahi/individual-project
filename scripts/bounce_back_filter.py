"""
DRR (Dickerson, Robotti, Rossetti 2026) bounce-back filter for TRACE.

Runs after preprocess_trace.py and before build_monthly_panel.py. Reads each
of data/development/trace_clean.parquet and data/holdout/trace_clean.parquet,
removes transaction-level price spikes that revert within a short lookahead
window, and overwrites the input atomically (write to .tmp, verify row count,
then os.replace).

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
to inform any decision — consistent with how preprocess_trace.py already
writes data/holdout/trace_clean.parquet.

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
DEV_PARQUET = REPO_ROOT / "data" / "development" / "trace_clean.parquet"
HOLD_PARQUET = REPO_ROOT / "data" / "holdout" / "trace_clean.parquet"
REPORT_PATH = REPO_ROOT / "data" / "development" / "cleaning_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Identical to preprocess_trace.OUTPUT_SCHEMA — copied verbatim because the
# downstream build_monthly_panel.py reads back from this exact schema and any
# drift would break it. Keep in lockstep with preprocess_trace.py:259-272.
OUTPUT_SCHEMA = pa.schema([
    pa.field("bond_id",              pa.string()),
    pa.field("company_symbol",       pa.string()),
    pa.field("trd_exctn_dt",         pa.timestamp("us")),
    pa.field("trd_exctn_tm",         pa.string()),
    pa.field("rptd_pr",              pa.float64()),
    pa.field("entrd_vol_qt",         pa.float64()),
    pa.field("sub_prdct",            pa.string()),
    pa.field("rpt_side_cd",          pa.string()),
    pa.field("trdg_mkt_cd",          pa.string()),
    pa.field("trd_mod_3",            pa.string()),
    pa.field("bloomberg_identifier", pa.string()),
    pa.field("scrty_type_cd",        pa.string()),
])

REQUIRED_PARAM_KEYS = (
    "threshold_abs", "lookahead", "window",
    "back_to_anchor_tol", "candidate_slack_abs",
    "par_cooldown_after_flag",
    "par_spike_heuristic", "par_level", "par_band", "par_min_run",
)

# Legacy keys from the removed initial-price-error filter (a the project cold-
# start patch with no DRR provenance). Their presence in a prior run's report
# is a hard error — re-run preprocess_trace.py to regenerate a clean report.
LEGACY_INIT_PRICE_KEYS = (
    "dropped_init_price_error_dev",
    "dropped_init_price_error_hold",
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

    Returns (keep_mask, n_dropped). See spec Section 3.

    Anchor decoupling (the project extension): par-snap, when active, applies
    only to the *flagging* decision — it prevents false positives for at-par
    bonds whose trailing median has drifted slightly off par. The *recovery*
    check always uses the raw trailing median, because recovery asks "did the
    bond return to its actual trading level", which is the median by
    construction. A single par-snapped anchor for both checks creates a dead
    zone for any bond whose median is in [par - par_band, par + par_band] but
    not exactly at par.

    Cooldown scope (DRR Table A.2 clause iii): the cooldown is described as
    "rows skipped after flagging par blocks". It fires only when the dropped
    trade was flagged via the par-snap path, not after every drop.
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

        # Read-only lookahead — does not consume future trades. Recovery is
        # judged against the median (the bond's actual level), never par.
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

def process_partition(input_path: Path, output_path: Path, params: dict,
                      output_schema: pa.Schema = OUTPUT_SCHEMA) -> dict:
    """Read input parquet, apply per-bond filter, atomically overwrite output.

    Returns dict with: input_rows, kept_rows, dropped_bounce_back.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    sorted_tmp = input_path.with_name(input_path.name + ".sorted.tmp")
    tmp_path = output_path.with_suffix(".parquet.tmp")
    # Ensure no stale .tmp from a prior crashed run is around.
    for stale in (sorted_tmp, tmp_path):
        if stale.exists():
            stale.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stage 1: stream-sort to intermediate parquet via DuckDB. DuckDB's
    # external merge sort spills to disk transparently and bounds peak RAM
    # to its buffer-pool target — well within the workstation's memory.
    # polars sink_parquet does NOT externalize sort at this scale (verified
    # by repeated OOM kills); DuckDB is the right tool for this step.
    # The intermediate schema may differ slightly from OUTPUT_SCHEMA (e.g.
    # large_utf8 vs utf8) — the cast at flush time normalises it.
    print(f"  Stream-sorting {input_path.name} → {sorted_tmp.name} via DuckDB ...")
    # DuckDB's `COPY ... TO` clause does not reliably accept parameter binding
    # for the target path, so paths are interpolated directly. Both sides are
    # internal (built from REPO_ROOT) — not user input — and we escape any
    # single quote that could appear in unusual filesystem layouts.
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

    writer = None
    kept_rows = 0
    n_bb_total = 0
    bond_segments: list = []  # list[pa.RecordBatch] for the bond currently being accumulated
    current_bond_id = None

    def _flush_current_bond():
        nonlocal kept_rows, n_bb_total, writer
        if not bond_segments:
            return
        sub_table = pa.Table.from_batches(bond_segments)
        sub_df = pl.from_arrow(sub_table)
        prices = sub_df.get_column("rptd_pr").to_list()
        mask, n_bb = _apply_bounce_back_loop(prices, params)
        n_bb_total += n_bb
        if not any(mask):
            return
        kept_sub = sub_df if all(mask) else sub_df.filter(pl.Series(mask))
        table = kept_sub.to_arrow().cast(output_schema)
        if writer is None:
            writer = pq.ParquetWriter(str(tmp_path), output_schema)
        writer.write_table(table)
        kept_rows += kept_sub.height

    try:
        pf = pq.ParquetFile(str(sorted_tmp))
        n_batches_est = max(1, (input_rows + 199_999) // 200_000)
        for batch in tqdm(pf.iter_batches(batch_size=200_000),
                          total=n_batches_est,
                          desc=output_path.parent.name,
                          unit="batch"):
            if batch.num_rows == 0:
                continue
            # Partition-boundary invariant: Stage 1 (preprocess_trace.py)
            # guarantees positive, finite, non-null prices. Assert here because
            # both NaN (IEEE-754: every comparison evaluates False) and null
            # (None propagating into Python arithmetic) would silently break
            # the filter for any bond containing such a price. pc.is_finite
            # returns NULL for nulls and pc.all skips nulls, so the two cases
            # must be checked separately.
            price_col = batch.column("rptd_pr")
            if (price_col.null_count > 0
                    or not pc.all(pc.is_finite(price_col)).as_py()):
                raise AssertionError(
                    f"Non-finite price found in {input_path.name} batch — "
                    "Stage 1 contract violated. Re-run preprocess_trace.py."
                )
            bond_col = batch.column("bond_id").to_pylist()
            # Bonds are sorted, so contiguous runs share a bond_id. Detect
            # transitions in a single pass.
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
        if writer is not None:
            writer.close()
        # Always remove the intermediate sorted parquet — it can be ~25 GB
        # for the dev partition.
        if sorted_tmp.exists():
            try:
                sorted_tmp.unlink()
            except OSError:
                pass

    # Verify the .tmp parquet matches the expected row count AND schema before
    # atomic replace — never destroy the input unless the new file is fully
    # written and structurally identical to the input contract.
    if tmp_path.exists():
        actual = pq.read_metadata(str(tmp_path)).num_rows
        if actual != kept_rows:
            tmp_path.unlink()
            raise AssertionError(
                f"Tmp parquet row count {actual:,} != counter {kept_rows:,} "
                f"for {output_path}"
            )
        written_schema = pq.read_schema(str(tmp_path))
        if not written_schema.equals(output_schema):
            tmp_path.unlink()
            raise AssertionError(
                f"Schema drift in tmp parquet for {output_path.name}:\n"
                f"  expected: {output_schema}\n  written:  {written_schema}"
            )
        os.replace(tmp_path, output_path)
    else:
        # No tmp written → no writer was opened → every bond's mask was empty.
        # Original file is untouched; safe to re-run after investigating.
        if kept_rows != 0:
            raise AssertionError(
                f"No tmp parquet written but kept_rows={kept_rows} for {output_path}"
            )
        raise AssertionError(
            f"Bounce-back removed every row in {input_path}; no tmp parquet "
            f"was written. Original file is unchanged. Investigate before re-running."
        )

    if input_rows - kept_rows != n_bb_total:
        raise AssertionError(
            f"Row arithmetic broken for {output_path.name}: "
            f"input={input_rows:,} kept={kept_rows:,} bb_drop={n_bb_total:,} "
            f"(input-kept={input_rows - kept_rows:,} != bb_drop={n_bb_total:,})"
        )

    return {
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "dropped_bounce_back": n_bb_total,
    }


# ---------------------------------------------------------------------------
# cleaning_report.json update
# ---------------------------------------------------------------------------

def _assert_additivity(rows: dict) -> None:
    expected = rows["raw_total"]
    actual = (
        rows["dropped_trc_st"]
        + rows["dropped_cancelled_original"]
        + rows["dropped_asof_cd"]
        + rows["dropped_wis_fl"]
        + rows["dropped_price_plausibility"]
        + rows["decimal_shift_unresolvable_dropped"]
        + rows["dropped_interdealer_duplicate"]
        + rows["dropped_invalid_date"]
        + rows["dropped_bounce_back_dev"]
        + rows["dropped_bounce_back_hold"]
        + rows["development_rows"]
        + rows["holdout_rows"]
    )
    if actual != expected:
        raise AssertionError(
            f"Counter additivity broken: components sum to {actual:,} "
            f"but raw_total is {expected:,} (gap {expected - actual:,})."
        )
    final = rows["final_clean_total"]
    expected_final = rows["development_rows"] + rows["holdout_rows"]
    if final != expected_final:
        raise AssertionError(
            f"final_clean_total {final:,} != development_rows + holdout_rows "
            f"{expected_final:,} — report has drifted from its own counters."
        )


def update_cleaning_report(report_path: Path, dev_stats: dict, hold_stats: dict,
                           thresholds_sha: str) -> None:
    if not report_path.exists():
        raise FileNotFoundError(
            f"cleaning_report.json not found at {report_path}. "
            "Run preprocess_trace.py first."
        )
    with open(report_path) as f:
        report = json.load(f)

    rows = report["rows"]

    # Legacy fields from a removed cold-start filter — their presence means
    # this report is from a prior run that used the pre-strict-fidelity
    # implementation. Re-running preprocess_trace.py is required to regenerate
    # a clean report; we refuse to silently migrate.
    legacy_found = sorted(k for k in LEGACY_INIT_PRICE_KEYS if k in rows)
    if legacy_found:
        raise AssertionError(
            f"Legacy init_price_error fields detected in {report_path}: "
            f"{legacy_found}. These belong to a removed filter (the project "
            "cold-start patch, deleted for DRR fidelity). Delete the report "
            "and re-run preprocess_trace.py."
        )

    rows["dropped_bounce_back_dev"] = int(dev_stats["dropped_bounce_back"])
    rows["dropped_bounce_back_hold"] = int(hold_stats["dropped_bounce_back"])
    rows["development_rows"] = int(dev_stats["kept_rows"])
    rows["holdout_rows"] = int(hold_stats["kept_rows"])
    rows["final_clean_total"] = int(dev_stats["kept_rows"] + hold_stats["kept_rows"])
    rows["parquet_row_count_verified"] = True

    report["bounce_back_filter_applied"] = True
    report["bounce_back_run_timestamp"] = datetime.now(timezone.utc).isoformat()
    report["bounce_back_params_sha256"] = thresholds_sha

    _assert_additivity(rows)

    tmp = report_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, report_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    params = load_bounce_back_config()
    sha = thresholds_sha256()
    print(f"Loaded bounce_back_filter config (thresholds sha256: {sha[:12]}...)")

    print("Processing development partition...")
    dev_stats = process_partition(DEV_PARQUET, DEV_PARQUET, params)
    print(
        f"  dev: input={dev_stats['input_rows']:,} kept={dev_stats['kept_rows']:,} "
        f"bb_drop={dev_stats['dropped_bounce_back']:,}"
    )

    print("Processing holdout partition (mechanical; no statistics surfaced)...")
    hold_stats = process_partition(HOLD_PARQUET, HOLD_PARQUET, params)
    print(
        f"  holdout: input={hold_stats['input_rows']:,} kept={hold_stats['kept_rows']:,} "
        f"bb_drop={hold_stats['dropped_bounce_back']:,}"
    )

    update_cleaning_report(REPORT_PATH, dev_stats, hold_stats, sha)
    print(f"Updated {REPORT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
