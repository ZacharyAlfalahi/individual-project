"""
DRR (Dickerson, Robotti, Rossetti 2026) distressed daily filters — corrected
branch, stage 3 of the meas_err pipeline.

Reads `trace_daily_corr.parquet` (output of build_daily_panel.py --family corr),
applies FOUR per-cusip filters on the time-sorted daily series, and writes
`trace_daily_corr_filtered.parquet` with dropped days removed. The dropped days
are persisted to `distressed_dropped_<partition>.parquet` companion artefacts
with one boolean flag column per filter so the audit trail records which filter
fired (per A7.6 of the registry amendments).

Stage order (repo Step 8): decimal-shift → bounce-back → VWAP→daily → distressed
filters. DRR's FINAL filters (repo Step 9) are handled by documented equivalence:
price_threshold=300 is realised in-place by the decimal-shift target band
(trace_cleaning.price_ceiling), and dip_threshold=35 (the 2002-07 first-change
filter) is a documented deferral — see thresholds.yaml + citations_verified.md §1.

PROVENANCE — port + disclose (spec v4 Part J). This implementation is
repo-derived from Dickerson's released `trace-data-pipeline`
(stage1/helper_functions.py::ultra_distressed_filter and the four detectors
_detect_anomalies_ultra / _detect_spikes_ultra / _detect_plateaus_ultra /
flag_intraday_inconsistency_vectorized + _compute_round_mask;
ULTRA_DISTRESSED_CONFIG in stage1/_stage1_settings.py; step8 in
stage1/stage1_pipeline.py). The earlier "independent coding from the appendix;
do NOT port repo code" firewall is RETIRED for the filter implementation — it
produced a wrong implementation (absolute price-point gaps; a 25/50/75/100 round
grid). The OSBAP / DRR published-OUTPUT validation firewall is UNAFFECTED. Full
provenance + appendix-vs-repo notes: docs/data/registers/citations_verified.md §1.

FOUR filter outputs; the round-number mask is a SHARED PREDICATE consumed inside
anomaly/spike/plateau, NOT a fifth OR'd flag. All prices are % of par (100 = par),
consistent with the project's `price_vwap`/`min_price`/`max_price` — no unit
conversion. A day is dropped if ANY filter fires (repo Step 10a). Column mapping:
DRR `pr` → `price_vwap`; DRR `prc_hi`/`prc_lo` → `max_price`/`min_price`.

Filter definitions (pinned repo):

  1. Anomaly  — candidate is ultra-low (price < ultra_low_threshold) OR round;
                flag if median(neighbours priced ABOVE, ±lookback/forward window)
                / price >= min_normal_price_ratio (a RATIO, not a gap).
  2. Spike    — candidate has price > high_spike_threshold (a raw LEVEL gate, not
                a ratio) OR is a round print > 0.50; flag if
                price / median(pre-window points BELOW) >= min_spike_ratio AND a
                post price recovers to <= median_pre * recovery_ratio within the
                lookahead.
  3. Plateau  — run of >= min_plateau_days EXACT-equal ultra-low/round prices;
                flag the run if it is round OR either-side displacement
                (pre/price or post/price) >= pre_post_price_ratio.
  4. Intraday — days where min_price < intraday_price_threshold AND
                (max_price - min_price)/mean(min,max) > intraday_range_threshold.

Usage:
  python scripts/apply_distressed_filters.py
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


REPO_ROOT = Path(__file__).resolve().parent.parent
DEV_IN = REPO_ROOT / "data" / "development" / "trace_daily_corr.parquet"
HOLD_IN = REPO_ROOT / "data" / "holdout" / "trace_daily_corr.parquet"
DEV_OUT = REPO_ROOT / "data" / "development" / "trace_daily_corr_filtered.parquet"
HOLD_OUT = REPO_ROOT / "data" / "holdout" / "trace_daily_corr_filtered.parquet"
DEV_DROPPED = REPO_ROOT / "data" / "development" / "distressed_dropped_dev.parquet"
HOLD_DROPPED = REPO_ROOT / "data" / "holdout" / "distressed_dropped_hold.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "distressed_filters_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Filtered-output schema: same shape as the input daily layer.
OUTPUT_SCHEMA = pa.schema([
    pa.field("cusip_id",      pa.string()),
    pa.field("trd_exctn_dt",  pa.timestamp("us")),
    pa.field("price_vwap",    pa.float64()),
    pa.field("total_vol",     pa.float64()),
    pa.field("n_trades",      pa.int64()),
    pa.field("min_price",     pa.float64()),
    pa.field("max_price",     pa.float64()),
])

# Companion (dropped) schema: input columns + four boolean flags identifying
# which filter(s) fired on each dropped day.
DROPPED_SCHEMA = pa.schema([
    pa.field("cusip_id",          pa.string()),
    pa.field("trd_exctn_dt",      pa.timestamp("us")),
    pa.field("price_vwap",        pa.float64()),
    pa.field("total_vol",         pa.float64()),
    pa.field("n_trades",          pa.int64()),
    pa.field("min_price",         pa.float64()),
    pa.field("max_price",         pa.float64()),
    pa.field("flagged_anomaly",   pa.bool_()),
    pa.field("flagged_spike",     pa.bool_()),
    pa.field("flagged_plateau",   pa.bool_()),
    pa.field("flagged_intraday",  pa.bool_()),
])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUIRED_KEYS = (
    "ultra_low_threshold", "min_normal_price_ratio",
    "high_spike_threshold", "min_spike_ratio", "recovery_ratio",
    "plateau_ultra_low_threshold", "min_plateau_days", "pre_post_price_ratio",
    "suspicious_round_numbers", "round_tolerance",
    "lookback", "lookforward",
    "intraday_price_threshold", "intraday_range_threshold",
)

# Prices are pre-rounded to this many decimals before filtering (repo l.1050),
# which makes the plateau EXACT-equality test well-defined.
PRICE_ROUND_DECIMALS = 4


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("meas_err_distressed_filters")
    if block is None:
        raise KeyError(
            f"`meas_err_distressed_filters` section missing from {THRESHOLDS_FILE}."
        )
    missing = [k for k in REQUIRED_KEYS if k not in block]
    if missing:
        raise KeyError(f"meas_err_distressed_filters missing keys: {missing}")
    return block


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Filter primitives (pure functions on a single cusip's daily series)
#
# Faithful to the pinned repo. Prices are % of par; a candidate for a filter
# must first satisfy that filter's gate (ultra-low / high / round). Ratios use
# the repo's (denominator + 1e-10) guard so an exact-zero price cannot divide.
# ---------------------------------------------------------------------------

def _compute_round_mask(prices: np.ndarray, round_numbers: np.ndarray,
                        round_tolerance: float) -> np.ndarray:
    """True where |price - r| < round_tolerance for any r in the round set.
    NaN prices are never round. (repo _compute_round_mask)."""
    n = len(prices)
    is_round = np.zeros(n, dtype=bool)
    for i in range(n):
        p = prices[i]
        if np.isnan(p):
            continue
        for r in round_numbers:
            if abs(p - r) < round_tolerance:
                is_round[i] = True
                break
    return is_round


def round_mask(prices: np.ndarray, params: dict) -> np.ndarray:
    """The shared round-number predicate over the operative round set."""
    return _compute_round_mask(
        prices,
        np.asarray(params["suspicious_round_numbers"], dtype=np.float64),
        float(params["round_tolerance"]),
    )


def filter_anomaly(prices: np.ndarray, params: dict,
                   is_round: "np.ndarray | None" = None) -> np.ndarray:
    """Filter 1 — anomaly (repo _detect_anomalies_ultra). Candidate is ultra-low
    OR round; flag if median(neighbours priced ABOVE, in the ±window) / price is
    >= min_normal_price_ratio. `is_round` may be shared across filters; computed
    from params when omitted."""
    if is_round is None:
        is_round = round_mask(prices, params)
    n = len(prices)
    ultra_low = float(params["ultra_low_threshold"])
    ratio_thr = float(params["min_normal_price_ratio"])
    lookback = int(params["lookback"])
    lookforward = int(params["lookforward"])
    flag = np.zeros(n, dtype=bool)
    valid = ~np.isnan(prices)
    is_ultra_low = valid & (prices < ultra_low)
    for i in range(n):
        if not (valid[i] and (is_ultra_low[i] or is_round[i])):
            continue
        cur = float(prices[i])
        lo = max(0, i - lookback)
        hi = min(n, i + lookforward + 1)
        above = [float(prices[j]) for j in range(lo, hi)
                 if j != i and valid[j] and prices[j] > cur]
        if not above:
            continue
        med = float(np.median(above))
        if med / (cur + 1e-10) >= ratio_thr:
            flag[i] = True
    return flag


def filter_spike(prices: np.ndarray, params: dict,
                 is_round: "np.ndarray | None" = None) -> np.ndarray:
    """Filter 2 — spike (repo _detect_spikes_ultra). Candidate has price above a
    raw LEVEL gate (high_spike_threshold) OR is a round print > 0.50; flag if
    price / median(pre-window points BELOW) >= min_spike_ratio AND recovers to
    <= median_pre * recovery_ratio within the lookahead."""
    if is_round is None:
        is_round = round_mask(prices, params)
    n = len(prices)
    high_thr = float(params["high_spike_threshold"])
    ratio_thr = float(params["min_spike_ratio"])
    recovery = float(params["recovery_ratio"])
    lookback = int(params["lookback"])
    lookforward = int(params["lookforward"])
    flag = np.zeros(n, dtype=bool)
    valid = ~np.isnan(prices)
    is_high = valid & (prices > high_thr)
    # For spikes the round set is restricted to prints > 0.50 (repo l.1120).
    is_round_spike = is_round & valid & (prices > 0.50)
    for i in range(n):
        if not (valid[i] and (is_high[i] or is_round_spike[i])):
            continue
        cur = float(prices[i])
        lo = max(0, i - lookback)
        below = [float(prices[j]) for j in range(lo, i)
                 if valid[j] and prices[j] < cur]
        if not below:
            continue
        med = float(np.median(below))
        if cur / (med + 1e-10) < ratio_thr:
            continue
        rec_thr = med * recovery
        hi = min(n, i + lookforward + 1)
        if any(valid[j] and prices[j] <= rec_thr for j in range(i + 1, hi)):
            flag[i] = True
    return flag


def filter_plateau(prices: np.ndarray, params: dict,
                   is_round: "np.ndarray | None" = None) -> np.ndarray:
    """Filter 3 — plateau (repo _detect_plateaus_ultra). Run of >= min_plateau_days
    EXACT-equal ultra-low/round prices; flag the run if it is round OR either-side
    displacement (pre/price or post/price) >= pre_post_price_ratio."""
    if is_round is None:
        is_round = round_mask(prices, params)
    n = len(prices)
    ultra_low = float(params["plateau_ultra_low_threshold"])
    min_days = int(params["min_plateau_days"])
    disp = float(params["pre_post_price_ratio"])
    flag = np.zeros(n, dtype=bool)
    valid = ~np.isnan(prices)
    is_ultra_low = valid & (prices < ultra_low)
    i = 0
    while i < n:
        if not (valid[i] and (is_ultra_low[i] or is_round[i])):
            i += 1
            continue
        cur = float(prices[i])
        j = i + 1
        while j < n and prices[j] == cur:  # exact equality (post round to 4 dp)
            j += 1
        if (j - i) >= min_days:
            pre_price = float(prices[i - 1]) if i > 0 and valid[i - 1] else -1.0
            post_price = float(prices[j]) if j < n and valid[j] else -1.0
            suspicious = bool(is_round[i])
            if pre_price > 0 and pre_price / (cur + 1e-10) >= disp:
                suspicious = True
            if post_price > 0 and post_price / (cur + 1e-10) >= disp:
                suspicious = True
            if suspicious:
                flag[i:j] = True
        i = j
    return flag


def filter_intraday(min_price: np.ndarray, max_price: np.ndarray,
                    params: dict) -> np.ndarray:
    """Filter 4 — intraday inconsistency (repo flag_intraday_inconsistency_
    vectorized). Low-price day (min_price < intraday_price_threshold) whose
    high/low range normalised by mean(low,high) exceeds intraday_range_threshold.
    Requires both daily low and high present."""
    n = len(min_price)
    tau = float(params["intraday_price_threshold"])
    gamma = float(params["intraday_range_threshold"])
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = min_price[i]
        hi = max_price[i]
        if np.isnan(lo) or np.isnan(hi):
            continue
        # Candidate if either endpoint is below the low-price gate (== lo < tau
        # since lo <= hi).
        if not (lo < tau or hi < tau):
            continue
        mean = (float(lo) + float(hi)) / 2.0
        if mean > 0 and (float(hi) - float(lo)) / mean > gamma:
            flag[i] = True
    return flag


def apply_filters_to_cusip(cusip_df: pl.DataFrame, params: dict):
    """Apply all four filters to one cusip's time-sorted daily series.

    Returns (keep_mask, flag_dict). A day is dropped if ANY filter fires (OR).
    Per-cusip minimum-observation guards match the repo (anomaly/spike need >= 3
    days; plateau needs >= min_plateau_days)."""
    prices = np.round(cusip_df["price_vwap"].to_numpy().astype(np.float64),
                      PRICE_ROUND_DECIMALS)
    mins = np.round(cusip_df["min_price"].to_numpy().astype(np.float64),
                    PRICE_ROUND_DECIMALS)
    maxs = np.round(cusip_df["max_price"].to_numpy().astype(np.float64),
                    PRICE_ROUND_DECIMALS)
    n = len(prices)

    is_round = round_mask(prices, params)

    min_days = int(params["min_plateau_days"])
    if n >= 3:
        f1 = filter_anomaly(prices, params, is_round=is_round)
        f2 = filter_spike(prices, params, is_round=is_round)
    else:
        f1 = np.zeros(n, dtype=bool)
        f2 = np.zeros(n, dtype=bool)
    f3 = filter_plateau(prices, params, is_round=is_round) if n >= min_days \
        else np.zeros(n, dtype=bool)
    f4 = filter_intraday(mins, maxs, params)  # row-wise; no min-obs gate

    any_flag = f1 | f2 | f3 | f4
    keep_mask = ~any_flag
    flag_dict = {
        "flagged_anomaly":  f1,
        "flagged_spike":    f2,
        "flagged_plateau":  f3,
        "flagged_intraday": f4,
    }
    return keep_mask, flag_dict


# ---------------------------------------------------------------------------
# Partition orchestration
# ---------------------------------------------------------------------------

def process_partition(input_path: Path, output_path: Path, dropped_path: Path,
                      params: dict) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_path.parent.mkdir(parents=True, exist_ok=True)
    out_tmp = output_path.with_suffix(".parquet.tmp")
    drop_tmp = dropped_path.with_suffix(".parquet.tmp")
    for s in (out_tmp, drop_tmp):
        if s.exists():
            s.unlink()

    print(f"  Loading {input_path.name} ...")
    df = pl.read_parquet(str(input_path)).sort(["cusip_id", "trd_exctn_dt"])
    input_rows = df.height
    print(f"  {input_rows:,} cusip-day rows; iterating per cusip...")

    kept_frames: list = []
    dropped_frames: list = []
    counts = {
        "input_rows": input_rows,
        "kept_rows": 0,
        "dropped_total": 0,
        "dropped_anomaly": 0,
        "dropped_spike": 0,
        "dropped_plateau": 0,
        "dropped_intraday": 0,
    }

    # Iterate cusip blocks. df is sorted so contiguous runs of cusip_id work.
    cusip_col = df["cusip_id"].to_numpy()
    n = len(cusip_col)
    block_starts = [0]
    for i in range(1, n):
        if cusip_col[i] != cusip_col[i - 1]:
            block_starts.append(i)
    block_ends = block_starts[1:] + [n]

    for s, e in tqdm(list(zip(block_starts, block_ends)),
                     desc=output_path.parent.name, unit="cusip"):
        sub = df[s:e]
        keep_mask, flags = apply_filters_to_cusip(sub, params)

        # Tally per-filter drop counts BEFORE OR-ing into the keep mask.
        counts["dropped_anomaly"]  += int(flags["flagged_anomaly"].sum())
        counts["dropped_spike"]    += int(flags["flagged_spike"].sum())
        counts["dropped_plateau"]  += int(flags["flagged_plateau"].sum())
        counts["dropped_intraday"] += int(flags["flagged_intraday"].sum())

        if keep_mask.any():
            kept_frames.append(sub.filter(pl.Series(keep_mask)))

        if (~keep_mask).any():
            dropped_idx = ~keep_mask
            dropped_sub = sub.filter(pl.Series(dropped_idx))
            # Attach per-filter flag columns to the dropped artefact
            dropped_sub = dropped_sub.with_columns(
                pl.Series("flagged_anomaly",  flags["flagged_anomaly"][dropped_idx]),
                pl.Series("flagged_spike",    flags["flagged_spike"][dropped_idx]),
                pl.Series("flagged_plateau",  flags["flagged_plateau"][dropped_idx]),
                pl.Series("flagged_intraday", flags["flagged_intraday"][dropped_idx]),
            )
            dropped_frames.append(dropped_sub)

    kept_df = pl.concat(kept_frames) if kept_frames else df[:0]
    counts["kept_rows"] = kept_df.height
    counts["dropped_total"] = input_rows - kept_df.height

    # Write kept
    pq.write_table(kept_df.to_arrow().cast(OUTPUT_SCHEMA), str(out_tmp))
    actual = pq.read_metadata(str(out_tmp)).num_rows
    if actual != counts["kept_rows"]:
        out_tmp.unlink()
        raise AssertionError(
            f"Kept tmp row count {actual:,} != {counts['kept_rows']:,}"
        )
    os.replace(out_tmp, output_path)

    # Write dropped companion (if any)
    if dropped_frames:
        dropped_df = pl.concat(dropped_frames)
        pq.write_table(dropped_df.to_arrow().cast(DROPPED_SCHEMA), str(drop_tmp))
        actual = pq.read_metadata(str(drop_tmp)).num_rows
        if actual != dropped_df.height:
            drop_tmp.unlink()
            raise AssertionError(
                f"Dropped tmp row count {actual:,} != {dropped_df.height:,}"
            )
        os.replace(drop_tmp, dropped_path)

    return counts


# ---------------------------------------------------------------------------
# Report + entry point
# ---------------------------------------------------------------------------

def write_report(dev_counts: dict, hold_counts: "dict | None",
                 params: dict) -> None:
    """hold_counts is None during development — no holdout statistic is
    computed or surfaced into this (development-side) report."""
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "thresholds_sha256": thresholds_sha256(),
        "thresholds_used": params,
        "stage": "distressed_filters_corrected_branch",
        "registry_role": (
            "meas_err = ON, stage 3: DRR-2026 daily distressed filters "
            "(anomaly, spike, plateau, intraday; round-number mask a shared "
            "predicate). Raw family bypasses entirely per A1.1; corrected "
            "family processes per A7. Filter implementation is repo-derived "
            "from Dickerson's released trace-data-pipeline (spec v4 Part J: port + "
            "disclose; the OSBAP/DRR published-output validation firewall is "
            "unaffected). See docs/data/registers/citations_verified.md §1."
        ),
        "source_reference": "trace-data-pipeline (released; spec v4 Part J)",
        "holdout_processed": hold_counts is not None,
        "rows_dev": dev_counts,
        "inputs": {
            "dev": str(DEV_IN.relative_to(REPO_ROOT)),
        },
        "outputs": {
            "dev_kept": str(DEV_OUT.relative_to(REPO_ROOT)),
            "dev_dropped": str(DEV_DROPPED.relative_to(REPO_ROOT)),
        },
        "audit_trail_note": (
            "Companion 'distressed_dropped' parquets persist the cusip-day "
            "rows the filters dropped, with one boolean column per filter "
            "recording which fired. Per-filter empirical counts (rows_dev) are "
            "a diagnostic, NOT a gate — a zero count is a legitimate result "
            "(spec v4 C6)."
        ),
    }
    if hold_counts is not None:
        report["rows_holdout"] = hold_counts
        report["inputs"]["holdout"] = str(HOLD_IN.relative_to(REPO_ROOT))
        report["outputs"]["holdout_kept"] = str(HOLD_OUT.relative_to(REPO_ROOT))
        report["outputs"]["holdout_dropped"] = str(HOLD_DROPPED.relative_to(REPO_ROOT))
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


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
        print("Run scripts/build_daily_panel.py --family corr first.", file=sys.stderr)
        sys.exit(1)

    params = load_config()
    print(f"Loaded meas_err_distressed_filters config "
          f"(thresholds sha256: {thresholds_sha256()[:12]}...)")
    print("Stage: meas_err = ON — DRR-2026 distressed filters (trace-data-pipeline)")

    print("Processing development partition...")
    dev_counts = process_partition(DEV_IN, DEV_OUT, DEV_DROPPED, params)
    print(
        f"  dev: in={dev_counts['input_rows']:,} "
        f"kept={dev_counts['kept_rows']:,} "
        f"dropped={dev_counts['dropped_total']:,} "
        f"(anom={dev_counts['dropped_anomaly']:,} "
        f"spike={dev_counts['dropped_spike']:,} "
        f"plat={dev_counts['dropped_plateau']:,} "
        f"intr={dev_counts['dropped_intraday']:,})"
    )

    hold_counts = None
    if args.holdout:
        print("Processing holdout partition (post-freeze run)...")
        hold_counts = process_partition(HOLD_IN, HOLD_OUT, HOLD_DROPPED, params)
    else:
        print("Holdout partition NOT processed (development mode; "
              "pass --holdout for the post-freeze run).")

    write_report(dev_counts, hold_counts, params)

    print("\nDone.")
    print(f"  Dev: {dev_counts['kept_rows']:,} rows → {DEV_OUT}")
    if hold_counts is not None:
        print(f"  Hold: {hold_counts['kept_rows']:,} rows → {HOLD_OUT}")


if __name__ == "__main__":
    main()
