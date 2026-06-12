"""
DRR (Dickerson, Robotti, Rossetti 2026) Appendix A.3 distressed daily
filters 1-4 — corrected branch, stage 3 of the meas_err pipeline.

Reads `trace_daily_corr.parquet` (output of build_daily_panel.py --family
corr), applies four per-cusip filters on the time-sorted daily series, and
writes `trace_daily_corr_filtered.parquet` with dropped days removed. Per
A7.6 of the registry amendments, the dropped days are persisted to
`distressed_dropped_<partition>.parquet` companion artefacts with one boolean
flag column per filter so the audit trail records which filter fired.

Stage order (normative per A7.2): decimal-shift → bounce-back → VWAP→daily
→ distressed filters 1-4. This script is stage 3.

**Independent coding per A7.3.** This implementation transcribes the four
filters from DRR-2026 Appendix A.3's published description. It is NOT a
port of any DRR distribution code. Where the description is ambiguous,
interpretation is documented inline; the parameter set comes from Table A.3
via docs/thresholds.yaml.

Filter definitions (paraphrased from A7.1 + A7.3 of the amendments):

  1. Anomaly       — isolated ultra-low prints sitting ≥ρ_anomaly price
                     points below the surrounding ±L-day median, AND the
                     print itself is ≤ τ_low (ultra-low).
  2. Spike         — prints ≥ρ_spike above the pre-spike-L-day median that
                     recover within the next L days to within ρ_recovery
                     of that median.
  3. Plateau       — runs of ≥ℓ_min consecutive days with identical prices
                     (within τ_plateau) that are EITHER ultra-low (≤τ_low)
                     OR near a multiple of 25 (round-numbered), with
                     pre/post displacement of ≥ρ_plateau on both sides.
  4. Intraday      — days where min_price < τ_intraday AND
                     (max_price − min_price) > γ_range × price_vwap.

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
    "rho_anomaly", "L",
    "rho_spike", "rho_recovery",
    "rho_plateau", "ell_min", "tau_plateau", "round_step",
    "tau_intraday", "gamma_range",
    "tau_low",
)


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
# ---------------------------------------------------------------------------

def _is_near_round(price: float, tau: float, step: float) -> bool:
    """A price is "near round" if within tau of an integer multiple of step.
    step comes from thresholds.yaml round_step (25 covers 25/50/75/100/...,
    the conventional bond round-number plateaus)."""
    nearest = round(price / step) * step
    return abs(price - nearest) <= tau


def filter_anomaly(prices: np.ndarray, params: dict) -> np.ndarray:
    """Filter 1: isolated ultra-low prints ≥ρ_anomaly below ±L-day median.
    Returns a boolean array; True where the day is flagged."""
    n = len(prices)
    rho = float(params["rho_anomaly"])
    L = int(params["L"])
    tau_low = float(params["tau_low"])
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if prices[i] > tau_low:
            continue  # not ultra-low → not a Filter-1 candidate
        # ±L-day window EXCLUDING the current day
        lo = max(0, i - L)
        hi = min(n, i + L + 1)
        nbr_idx = [j for j in range(lo, hi) if j != i]
        if len(nbr_idx) < 1:
            continue
        nbr_med = float(np.median(prices[nbr_idx]))
        if nbr_med - float(prices[i]) >= rho:
            flag[i] = True
    return flag


def filter_spike(prices: np.ndarray, params: dict) -> np.ndarray:
    """Filter 2: prints ≥ρ_spike above the trailing-L median that recover
    within L days to within ρ_recovery of that median."""
    n = len(prices)
    rho = float(params["rho_spike"])
    rec = float(params["rho_recovery"])
    L = int(params["L"])
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == 0:
            continue
        pre_lo = max(0, i - L)
        pre_window = prices[pre_lo:i]
        if len(pre_window) == 0:
            continue
        pre_med = float(np.median(pre_window))
        if float(prices[i]) - pre_med < rho:
            continue
        # Recovery check: any of the next L days back to within rec of pre_med
        post_hi = min(n, i + 1 + L)
        post_window = prices[i + 1:post_hi]
        if any(abs(float(p) - pre_med) <= rec for p in post_window):
            flag[i] = True
    return flag


def filter_plateau(prices: np.ndarray, params: dict) -> np.ndarray:
    """Filter 3: runs of ≥ℓ_min identical-or-near-round prices with pre/post
    displacement ≥ρ_plateau. Plateau prices must be ultra-low OR near round."""
    n = len(prices)
    ell_min = int(params["ell_min"])
    rho = float(params["rho_plateau"])
    tau = float(params["tau_plateau"])
    tau_low = float(params["tau_low"])
    round_step = float(params["round_step"])
    flag = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        # Extend a run as long as the next price is within tau of the run's
        # first price.
        j = i
        while j + 1 < n and abs(float(prices[j + 1]) - float(prices[i])) <= tau:
            j += 1
        run_len = j - i + 1
        if run_len >= ell_min:
            p_run = float(prices[i])
            # Plateau prices must be ultra-low or near round.
            is_qualifying = (p_run <= tau_low) or _is_near_round(p_run, tau, round_step)
            pre = float(prices[i - 1]) if i > 0 else None
            post = float(prices[j + 1]) if j + 1 < n else None
            pre_disp = (pre is not None) and (abs(pre - p_run) >= rho)
            post_disp = (post is not None) and (abs(post - p_run) >= rho)
            # Both pre and post displaced — interpretation of "≥ρ_plateau
            # pre/post displacement" as both-sides displaced (a one-sided
            # displacement at the start or end of the series doesn't
            # confirm a plateau).
            if is_qualifying and pre_disp and post_disp:
                flag[i:j + 1] = True
        i = j + 1

    return flag


def filter_intraday(price_vwap: np.ndarray, min_price: np.ndarray,
                    max_price: np.ndarray, params: dict) -> np.ndarray:
    """Filter 4: low-price days with abnormal intraday range.
    min_price < τ_intraday AND (max - min) > γ_range × VWAP."""
    n = len(price_vwap)
    tau = float(params["tau_intraday"])
    gamma = float(params["gamma_range"])
    flag = np.zeros(n, dtype=bool)
    for i in range(n):
        if (float(min_price[i]) < tau
                and (float(max_price[i]) - float(min_price[i])) > gamma * float(price_vwap[i])):
            flag[i] = True
    return flag


def apply_filters_to_cusip(cusip_df: pl.DataFrame, params: dict):
    """Apply all four filters to one cusip's time-sorted daily series.

    Returns (keep_mask, flag_dict). flag_dict has per-filter boolean arrays.
    A day is dropped if ANY filter fires (logical OR)."""
    prices = cusip_df["price_vwap"].to_numpy()
    mins = cusip_df["min_price"].to_numpy()
    maxs = cusip_df["max_price"].to_numpy()
    f1 = filter_anomaly(prices, params)
    f2 = filter_spike(prices, params)
    f3 = filter_plateau(prices, params)
    f4 = filter_intraday(prices, mins, maxs, params)
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
            "meas_err = ON, stage 3: DRR-2026 Appendix A.3 daily distressed "
            "filters 1-4 (anomaly, spike, plateau, intraday). Raw family "
            "bypasses entirely per A1.1; corrected family processes per A7. "
            "Filters are independent codings from the published appendix per "
            "A7.3 (no DRR distribution code ported)."
        ),
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
            "recording which fired. Satisfies the Phase 1 verification "
            "step 4 (spot-check a known distressed-flagged day)."
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
    print("Stage: meas_err = ON — DRR-2026 distressed filters 1-4")

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
