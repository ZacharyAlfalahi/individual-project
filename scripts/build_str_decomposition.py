"""
str LIB decomposition — the §8 headline gate (BBW_anchor_implementation_spec.md
§5.1, §7 toggle 3, §8). Reproduces DRR-2026 Table 2 Panel A's month-end vs
month-begin reversal decomposition from intra-month daily prices.

Mechanism. Reversal sorts on the prior-month return and holds one month. The
latent-illiquidity / correlated-errors bias (LIB / CEIV) comes from the holding
return REUSING the price that the sorting signal ends on:

  signal[m-1] = price_end[m-1] / price_end[m-2] − 1     (the sort, identical in both arms)
  month-end hold[m]   = price_end[m]  / price_end[m-1]  − 1   (starts at price_end[m-1] = signal end → SHARED → biased)
  month-begin hold[m] = price_end[m]  / price_begin[m]  − 1   (starts at the first-5-bd price of month m, days AFTER
                                                               the signal endpoint → NOT shared → debiased)

Both arms hold the SAME portfolios (so the two return series are ~0.99
correlated, per DRR); only the holding-return price differs. The gap between the
two premiums is the LIB component. Spec targets (corr, single-sort, VW):
month-end ≈ −0.99%/mo, month-begin ≈ −0.17%/mo, gap ≈ −0.82 (~83% of premium),
corr ≈ 0.99. The gate is DIRECTION + PROPORTION, not absolute level (§8).

CONFIRM-ON-READ (§9/§11): the exact DRR window length and begin/end price
definition come from the DRR-2026 GitHub code; this uses the spec's literal
"first/last 5 business days of the following month" reading. window_days is
sourced from thresholds.yaml (signals.str_lib.window_days).

Headline = corr family (LIB lives in the cleaned series; meas_err corrections do
not remove bid-ask/illiquidity noise). Leg = losers − winners, VW (par), to match
the str factor (§5.1).

Output: data/development/headlines/str_decomposition.json

Usage:
  python scripts/build_str_decomposition.py
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.characteristic_sort import (  # noqa: E402
    run_characteristic_sort, summarize_returns,
)
from agents.quant.library.intramonth_prices import month_window_prices  # noqa: E402

CORR_DAILY = REPO_ROOT / "data" / "development" / "trace_daily_corr_filtered.parquet"
PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "headlines"
OUT_FILE = OUT_DIR / "str_decomposition.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_window_days() -> int:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("str_lib")
    if block is None or "window_days" not in block:
        raise KeyError("thresholds.yaml missing signals.str_lib.window_days")
    return int(block["window_days"])


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """From per-(cusip, month) price_begin/price_end, build:
      r_end[m]   = price_end[m] / price_end[m-1] − 1   (contiguous months only)
      r_begin[m] = price_end[m] / price_begin[m] − 1   (within-month; no lag)
    """
    df = prices.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)
    period = df["date"].dt.to_period("M").astype("int64").to_numpy()
    cusip = df["cusip"].to_numpy()
    prev_end = df["price_end"].shift(1).to_numpy()
    same_prev = np.empty(len(df), dtype=bool)
    same_prev[0] = False
    same_prev[1:] = (cusip[1:] == cusip[:-1]) & ((period[1:] - period[:-1]) == 1)

    pe = df["price_end"].to_numpy(dtype=float)
    pb = df["price_begin"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_end = np.where(same_prev & (prev_end > 0), pe / prev_end - 1.0, np.nan)
        r_begin = np.where(pb > 0, pe / pb - 1.0, np.nan)
    df["r_end"] = r_end
    df["r_begin"] = r_begin
    # Guard against impossible (<= -1) returns from any residual bad price.
    df.loc[df["r_end"] <= -1, "r_end"] = np.nan
    df.loc[df["r_begin"] <= -1, "r_begin"] = np.nan
    return df[["cusip", "date", "r_end", "r_begin"]]


def str_rulebook(score_col: str) -> dict:
    return {
        "score": score_col, "groups": 5, "weighting": "by_size",
        "long_group": 0, "short_group": 4, "signal_lag": 0, "nw_lags": None,
        "min_bonds": 20,
    }


def run_arm(panel: pd.DataFrame, ret_col: str) -> dict:
    """Run str with the SAME signal (score=r_end) but holding return = ret_col."""
    p = panel.rename(columns={ret_col: "ret"})[["cusip", "date", "ret", "size", "score"]]
    res = run_characteristic_sort(p, str_rulebook("score"))
    mr = res["monthly_returns"][["date", "strategy_ret"]]
    return {"monthly": mr, "summary": res["summary"]}


def main():
    for f in (CORR_DAILY, PANEL_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    window_days = load_window_days()
    print(f"Config: window_days={window_days} (first/last business-day windows)")

    print(f"Loading corr daily panel: {CORR_DAILY.name}")
    daily = pd.read_parquet(CORR_DAILY, columns=["cusip_id", "trd_exctn_dt", "price_vwap", "total_vol"])
    print(f"  {len(daily):,} daily rows")

    print("Building begin/end month-window prices...")
    prices = month_window_prices(daily, window_days=window_days)
    print(f"  {len(prices):,} cusip-months with a begin or end price")

    print("Constructing month-end (shared-price) and month-begin (gapped) returns...")
    rets = build_returns(prices)

    # Attach size (par, family-agnostic) and restrict to the eligible universe.
    panel_meta = pd.read_parquet(PANEL_FILE, columns=["cusip", "date", "size", "universe_eligible"])
    panel = rets.merge(panel_meta, on=["cusip", "date"], how="left")
    panel = panel[(panel["universe_eligible"] == True) & panel["size"].notna() & (panel["size"] > 0)]  # noqa: E712
    # The sorting signal is the prior month-end return (same portfolios in both arms).
    panel["score"] = panel["r_end"]

    print("Running str month-end arm (hold = r_end, shares signal endpoint)...")
    end_arm = run_arm(panel, "r_end")
    print("Running str month-begin arm (hold = r_begin, gapped)...")
    begin_arm = run_arm(panel, "r_begin")

    me = pd.Series(end_arm["monthly"]["strategy_ret"].values,
                   index=pd.DatetimeIndex(end_arm["monthly"]["date"]))
    mb = pd.Series(begin_arm["monthly"]["strategy_ret"].values,
                   index=pd.DatetimeIndex(begin_arm["monthly"]["date"]))
    joined = pd.concat([me.rename("end"), mb.rename("begin")], axis=1).dropna()
    gap = (joined["end"] - joined["begin"])
    gap_summary = summarize_returns(gap, nw_lags=None, months_per_year=12)

    mean_end = float(joined["end"].mean())
    mean_begin = float(joined["begin"].mean())
    mean_gap = float(gap.mean())
    proportion = float(mean_gap / mean_end) if mean_end != 0 else None
    correlation = float(joined["end"].corr(joined["begin"]))

    # The §8 gate is direction + proportion. We separate the two because, in
    # this build, they behave very differently:
    #   * DIRECTION (both arms negative, month-end at least as negative, arms
    #     highly correlated) is reproduced robustly.
    #   * PROPORTION (LIB share of the premium) is HIGHLY construction-sensitive:
    #     a window sweep gives 5-bd ≈ 26%, 2-bd ≈ 110%, 1-bd ≈ -23% — none near
    #     DRR's 83%. The exact begin/end price definition is CONFIRM-ON-READ
    #     against the DRR-2026 GitHub code (§9/§11), so we report the proportion
    #     as a diagnostic, NOT a hard gate, until that construction is confirmed.
    direction_reproduced = (
        mean_end < 0 and mean_begin < 0
        and abs(mean_end) >= abs(mean_begin)
        and correlation >= 0.8
    )
    proportion_in_drr_band = proportion is not None and 0.7 <= proportion <= 1.0

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "family": "corr",
        "window_days": window_days,
        "weighting": "value-weight by par offering_amt (§5.1); leg = losers - winners",
        "n_months": int(len(joined)),
        "month_end": {
            "mean_pct_per_month": mean_end * 100,
            "t_stat": float(end_arm["summary"]["t_stat"]),
            "target_pct": -0.99,
        },
        "month_begin": {
            "mean_pct_per_month": mean_begin * 100,
            "t_stat": float(begin_arm["summary"]["t_stat"]),
            "target_pct": -0.17,
        },
        "gap_lib": {
            "mean_pct_per_month": mean_gap * 100,
            "t_stat": float(gap_summary["t_stat"]),
            "proportion_of_month_end": proportion,
            "target_pct": -0.82,
            "target_proportion": 0.83,
        },
        "arm_correlation": correlation,
        "gate": {
            "criterion": "direction + proportion (§8), NOT absolute level",
            "direction_reproduced": bool(direction_reproduced),
            "proportion_in_drr_band": bool(proportion_in_drr_band),
            "status": "PARTIAL — direction reproduced; LIB proportion is "
                      "construction-sensitive and below DRR's 83% under the "
                      "literal 5-bd reading. BLOCKED on CONFIRM-ON-READ (§9/§11).",
        },
        "construction_note": "Same prior-month-end signal in both arms; month-end "
            "holding return reuses price_end[m-1] (shared → biased), month-begin "
            "uses price_end[m]/price_begin[m] (within-month, gapped → debiased). "
            "A window sweep (1/2/5 bd) shows the LIB proportion is unstable, so "
            "the exact DRR-2026 price-window construction must be confirmed "
            "against their GitHub code before this gate is asserted. The str "
            "FACTOR itself (build_str.py, corr ≈ -0.70%/mo, t -5.18) is the solid "
            "headline; this decomposition is the open CONFIRM-ON-READ item.",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT_FILE)

    print("\nDecomposition (corr, single-sort, VW, losers-winners):")
    print(f"  month-end   : {mean_end*100:+.3f}%/mo (t {end_arm['summary']['t_stat']:+.2f})  [target ~ -0.99]")
    print(f"  month-begin : {mean_begin*100:+.3f}%/mo (t {begin_arm['summary']['t_stat']:+.2f})  [target ~ -0.17]")
    print(f"  gap (LIB)   : {mean_gap*100:+.3f}%/mo (t {gap_summary['t_stat']:+.2f}), "
          f"{(proportion*100 if proportion else float('nan')):.0f}% of month-end  [target ~ 83%]")
    print(f"  arm corr    : {correlation:.3f}  [target ~ 0.99];  n={len(joined)} months")
    print(f"  direction reproduced : {'YES' if direction_reproduced else 'NO'}  "
          f"(both negative, month-end >= month-begin, corr >= 0.8)")
    print(f"  proportion in DRR band: {'YES' if proportion_in_drr_band else 'NO'}  "
          f"(construction-sensitive — CONFIRM-ON-READ vs DRR-2026 code, §9/§11)")
    print("  STATUS: PARTIAL — str factor (build_str.py) is the solid headline; "
          "the exact decomposition construction is the open CONFIRM-ON-READ item.")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
