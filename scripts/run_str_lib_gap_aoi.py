"""
str × lib_gap AOI headline demo — Phase 1's pre-FISD validation gate.

The first end-to-end consumer of `monthly_panel_maximal.parquet`. Runs
short-term reversal on the corrected family twice — `signal_lag = 0` (the
"as-published" / lib_gap=OFF configuration) and `signal_lag = 1` (the
corrected / lib_gap=ON configuration with the one-month gap inserted) —
and reports the paired-difference Newey-West t-stat. Also runs a bonus
str raw-vs-corrected differential at gap=0 — the real-data check-2 proxy.

Both are PATTERN gates, not tolerance gates, while O4 (EW DRR target)
remains open. The pass criteria:

  PRIMARY (lib_gap AOI on corr family):
    - |str_premium(gap=0)| > |str_premium(gap=1)|  (uncorrected larger)
    - sign(str_premium(gap=0)) is consistent with the reversal prediction
      (positive when long bottom-quintile, short top-quintile)
    - |t-stat(gap=0)| > |t-stat(gap=1)|

  BONUS (raw-vs-corr at gap=0):
    - |str_premium(raw, gap=0)| > |str_premium(corr, gap=0)|  (meas_err proxy)

STR convention: sort on prior-month return (score = ret with the signal_lag
controlling the gap). Long the BOTTOM group (losers), short the TOP group
(winners). Reversal effect predicts positive long-short premium.

Output: data/development/headlines/str_lib_gap_aoi.json

Usage:
  python scripts/run_str_lib_gap_aoi.py
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "agents" / "quant" / "library"))
from characteristic_sort import (  # noqa: E402
    run_characteristic_sort,
    summarize_returns,
)
from run_config import (  # noqa: E402
    RunConfig,
    PanelViewConfig,
    ConstructionConfig,
    EvaluationConfig,
)
from views import view  # noqa: E402


PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "headlines"
OUT_FILE = OUT_DIR / "str_lib_gap_aoi.json"


def _str_view_config(family: str) -> RunConfig:
    """Build a RunConfig that selects the named family with no view-layer
    transformations beyond family selection. signal_lag and expost_trim
    are construction-layer choices the engine handles directly, so we
    keep the view layer minimal here and let the rulebook drive the
    construction-side toggles."""
    return RunConfig(
        panel_view=PanelViewConfig(
            price_family=family,
            stale_mask=False,
            include_terminal_rows=False,
        ),
        # Construction toggles are exercised by varying signal_lag in the
        # rulebook itself (see _str_rulebook). The view layer only owns
        # the panel-level selection here.
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )


def _build_family_panel(maximal: pd.DataFrame, family: str) -> pd.DataFrame:
    """Materialise the engine-shape panel for the STR run via the canonical
    view() interface. STR's score column is `ret` itself (sort on
    prior-month return), so after view() selects the family's `ret_<family>`
    and renames it to `ret`, we add `score = ret` as the final adapter step.
    """
    cfg = _str_view_config(family)
    panel = view(maximal, cfg)  # adds engine-contract columns
    panel = panel.drop_duplicates(subset=["cusip", "date"]).reset_index(drop=True)
    panel["score"] = panel["ret"]
    return panel[["cusip", "date", "ret", "size", "score"]]


def _str_rulebook(signal_lag: int) -> dict:
    """STR: sort on prior-month return (score=ret), long bottom group,
    short top group. Quintile sort, equal weighting (size is a placeholder
    so weighting must be equal per the maximal panel's policy)."""
    return {
        "score": "score",
        "groups": 5,
        "weighting": "equal",
        "long_group": 0,
        "short_group": 4,
        "min_bonds": 20,
        "signal_lag": signal_lag,
        "nw_lags": None,  # let the engine pick floor(4*(T/100)^(2/9))
    }


def _summarize(result: dict, label: str) -> dict:
    mr = result["monthly_returns"]
    summary = result["summary"]
    return {
        "label": label,
        "n_months": int(summary["n_months"]),
        "first_date": str(summary["first_date"].date()) if summary["first_date"] is not None else None,
        "last_date":  str(summary["last_date"].date())  if summary["last_date"]  is not None else None,
        "mean_strategy_ret": float(summary["average"]),
        "annualised": float(summary["annualised_average"]),
        "sd": float(summary["bumpiness"]),
        "sharpe": float(summary["sharpe"]),
        "t_stat": float(summary["t_stat"]),
        "nw_lags_used": int(summary["nw_lags_used"]),
        "avg_bonds_per_month": float(summary.get("avg_bonds_per_month", float("nan"))),
    }


def _paired_difference(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Paired monthly difference (a − b) on the overlapping date set with
    NW t-stat. Used for both the lib_gap AOI marginal (gap0 − gap1) and the
    raw-vs-corr check-2 proxy at gap0."""
    a_s = pd.Series(a["strategy_ret"].values, index=pd.DatetimeIndex(a["date"]))
    b_s = pd.Series(b["strategy_ret"].values, index=pd.DatetimeIndex(b["date"]))
    diff = (a_s - b_s).dropna()
    summary = summarize_returns(diff, nw_lags=None, months_per_year=12)
    return {
        "n_months": int(summary["n_months"]),
        "first_date": str(summary["first_date"].date()) if summary["first_date"] is not None else None,
        "last_date":  str(summary["last_date"].date())  if summary["last_date"]  is not None else None,
        "mean_diff": float(summary["average"]),
        "sd": float(summary["bumpiness"]),
        "t_stat": float(summary["t_stat"]),
        "nw_lags_used": int(summary["nw_lags_used"]),
    }


def main():
    if not PANEL_FILE.exists():
        print(f"ERROR: Maximal panel not found: {PANEL_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {PANEL_FILE}")
    # view() now drives the family selection — load the full maximal-panel
    # column set it expects (price_eom_*, xret_*, n_trades_*, total_vol_*,
    # last_trade_date_*, exit_reason, rf_monthly) in addition to ret_*.
    maximal = pd.read_parquet(PANEL_FILE)
    print(f"  {len(maximal):,} rows, {maximal['cusip'].nunique():,} cusips")

    # ------------------------------------------------------------------
    # Primary differential: lib_gap on CORRECTED family (gap=0 vs gap=1).
    # ------------------------------------------------------------------
    print("\nPrimary — str × lib_gap on corrected family")
    corr_panel = _build_family_panel(maximal, "corr")
    print(f"  corr panel: {len(corr_panel):,} rows, "
          f"{int(corr_panel['ret'].notna().sum()):,} non-NaN ret")
    print("  Running engine: lib_gap=OFF (signal_lag=0) ...")
    res_corr_gap0 = run_characteristic_sort(corr_panel, _str_rulebook(0))
    print(f"    {res_corr_gap0['summary']['n_months']} months")
    print("  Running engine: lib_gap=ON  (signal_lag=1) ...")
    res_corr_gap1 = run_characteristic_sort(corr_panel, _str_rulebook(1))
    print(f"    {res_corr_gap1['summary']['n_months']} months")

    summary_corr_gap0 = _summarize(res_corr_gap0, "corr_gap0")
    summary_corr_gap1 = _summarize(res_corr_gap1, "corr_gap1")
    aoi_diff_lib_gap = _paired_difference(
        res_corr_gap0["monthly_returns"], res_corr_gap1["monthly_returns"]
    )

    # ------------------------------------------------------------------
    # Bonus: str raw-vs-corrected at gap=0 (real-data check-2 proxy).
    # ------------------------------------------------------------------
    print("\nBonus — str raw-vs-corr at gap=0 (check-2 proxy)")
    raw_panel = _build_family_panel(maximal, "raw")
    print(f"  raw panel: {len(raw_panel):,} rows, "
          f"{int(raw_panel['ret'].notna().sum()):,} non-NaN ret")
    print("  Running engine: family=raw, signal_lag=0 ...")
    res_raw_gap0 = run_characteristic_sort(raw_panel, _str_rulebook(0))
    print(f"    {res_raw_gap0['summary']['n_months']} months")

    summary_raw_gap0 = _summarize(res_raw_gap0, "raw_gap0")
    aoi_diff_meas_err_proxy = _paired_difference(
        res_raw_gap0["monthly_returns"], res_corr_gap0["monthly_returns"]
    )

    # ------------------------------------------------------------------
    # Pattern gates
    # ------------------------------------------------------------------
    print("\nPattern gates:")
    p_gap0 = summary_corr_gap0["mean_strategy_ret"]
    p_gap1 = summary_corr_gap1["mean_strategy_ret"]
    t_gap0 = summary_corr_gap0["t_stat"]
    t_gap1 = summary_corr_gap1["t_stat"]
    p_raw0 = summary_raw_gap0["mean_strategy_ret"]
    p_corr0 = summary_corr_gap0["mean_strategy_ret"]

    gate_lib_premium_magnitude = abs(p_gap0) > abs(p_gap1)
    gate_lib_t_magnitude = abs(t_gap0) > abs(t_gap1) if not (math.isnan(t_gap0) or math.isnan(t_gap1)) else None
    gate_lib_sign_consistent = p_gap0 > 0  # STR (long bot, short top) → positive expected
    gate_meas_premium_magnitude = abs(p_raw0) > abs(p_corr0)

    print(f"  lib_gap |premium(gap0)| > |premium(gap1)| : "
          f"{gate_lib_premium_magnitude} ({abs(p_gap0):.6f} vs {abs(p_gap1):.6f})")
    print(f"  lib_gap sign(premium(gap0)) > 0          : "
          f"{gate_lib_sign_consistent} (premium(gap0) = {p_gap0:.6f})")
    if gate_lib_t_magnitude is not None:
        print(f"  lib_gap |t(gap0)| > |t(gap1)|            : "
              f"{gate_lib_t_magnitude} ({abs(t_gap0):.3f} vs {abs(t_gap1):.3f})")
    print(f"  meas_err |premium(raw)| > |premium(corr)| : "
          f"{gate_meas_premium_magnitude} ({abs(p_raw0):.6f} vs {abs(p_corr0):.6f})")

    payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "strategy": "str",
        "convention": (
            "long bottom-quintile (losers), short top-quintile (winners), "
            "quintile sort on prior-month return, equal-weighted "
            "(size is a placeholder in the maximal panel)."
        ),
        "primary_lib_gap_aoi_corr_family": {
            "gap0_uncorrected": summary_corr_gap0,
            "gap1_corrected":   summary_corr_gap1,
            "paired_diff_gap0_minus_gap1": aoi_diff_lib_gap,
        },
        "bonus_meas_err_proxy_at_gap0": {
            "raw":  summary_raw_gap0,
            "corr": summary_corr_gap0,
            "paired_diff_raw_minus_corr": aoi_diff_meas_err_proxy,
        },
        "pattern_gates": {
            "lib_premium_magnitude_gate":   bool(gate_lib_premium_magnitude),
            "lib_premium_sign_gate":        bool(gate_lib_sign_consistent),
            "lib_t_magnitude_gate":         (None if gate_lib_t_magnitude is None
                                              else bool(gate_lib_t_magnitude)),
            "meas_err_premium_magnitude_gate": bool(gate_meas_premium_magnitude),
        },
        "interpretation_note": (
            "Pattern gates only, not tolerance gates — O4 (EW DRR target) "
            "remains open. A gate that fails warrants investigation but is "
            "not by itself a stop-ship."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_FILE.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, OUT_FILE)
    print(f"\nWritten: {OUT_FILE}")


if __name__ == "__main__":
    main()
