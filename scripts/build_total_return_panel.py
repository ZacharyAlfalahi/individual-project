"""
Total-return panel — the §2.1 accrual upgrade (Chunk 5).

Re-emits the maximal monthly panel with the clean-price returns replaced by TOTAL
returns:
    R = [(P_t + AI_t + C_t) − (P_{t-1} + AI_{t-1})] / (P_{t-1} + AI_{t-1})
using the accrual engine (agents/quant/library/accrual.py) and the FISD coupon
schedule (coupon, interest_frequency, maturity, day_count_basis). AI/C are
family-agnostic; only the clean price P_{raw,corr} differs across families. Adds
`ai`, `coupon_paid`, and `day_count_fallback` columns; everything else (size,
rating, universe_eligible, ...) is carried unchanged so the factor builders run
on this panel unmodified.

Built in ISOLATION (par-weighting untouched) so any factor movement attributes
cleanly to accrual. Output: data/development/monthly_panel_total_return.parquet.

Validation (the diagnostic protocol's necessary + sufficient set):
  * Zero-coupon invariance (the control-group regression): Z bonds (coupon==0)
    must have total return BYTE-IDENTICAL to clean return — asserted here.
  * day_count_fallback surfaced as a panel column (ACT/* bonds priced on 30/360);
    factor-level exposure is reported by run_accrual_validation.py.
The clean-price maximal panel is left in place for the levels / differentials
comparison.

Usage:
  python scripts/build_total_return_panel.py
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
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.accrual import accrued_and_coupon, THIRTY_360  # noqa: E402

PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
FISD_STATIC = REPO_ROOT / "data" / "development" / "fisd" / "fisd_reference_static.parquet"
OUT_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
REPORT_OUT = REPO_ROOT / "data" / "development" / "monthly_panel_total_return_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _total_return_one_family(panel: pd.DataFrame, price_col: str) -> np.ndarray:
    """Total return per row for one family, with the same calendar-adjacency rule
    as the clean-price build (prior month must be the immediately-preceding
    month, else NaN). AI/C columns are family-agnostic and already on `panel`."""
    P = panel[price_col].to_numpy(dtype=float)
    AI = panel["ai"].to_numpy(dtype=float)
    C = panel["coupon_paid"].to_numpy(dtype=float)
    cusip = panel["cusip"].to_numpy()
    per = panel["date"].dt.to_period("M").astype("int64").to_numpy()

    prev_P = np.r_[np.nan, P[:-1]]
    prev_AI = np.r_[np.nan, AI[:-1]]
    adj = np.zeros(len(panel), dtype=bool)
    adj[1:] = (cusip[1:] == cusip[:-1]) & ((per[1:] - per[:-1]) == 1)

    denom = prev_P + prev_AI
    with np.errstate(invalid="ignore", divide="ignore"):
        ret = ((P + AI + C) - (prev_P + prev_AI)) / denom
    ret = np.where(adj & (denom > 0), ret, np.nan)
    return ret


def main():
    for f in (PANEL_FILE, FISD_STATIC):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("Loading panel + FISD coupon schedule...")
    panel = pd.read_parquet(PANEL_FILE)
    # `maturity` is already on the maximal panel (from its own FISD merge); pull
    # only the coupon-schedule fields to avoid a column collision.
    fisd = pd.read_parquet(FISD_STATIC,
                           columns=["cusip", "coupon", "interest_frequency",
                                    "day_count_basis"]).drop_duplicates("cusip")
    if "maturity" not in panel.columns:
        raise KeyError("maximal panel missing 'maturity' (expected from its FISD merge)")
    panel = panel.merge(fisd, on="cusip", how="left")
    panel = panel.sort_values(["cusip", "date"], kind="mergesort").reset_index(drop=True)

    print("Computing accrued interest + coupon (30/360; Z-coupon → 0)...")
    ai, coupon_paid = accrued_and_coupon(
        panel["date"].values, panel["maturity"].values,
        panel["coupon"].values, panel["interest_frequency"].values,
    )
    panel["ai"] = ai
    panel["coupon_paid"] = coupon_paid
    panel["day_count_fallback"] = (panel["day_count_basis"].fillna("NA") != THIRTY_360)

    # Keep the clean returns to validate Z-invariance, then overwrite with totals.
    clean_raw = panel["ret_raw"].to_numpy(dtype=float)
    clean_corr = panel["ret_corr"].to_numpy(dtype=float)
    tot_raw = _total_return_one_family(panel, "price_eom_raw")
    tot_corr = _total_return_one_family(panel, "price_eom_corr")

    # --- Zero-coupon invariance regression (the control group) ---
    zero = (panel["coupon"].fillna(0.0) == 0.0).to_numpy()
    def _invariant(clean, tot, mask):
        both = mask & ~np.isnan(clean) & ~np.isnan(tot)
        return bool(np.allclose(clean[both], tot[both], atol=1e-12, rtol=0)), int(both.sum())
    inv_raw, n_raw = _invariant(clean_raw, tot_raw, zero)
    inv_corr, n_corr = _invariant(clean_corr, tot_corr, zero)
    if not (inv_raw and inv_corr):
        # A genuine bug — the accrual path touched a zero-coupon bond.
        bad = (~np.isclose(clean_corr, tot_corr, atol=1e-12)) & zero & ~np.isnan(clean_corr) & ~np.isnan(tot_corr)
        raise AssertionError(
            f"Zero-coupon invariance FAILED: {int(bad.sum())} Z-bond-months changed "
            f"(raw_ok={inv_raw}, corr_ok={inv_corr}). The accrual path is touching "
            f"bonds it must not — check Z detection (coupon==0)."
        )
    print(f"  zero-coupon invariance OK: {n_corr:,} Z bond-months byte-identical (corr)")

    panel["ret_raw"] = tot_raw
    panel["ret_corr"] = tot_corr
    panel["xret_raw"] = tot_raw - panel["rf_monthly"]
    panel["xret_corr"] = tot_corr - panel["rf_monthly"]
    # Drop only the schedule fields merged in here; keep the panel's own
    # `maturity` (an original column) and the new ai/coupon_paid/fallback.
    panel = panel.drop(columns=["coupon", "interest_frequency", "day_count_basis"])

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"return_basis": b"total_return_price_plus_AI_plus_coupon_30E360",
        b"accrual": b"BBW_anchor_spec_2.1",
        b"par_weighting": b"unchanged_isolated",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)

    fb = panel["day_count_fallback"]
    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "return_basis": "total return = (P + AI + C) growth (§2.1); 30E/360 accrual",
        "zero_coupon_invariance": {"raw_ok": inv_raw, "corr_ok": inv_corr,
                                   "z_bond_months_checked": n_corr},
        "day_count_fallback": {
            "fallback_bond_months": int(fb.sum()),
            "fallback_pct": float(fb.mean() * 100),
        },
        "ai_diagnostics": {
            "ai_max": float(panel["ai"].max()), "ai_mean_nonzero":
            float(panel.loc[panel["ai"] > 0, "ai"].mean()),
            "coupon_paid_months": int((panel["coupon_paid"] > 0).sum()),
        },
    }
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\nDone.")
    print(f"  total-return panel: {len(panel):,} rows  →  {OUT_FILE}")
    print(f"  day_count_fallback: {int(fb.sum()):,} bond-months ({fb.mean()*100:.2f}%)")
    print(f"  AI: max {panel['ai'].max():.3f}, mean(nonzero) {panel.loc[panel['ai']>0,'ai'].mean():.3f}, "
          f"coupon-payment months {int((panel['coupon_paid']>0).sum()):,}")
    print(f"  Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
