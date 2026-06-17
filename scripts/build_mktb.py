"""
Build the BBW MKTB market factor: the value-weighted (par) average EXCESS
return of all eligible bonds each month (spec BBW_anchor_implementation_spec.md
§3.6). No sort — this is the market basket the four BBW long-short factors price
against, and the simplest end-to-end check of the VW-excess plumbing before any
sort exists (build order §10 step 3).

Family-indexed per A9: emits `mktb_raw` (from `xret_raw`) and `mktb_corr` (from
`xret_corr`) as parallel columns; cross-family mixing is forbidden.

Weighting basis: par `offering_amt`, surfaced as the panel `size` column (§2.4),
NOT market value — a deliberate, documented divergence from OSBAP `mcap_e`.

Excess-return basis: the panel's `xret_* = ret_* − rf_monthly`, where rf is FRED
TB3MS (3-month T-bill). Spec §2.2 names the 1-month T-bill; the panel ships a
3-month-bill excess return, so MKTB inherits that pre-existing divergence. It is
recorded here (and matters for MKTB's level, though it cancels in every
long-short factor). Switching to a 1-month bill is a panel-layer change, out of
scope for this builder.

Output: data/development/factors/mktb.parquet with columns
  date, mktb_raw, mktb_corr, n_bonds_raw, n_bonds_corr.

Usage:
  python scripts/build_mktb.py

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.market_factor import compute_market_factor  # noqa: E402

# Anchor-layer headline uses the §2.1 TOTAL-return panel when present (the
# clean-price levels are wrong for the credit/liquidity legs); falls back to the
# clean maximal panel, and is overridable via BBW_ANCHOR_PANEL for comparison.
_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "mktb.parquet"
REPORT_OUT = OUT_DIR / "mktb_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def compute_dual_family(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute MKTB on each family and join on date.

    Par value-weighted (size) average excess return of universe-eligible bonds.
    """
    raw = compute_market_factor(panel, ret_col="xret_raw").rename(
        columns={"mktb": "mktb_raw", "n_bonds": "n_bonds_raw"}
    )
    corr = compute_market_factor(panel, ret_col="xret_corr").rename(
        columns={"mktb": "mktb_corr", "n_bonds": "n_bonds_corr"}
    )
    return raw.merge(corr, on="date", how="outer").sort_values("date").reset_index(drop=True)


def write_factor(factor: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"mktb",
        b"factor_source": b"BBW_2019",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"value_weight_par_offering_amt",
        b"return_basis": b"excess_over_TB3MS",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(factor: pd.DataFrame) -> None:
    def _stats(col: str) -> dict:
        s = factor[col].dropna()
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "max_abs_pct_per_month": float(s.abs().max() * 100) if len(s) else None,
            "first_date": str(factor.loc[factor[col].notna(), "date"].min().date())
            if len(s) else None,
            "last_date": str(factor.loc[factor[col].notna(), "date"].max().date())
            if len(s) else None,
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "weighting": "value-weight by par offering_amt (panel `size`); not mcap_e (§2.4)",
        "return_basis": "excess over rf = TB3MS/12 (3-month bill); spec §2.2 names "
                        "the 1-month bill — divergence inherited from the panel, "
                        "cancels in long-short factors but affects MKTB level",
        "eligibility": "universe_eligible == True, finite xret, size > 0",
        "headline_series": "mktb_corr",
        "notes": (
            "HEADLINE = mktb_corr. mktb_raw is the meas_err=OFF (uncorrected) "
            "counterpart and is DOMINATED BY OUTLIERS: the raw family retains "
            "decimal-slip / par-snap price errors (xret_raw reaches ~10^6) that "
            "the decimal-shift + bounce-back + distressed filters remove in the "
            "corr family. A long-only value-weighted mean does not difference "
            "these out (unlike a long-short factor), so mktb_raw's level is not a "
            "usable market return — it is kept only for A9 family completeness and "
            "is itself a demonstration of why meas_err correction matters. "
            "mktb_corr's level is low vs DRR-2023 (~0.47%/mo, 2004:08–2016:12) "
            "because returns are clean-price-only (no accrued interest / coupon, "
            "spec §2.1) — the documented level offset that the §8 differential "
            "gates are designed to tolerate."
        ),
        "mktb_raw": _stats("mktb_raw"),
        "mktb_corr": _stats("mktb_corr"),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    if not PANEL_FILE.exists():
        print(f"ERROR: required input not found: {PANEL_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading panel: {PANEL_FILE}")
    panel = pd.read_parquet(
        PANEL_FILE,
        columns=["date", "size", "universe_eligible", "xret_raw", "xret_corr"],
    )
    print(f"  {len(panel):,} rows")

    print("Computing MKTB (par value-weighted excess return) per family...")
    factor = compute_dual_family(panel)

    write_factor(factor)
    write_report(factor)

    raw = factor["mktb_raw"].dropna()
    corr = factor["mktb_corr"].dropna()
    print("\nDone.")
    print(f"  {len(factor):,} months")
    if len(corr):
        print(f"  mktb_corr (HEADLINE): mean {corr.mean()*100:.3f}%/mo, "
              f"sd {corr.std(ddof=1)*100:.3f}%, {len(corr)} months")
    if len(raw):
        print(f"  mktb_raw  (meas_err=OFF): mean {raw.mean()*100:.1f}%/mo "
              f"— outlier-dominated by uncorrected price errors; not a usable "
              f"level (see report notes)")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
