"""
Build the standalone short-term-reversal (`str`) anchor factor (spec
BBW_anchor_implementation_spec.md §5.1): a single-sort, value-weighted (par),
monthly-rebalanced, one-month-holding long-short on the prior-month return.

Construction A (the stored convention): leg = LOSERS − WINNERS = low-prior-
return minus high-prior-return. This produces a NEGATIVE raw premium (≈ −0.99%
/mo in DRR-2026), which is what DRR's downloadable data series stores; their
FIGURES show +0.99 (sign-corrected for display). The data file is authoritative,
so we build and store the negative series and make every downstream comparison
MAGNITUDE-based, never signed-level (§5.1) — this immunises the RQ3 verdict
against the figure-vs-datafile sign inversion.

This builder emits the AS-PUBLISHED str factor (lib_gap OFF, signal_lag=0) for
both families. The lib_gap repair is reproduced faithfully by the daily
month-begin/month-end decomposition (build_str_decomposition.py) and toggled in
the bias lattice; it is not this builder's job.

Weighting: value-weight by par `offering_amt` (panel `size`, §2.4) — now that
FISD supplies real par sizes, str is VW (the run_str_lib_gap_aoi.py pattern-gate
used equal weighting only because size was a pre-FISD placeholder).

Family-indexed per A9: emits `str_raw` (xret/ret on the raw family) and
`str_corr` (corr family). The engine's long-short nets out the safe rate, so
sorting on `ret` vs `xret` is identical; we sort on the family `ret`.

Output: data/development/factors/str.parquet with columns
  date, str_raw, str_corr, n_bonds_raw, n_bonds_corr.

Usage:
  python scripts/build_str.py

Requires: data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.characteristic_sort import run_characteristic_sort  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402

# Anchor headline uses the §2.1 total-return panel when present (BBW_ANCHOR_PANEL
# overrides; clean maximal fallback). For str/mom6 the coupon carry largely
# cancels in the long-short, so the basis barely matters — but kept consistent.
_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "str.parquet"
REPORT_OUT = OUT_DIR / "str_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def str_rulebook(signal_lag: int = 0) -> dict:
    """str: sort on prior-month return (score=ret), VW par, long the LOSERS
    (bottom group 0), short the WINNERS (top group 4). signal_lag=0 is the
    as-published lib_gap=OFF arm."""
    return {
        "score": "score",
        "groups": 5,
        "weighting": "by_size",
        "long_group": 0,   # losers (low prior return)
        "short_group": 4,  # winners (high prior return)
        "signal_lag": signal_lag,
        "nw_lags": None,
    }


def _family_panel(maximal: pd.DataFrame, family: str) -> pd.DataFrame:
    """Materialise the engine-shape panel via the canonical view() interface,
    then set score = ret (str sorts on the prior-month return itself)."""
    cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    panel = view(maximal, cfg).drop_duplicates(subset=["cusip", "date"]).reset_index(drop=True)
    panel["score"] = panel["ret"]
    return panel[["cusip", "date", "ret", "size", "score"]]


def run_family(maximal: pd.DataFrame, family: str) -> dict:
    panel = _family_panel(maximal, family)
    res = run_characteristic_sort(panel, str_rulebook(signal_lag=0))
    mr = res["monthly_returns"][["date", "strategy_ret", "n_bonds"]].rename(
        columns={"strategy_ret": f"str_{family}", "n_bonds": f"n_bonds_{family}"}
    )
    return {"monthly": mr, "summary": res["summary"]}


def write_factor(factor: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"str",
        b"factor_source": b"DRR_2026_BBW_anchor",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"value_weight_par_offering_amt",
        b"leg_convention": b"losers_minus_winners_raw_negative",
        b"lib_gap":      b"OFF_signal_lag_0_as_published",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(factor: pd.DataFrame, summaries: dict) -> None:
    def _stats(fam: str) -> dict:
        s = factor[f"str_{fam}"].dropna()
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "t_stat": float(summaries[fam]["t_stat"]),
            "sharpe": float(summaries[fam]["sharpe"]),
            "sign": "negative" if (len(s) and s.mean() < 0) else "non-negative",
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "input_panel": str(PANEL_FILE.relative_to(REPO_ROOT)),
        "leg_convention": "losers - winners (Construction A); raw premium is "
                          "NEGATIVE and matches DRR's stored data series sign "
                          "(figures show +0.99 sign-corrected). Gate on magnitude.",
        "weighting": "value-weight by par offering_amt (panel `size`, §2.4)",
        "lib_gap": "OFF (signal_lag=0, as-published). Repair handled by the daily "
                   "month-begin/month-end decomposition + bias lattice, not here.",
        "target_reference": "DRR-2026 Table 2 Panel A single-sort month-end mean "
                            "≈ -0.99%/mo (t -4.46). Magnitude-based gate (§8).",
        "headline_series": "str_corr",
        "notes": "HEADLINE = str_corr. str_raw is the meas_err=OFF family and is "
                 "outlier-dominated: VW leg means inherit uncorrected price-error "
                 "returns (xret_raw reaches ~10^6) that the corr family's filters "
                 "remove, so str_raw's level is not usable (kept for A9 family "
                 "completeness). The meas_err differential is examined at the leg "
                 "level in Chunk 4, not via this raw factor level.",
        "str_raw": _stats("raw"),
        "str_corr": _stats("corr"),
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
    maximal = pd.read_parquet(PANEL_FILE)
    print(f"  {len(maximal):,} rows, {maximal['cusip'].nunique():,} cusips")

    summaries = {}
    monthly = {}
    for fam in ("raw", "corr"):
        print(f"Running str engine on {fam} family (VW, losers-winners, signal_lag=0)...")
        out = run_family(maximal, fam)
        monthly[fam] = out["monthly"]
        summaries[fam] = out["summary"]

    factor = monthly["raw"].merge(monthly["corr"], on="date", how="outer").sort_values(
        "date").reset_index(drop=True)

    write_factor(factor)
    write_report(factor, summaries)

    print("\nDone.")
    for fam in ("raw", "corr"):
        s = factor[f"str_{fam}"].dropna()
        if len(s):
            print(f"  str_{fam}: mean {s.mean()*100:+.3f}%/mo (sign {'NEG' if s.mean()<0 else 'POS'}), "
                  f"sd {s.std(ddof=1)*100:.3f}%, t {summaries[fam]['t_stat']:+.2f}, "
                  f"{len(s)} months")
    print(f"  → {OUT_FILE}")
    print("  Note: raw premium NEGATIVE by design (losers-winners); DRR data file "
          "matches, DRR figures are sign-flipped. Compare on |magnitude| only.")


if __name__ == "__main__":
    main()
