"""
Build the standalone short-term-reversal (`str`) anchor factor (spec
BBW_anchor_implementation_spec.md §5.1): a single-sort, value-weighted (par),
monthly-rebalanced, one-month-holding long-short on the prior-month return.

Construction (per gold_str_drr_2026.md, DRR-2026 Table 1 Panel A — the paper's
STATED single sort): sort into DECILES (n_groups = 10), long the top decile P10
(past winners), short the bottom decile P1 (past losers): leg = WINNERS − LOSERS.
Score = the grounded `prior_1m_excess_return` concept, which the D27 concept->column
table binds to the engine-contract `xret` column (v2). Ranking on `xret` vs raw
`ret` is identical (the safe rate nets out cross-sectionally each month).

SIGN — the leg direction below is deliberate. DRR report this
construction at −0.99%/mo (reversal: winners underperform). On our development
corr panel it earns ≈ +0.95%/mo (t +5.1) — robust MOMENTUM, the opposite sign —
because DRR's short-term reversal is a microstructure (LIB) premium that the corr
cleaning removes (DRR's own clean estimate is only −0.17); the raw family that
would carry it is outlier-corrupted (the raw-family `Infinity`). This is a RESULT,
not a bug: the sign divergence is localised to the raw/LIB layer and owned by the
§8 bias-toggle decomposition, NOT reconciled here. Do NOT flip the leg or sweep a
parameter to manufacture −0.99; RQ3 gates on the raw->corr differential, never on
a naive magnitude match to the paper.

This builder emits the AS-PUBLISHED str factor (lib_gap OFF, signal_lag=0) for
both families. The lib_gap repair is reproduced faithfully by the daily
month-begin/month-end decomposition (build_str_decomposition.py) and toggled in
the bias lattice; it is not this builder's job.

Weighting: value-weight by par `offering_amt` (panel `size`, §2.4) — now that
FISD supplies real par sizes, str is VW (the earlier pre-FISD pattern-gate used
equal weighting only because `size` was then a placeholder).

Family-indexed per A9: emits `str_raw` (raw family) and `str_corr` (corr,
HEADLINE). Sort column = the grounded `prior_1m_excess_return` -> `xret` (D27 v2);
the long-short earns the realised `ret`. Ranking on `xret` vs `ret` is identical
(the safe rate nets out cross-sectionally), so the choice is a naming contract only.

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
    """str: sort on the prior-month excess return (score=xret, the grounded
    prior_1m_excess_return column), VW par, DECILES per DRR Table 1 Panel A —
    long the WINNERS (top decile 9, P10), short the LOSERS (bottom decile 0, P1).
    Leg = winners - losers, matching gold_str_drr_2026.md (long_leg: highest_signal,
    n_groups: 10). signal_lag=0 is the as-published lib_gap=OFF arm. (Read the
    SIGN note in the module docstring: this earns momentum on the corr dev panel,
    not DRR's −0.99 reversal, by design — do NOT flip the leg to chase the sign.)"""
    return {
        "score": "xret",
        "groups": 10,
        "weighting": "by_size",
        "long_group": 9,   # winners P10 (high prior return)
        "short_group": 0,  # losers P1 (low prior return)
        "signal_lag": signal_lag,
        "nw_lags": None,
    }


def _family_panel(maximal: pd.DataFrame, family: str) -> pd.DataFrame:
    """Materialise the engine-shape panel via the canonical view() interface.
    str sorts on `xret` (the grounded prior_1m_excess_return column, D27 v2); the
    engine earns the realised `ret`."""
    cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    panel = view(maximal, cfg).drop_duplicates(subset=["cusip", "date"]).reset_index(drop=True)
    return panel[["cusip", "date", "ret", "size", "xret"]]


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
        b"leg_convention": b"winners_minus_losers_p10_p1_deciles",
        b"n_groups":     b"10",
        b"score_column": b"xret_prior_1m_excess_return",
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
        "leg_convention": "winners - losers (long P10, short P1; DECILES, n_groups=10) "
                          "per DRR-2026 Table 1 Panel A and gold_str_drr_2026.md "
                          "(long_leg: highest_signal). Sort column = xret (grounded "
                          "prior_1m_excess_return, D27 v2).",
        "weighting": "value-weight by par offering_amt (panel `size`, §2.4)",
        "lib_gap": "OFF (signal_lag=0, as-published). Repair handled by the daily "
                   "month-begin/month-end decomposition + bias lattice, not here.",
        "target_reference": "DRR-2026 Table 1 Panel A unadjusted single-sort ≈ -0.99%/mo "
                            "(t -4.46), a REVERSAL. On our development corr panel this "
                            "SAME construction earns POSITIVE momentum (see `sign`): DRR's "
                            "reversal is a raw/LIB microstructure premium the corr cleaning "
                            "removes (DRR's own clean estimate is only -0.17). The sign "
                            "divergence is owned by the §8 bias-toggle decomposition, NOT "
                            "reconciled here. Do NOT gate on a "
                            "naive |magnitude| match to -0.99.",
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
        print(f"Running str engine on {fam} family (VW, winners-losers deciles P10-P1, signal_lag=0)...")
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
    print("  Note: winners-losers deciles per DRR Table 1 Panel A (gold construction). "
          "The corr premium is POSITIVE momentum, NOT DRR's -0.99 reversal — the reversal "
          "is a raw/LIB microstructure premium the corr cleaning removes (§8 owns it). "
          "do NOT flip the leg to chase the paper's sign.")


if __name__ == "__main__":
    main()
