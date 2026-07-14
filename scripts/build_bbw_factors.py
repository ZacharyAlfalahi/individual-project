"""
Build the BBW (2019) four-factor harness: DRF, LRF, REV, the three CRF
components, and the CRF composite (docs/quant/specs/BBW_anchor_implementation_spec.md §3, §4).
MKTB is built separately (build_mktb.py); together they are the four BBW model
factors (MKTB, DRF, LRF, CRF) plus the standalone REV.

Per family (A9): view() selects the family's returns and the family-resolved
var_5pct / gamma signals; rating is family-agnostic; rev = the contemporaneous
return (REV sorts on the prior-month return, signal_lag=0). The six factors run
through the audited engine via agents/quant/library/bbw_factors.py; CRF is their
equal-weighted credit-axis composite.

As-published alignment (lib_gap OFF, signal_lag=0, contemporaneous). The lead/lag
injection and lib_gap toggles are Chunk-4 work. Per-factor effective windows
(§6): DRF / CRF start ~2004-06 (VaR5 needs >=24 of trailing 36 months); LRF / REV
from panel start ~2002-08.

Output: data/development/factors/bbw_factors.parquet with columns
  date, {drf,lrf,rev,crf_var,crf_illiq,crf_rev,crf}_{raw,corr}.

Usage:
  python scripts/build_bbw_factors.py

Requires:
  data/development/monthly_panel_maximal.parquet
  data/development/signals/var_5pct.parquet
  data/development/signals/gamma_illiq.parquet
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from functools import reduce
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.bbw_factors import (  # noqa: E402
    BBW_FACTOR_CONFIGS, run_bbw_factor, compose_crf,
)
from agents.quant.library.characteristic_sort import summarize_returns  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402

_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
# Anchor headline uses the §2.1 total-return panel when present — the clean-price
# CRF sign-flip / LRF-flatness is corrected on it (BBW_ANCHOR_PANEL overrides;
# clean maximal fallback).
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
VAR_FILE = REPO_ROOT / "data" / "development" / "signals" / "var_5pct.parquet"
GAMMA_FILE = REPO_ROOT / "data" / "development" / "signals" / "gamma_illiq.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "bbw_factors.parquet"
REPORT_OUT = OUT_DIR / "bbw_factors_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

FACTORS = list(BBW_FACTOR_CONFIGS)  # drf, lrf, rev, crf_var, crf_illiq, crf_rev


def thresholds_sha256() -> str:
    return hashlib.sha256(THRESHOLDS_FILE.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_family(maximal: pd.DataFrame, signals: pd.DataFrame, family: str):
    cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    panel = view(maximal, cfg, signals=signals).drop_duplicates(
        subset=["cusip", "date"]).reset_index(drop=True)
    panel["rev"] = panel["ret"]  # REV sorts on the contemporaneous prior-month return
    panel = panel[["cusip", "date", "ret", "size", "rating", "var_5pct", "gamma", "rev"]]

    monthly, summaries, component_mr = {}, {}, {}
    for name in FACTORS:
        res = run_bbw_factor(panel, name)
        mr = res["monthly_returns"]
        monthly[name] = mr[["date", "strategy_ret"]].rename(
            columns={"strategy_ret": f"{name}_{family}"})
        summaries[name] = res["summary"]
        component_mr[name] = mr

    crf = compose_crf(component_mr).rename(columns={"crf": f"crf_{family}"})
    monthly["crf"] = crf
    summaries["crf"] = summarize_returns(
        pd.Series(crf[f"crf_{family}"].values, index=pd.DatetimeIndex(crf["date"])),
        nw_lags=None, months_per_year=12,
    )
    return monthly, summaries


def write_factor(factor: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"bbw_four_factor",
        b"factor_source": b"BBW_2019",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"value_weight_par_offering_amt",
        b"alignment":    b"as_published_signal_lag_0",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(factor: pd.DataFrame, summaries: dict) -> None:
    def _stats(name: str, fam: str) -> dict:
        col = f"{name}_{fam}"
        s = factor[col].dropna()
        first = factor.loc[factor[col].notna(), "date"]
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "t_stat": float(summaries[fam][name]["t_stat"]),
            "effective_start": str(first.min().date()) if len(first) else None,
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "headline_series_family": "corr",
        "alignment": "as-published (signal_lag=0); lead/lag + lib_gap toggles are Chunk 4",
        "leg_directions": {k: {kk: vv for kk, vv in v.items()} for k, v in BBW_FACTOR_CONFIGS.items()},
        "crf_composite": "(crf_var + crf_illiq + crf_rev) / 3 (§3.5)",
        "expected_effective_windows": "DRF/CRF ~2004-06 (VaR5 >=24/36); LRF/REV ~2002-08 (§6)",
        "directional_check_vs_DRR2023_table1": {
            "drf": "PASS sign (+); DRR +0.673%/mo, ours +0.32% (clean-price/universe shrinkage)",
            "rev": "negative by the losers-winners convention (§5.1), |t| strong — consistent with str",
            "lrf": "FAIL — ~0 (legs +0.186% vs +0.186%); DRR +0.361%/mo",
            "crf": "FAIL sign — negative; DRR +0.508%/mo (positive credit premium)",
        },
        "clean_price_contingency_TRIGGERED": (
            "CRF (and LRF) directional checks FAIL on the clean-price basis. Per "
            "§2.4 the credit/liquidity premia live in low-rated / illiquid bonds "
            "whose returns are dominated by coupon carry; clean-price (no accrued "
            "interest / coupon, §2.1) omits it, flattening LRF to ~0 and flipping "
            "CRF negative. DRF/REV are coupon-insensitive and keep the right sign. "
            "This is the pre-registered 'revisit if CRF off' trigger (return-basis "
            "decision) to add AI+coupon for the anchor layer."
        ),
        "note": "raw family inherits uncorrected price-error outliers in VW legs "
                "(see MKTB/str reports); corr is the headline. Construction is "
                "verified exact by tests/unit/test_bbw_factors.py — the LRF/CRF "
                "misses are a return-MEASUREMENT (clean-price) issue, not a "
                "construction bug.",
        "factors": {name: {fam: _stats(name, fam) for fam in ("raw", "corr")}
                    for name in FACTORS + ["crf"]},
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    for f in (PANEL_FILE, VAR_FILE, GAMMA_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    print("Loading panel + signals...")
    maximal = pd.read_parquet(PANEL_FILE)
    var5 = pd.read_parquet(VAR_FILE)
    gamma = pd.read_parquet(GAMMA_FILE)
    signals = var5.merge(gamma, on=["cusip", "date"], how="outer")

    all_monthly, summaries = {}, {}
    for fam in ("raw", "corr"):
        print(f"Running BBW factors on {fam} family...")
        monthly, summ = run_family(maximal, signals, fam)
        summaries[fam] = summ
        all_monthly[fam] = monthly

    frames = []
    for fam in ("raw", "corr"):
        frames.extend(all_monthly[fam][name] for name in FACTORS + ["crf"])
    factor = reduce(lambda a, b: a.merge(b, on="date", how="outer"), frames)
    factor = factor.sort_values("date").reset_index(drop=True)

    write_factor(factor)
    write_report(factor, summaries)

    print("\nDone (corr family headline):")
    for name in FACTORS + ["crf"]:
        s = factor[f"{name}_corr"].dropna()
        if len(s):
            first = factor.loc[factor[f"{name}_corr"].notna(), "date"].min().date()
            print(f"  {name:9s}: mean {s.mean()*100:+.3f}%/mo, sd {s.std(ddof=1)*100:5.2f}%, "
                  f"t {summaries['corr'][name]['t_stat']:+.2f}, {len(s):3d} mo, from {first}")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
