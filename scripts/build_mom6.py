"""
Build the mom6 (6-month momentum) anchor factor (Jostova et al. 2013;
docs/quant/specs/BBW_anchor_implementation_spec.md §5.2). The mom6_1 mnemonic = formation 6 /
skip 1; holding is H=6 staggered (Jostova; confirmed for DRR-2026 mom6_1).

Construction (all params from thresholds.yaml signals.mom6):
  * signal = trailing cumulative return (build_mom6_signal.py).
  * signal_lag = skip_months (1): rank on the window ending one month before
    formation, leaving the most recent month unused (the Jostova skip).
  * deciles (n_groups=10), EQUAL-weighted within decile (Jostova — NOT VW).
  * leg = winners − losers = P10 − P1 (long top decile, short bottom).
  * holding_months = 6, staggered/overlapping via overlap.run_with_holding_period:
    month-m return = mean over the cohorts formed in m-1..m-6 (non-overlapping
    inference). H=1 would reduce exactly to run_characteristic_sort.

This differs from str/BBW on two axes (deciles not quintiles; EW not VW) — both
deliberate and Jostova-faithful.

Family-indexed per A9: emits `mom6_raw` and `mom6_corr` (the staggered P10−P1
spread per family). The look-ahead (ex-post vs ex-ante winsorization) bias is a
Chunk-4 toggle; this builder emits the as-published staggered factor.

Output: data/development/factors/mom6.parquet with columns
  date, mom6_raw, mom6_corr, n_bonds_raw, n_bonds_corr.

Usage:
  python scripts/build_mom6.py

Requires:
  data/development/monthly_panel_maximal.parquet (build_monthly_panel.py)
  data/development/signals/mom6.parquet           (build_mom6_signal.py)
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
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.characteristic_sort import summarize_returns  # noqa: E402
from agents.quant.library.overlap import run_with_holding_period  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402

_TOTAL = REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet"
_CLEAN = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
# Anchor headline uses the §2.1 total-return panel when present (BBW_ANCHOR_PANEL
# overrides; clean maximal fallback).
PANEL_FILE = Path(os.environ.get("BBW_ANCHOR_PANEL", str(_TOTAL if _TOTAL.exists() else _CLEAN)))
SIGNAL_FILE = REPO_ROOT / "data" / "development" / "signals" / "mom6.parquet"
OUT_DIR = REPO_ROOT / "data" / "development" / "factors"
OUT_FILE = OUT_DIR / "mom6.parquet"
REPORT_OUT = OUT_DIR / "mom6_report.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def load_config() -> dict:
    with open(THRESHOLDS_FILE) as f:
        cfg = yaml.safe_load(f)
    block = cfg.get("signals", {}).get("mom6")
    if block is None:
        raise KeyError("thresholds.yaml is missing signals.mom6 block")
    for key in ("skip_months", "holding_months", "n_groups", "weighting"):
        if key not in block:
            raise KeyError(f"thresholds.yaml signals.mom6 missing '{key}'")
    return block


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


def mom6_rulebook(cfg: dict) -> dict:
    """mom6: sort on the trailing cumulative-return signal, deciles, EW, long
    the top decile (winners, group n-1), short the bottom (losers, group 0).
    signal_lag = skip_months is the Jostova skip."""
    n = int(cfg["n_groups"])
    return {
        "score": "mom6",
        "groups": n,
        "weighting": cfg["weighting"],     # "equal"
        "long_group": n - 1,               # P10 winners
        "short_group": 0,                  # P1 losers
        "signal_lag": int(cfg["skip_months"]),
        "nw_lags": None,
    }


def run_family(maximal: pd.DataFrame, signal: pd.DataFrame, family: str, cfg: dict) -> dict:
    view_cfg = RunConfig(
        panel_view=PanelViewConfig(price_family=family, stale_mask=False,
                                   include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=int(cfg["skip_months"]),
                                        expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    # view() resolves the family-indexed signal (mom6_<family> → mom6).
    panel = view(maximal, view_cfg, signals=signal)
    panel = panel.drop_duplicates(subset=["cusip", "date"]).reset_index(drop=True)
    panel = panel[["cusip", "date", "ret", "size", "mom6"]]

    mr = run_with_holding_period(
        panel, mom6_rulebook(cfg), holding_period=int(cfg["holding_months"])
    )
    out = mr[["date", "strategy_ret", "n_bonds"]].rename(
        columns={"strategy_ret": f"mom6_{family}", "n_bonds": f"n_bonds_{family}"}
    )
    ser = pd.Series(mr["strategy_ret"].values, index=pd.DatetimeIndex(mr["date"].values))
    summary = summarize_returns(ser, nw_lags=None, months_per_year=12)
    return {"monthly": out, "summary": summary}


def write_factor(factor: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(factor, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta.update({
        b"factor_name":  b"mom6",
        b"factor_source": b"Jostova_2013_DRR_2026",
        b"primary_key":  b"date",
        b"families":     b"raw,corr",
        b"weighting":    b"equal_decile",
        b"leg_convention": b"winners_minus_losers_P10_minus_P1",
        b"holding":      b"staggered_H6",
        b"family_policy": b"A9_no_cross_family_mixing",
    })
    table = table.replace_schema_metadata(meta)
    tmp = OUT_FILE.with_suffix(".parquet.tmp")
    pq.write_table(table, str(tmp))
    os.replace(tmp, OUT_FILE)
    print(f"  Written: {OUT_FILE}")


def write_report(factor: pd.DataFrame, summaries: dict, cfg: dict) -> None:
    def _stats(fam: str) -> dict:
        s = factor[f"mom6_{fam}"].dropna()
        return {
            "n_months": int(len(s)),
            "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
            "sd_pct_per_month": float(s.std(ddof=1) * 100) if len(s) > 1 else None,
            "t_stat": float(summaries[fam]["t_stat"]),
            "sharpe": float(summaries[fam]["sharpe"]),
        }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "construction": {k: cfg[k] for k in
                         ("formation_months", "min_obs", "skip_months",
                          "holding_months", "n_groups", "weighting")
                         if k in cfg},
        "leg_convention": "winners - losers = P10 - P1 (deciles, equal-weighted, "
                          "staggered H=6 holding; Jostova 2013)",
        "lab_filter": "NONE — no winsorization applied. The ex-post vs ex-ante "
                      "winsorization is a Chunk-4 toggle (the dominant mom6 bias, §7.1).",
        "expected_un_winsorized_level": "≈ 0. Per §5.2/§7.1 the +0.30%/mo momentum "
            "premium is ENTIRELY attributable to ex-post full-sample winsorization, "
            "so the un-winsorized factor here is expected to be ≈0 (and statistically "
            "insignificant) — NOT +0.30. This is the baseline the Chunk-4 toggle moves: "
            "ex-post winsorization → ≈+0.30 (DRR biased), ex-ante → ≈0 (DRR corrected); "
            "the EP−EA gap is the measured look-ahead bias (§8 gate).",
        "headline_series": "mom6_corr",
        "mom6_raw": _stats("raw"),
        "mom6_corr": _stats("corr"),
    }
    tmp = REPORT_OUT.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, REPORT_OUT)
    print(f"  Report: {REPORT_OUT}")


def main():
    for f in (PANEL_FILE, SIGNAL_FILE):
        if not f.exists():
            print(f"ERROR: required input not found: {f}", file=sys.stderr)
            sys.exit(1)

    cfg = load_config()
    print(f"Config: skip={cfg['skip_months']}, holding={cfg['holding_months']}, "
          f"deciles={cfg['n_groups']}, weighting={cfg['weighting']}")

    print("Loading panel + signal")
    maximal = pd.read_parquet(PANEL_FILE)
    signal = pd.read_parquet(SIGNAL_FILE)

    summaries, monthly = {}, {}
    for fam in ("raw", "corr"):
        print(f"Running mom6 (overlap H={cfg['holding_months']}) on {fam} family...")
        out = run_family(maximal, signal, fam, cfg)
        monthly[fam] = out["monthly"]
        summaries[fam] = out["summary"]

    factor = monthly["raw"].merge(monthly["corr"], on="date", how="outer").sort_values(
        "date").reset_index(drop=True)

    write_factor(factor)
    write_report(factor, summaries, cfg)

    print("\nDone.")
    for fam in ("raw", "corr"):
        s = factor[f"mom6_{fam}"].dropna()
        if len(s):
            tag = " (HEADLINE)" if fam == "corr" else ""
            print(f"  mom6_{fam}{tag}: mean {s.mean()*100:+.3f}%/mo, "
                  f"sd {s.std(ddof=1)*100:.3f}%, t {summaries[fam]['t_stat']:+.2f}, "
                  f"{len(s)} months")
    print(f"  → {OUT_FILE}")


if __name__ == "__main__":
    main()
