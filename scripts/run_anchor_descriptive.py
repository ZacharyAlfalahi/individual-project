"""
O5 D-Q3 — DESCRIPTIVE-ONLY anchor level comparison (development panel only).

For each anchor {drf, str, mom6} report the factor level as
  mean_pct_per_month + t_stat + n_months
under the 2x2 grid {price_family: raw, corr} x {basis: clean, total_return}.

This is DESCRIPTIVE CONTEXT for the O5 doc — NO pass/fail weight is attached.

FAITHFUL METHOD (reuses each builder, no re-implementation):
  We REUSE each builder's own run function so every anchor is constructed exactly
  as its committed builder constructs it, and we vary ONLY
    (a) the base panel  = basis  (clean | total_return), passed explicitly, and
    (b) price_family     = raw | corr.
  No generic uncorrected()/corrected() RunConfig is forced onto the anchors —
  that would override each anchor's own construction (e.g. mom6's Jostova skip
  signal_lag=1, str's lib_gap-OFF signal_lag=0) and mis-specify the factor.

Reused run units (all return the family factor + its summary; none write):
  * mom6: build_mom6.run_family(maximal, signal, family, cfg)   [build_mom6.py:110-131]
          config via build_mom6.load_config()                    [build_mom6.py:68-77]
  * str : build_str.run_family(maximal, family)                  [build_str.py:126-132]
  * drf : build_bbw_factors.run_family(maximal, signals, family) [build_bbw_factors.py:82-109]
          DRF is one of BBW_FACTOR_CONFIGS; run_family runs the audited engine
          (run_bbw_factor) per factor — we select 'drf' out of its output.

Mean / t / n are read out exactly as each builder's own write_report does, so the
total_return columns MUST reproduce the committed
  data/development/factors/{mom6,str,bbw_factors}_report.json  (sanity check).

Output (writes ONLY here — NEVER touches data/development/factors/*):
  results/quant/descriptive/run_<gitshort>/anchor_descriptive.json
  (hash-logged with git_commit + thresholds_sha256)

Development panel only. NEVER reads /data/holdout/.

Usage:
  ./.venv/bin/python scripts/run_anchor_descriptive.py
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the builders' own run functions (never re-implement a sort).
import build_mom6  # noqa: E402
import build_str  # noqa: E402
import build_bbw_factors  # noqa: E402

THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# Basis panels. clean = maximal (clean price); total_return = a legal basis panel
# (accrued interest + coupon), NOT a forbidden endpoint export.
BASES = {
    "clean": REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet",
    "total_return": REPO_ROOT / "data" / "development" / "monthly_panel_total_return.parquet",
}
MOM6_SIGNAL = REPO_ROOT / "data" / "development" / "signals" / "mom6.parquet"
VAR_SIGNAL = REPO_ROOT / "data" / "development" / "signals" / "var_5pct.parquet"
GAMMA_SIGNAL = REPO_ROOT / "data" / "development" / "signals" / "gamma_illiq.parquet"

FAMILIES = ("raw", "corr")


def _guard_no_holdout() -> None:
    for p in list(BASES.values()) + [MOM6_SIGNAL, VAR_SIGNAL, GAMMA_SIGNAL]:
        if "holdout" in p.parts:
            raise RuntimeError(f"REFUSED: path touches holdout: {p}")


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


def git_short() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _level(series: pd.Series, summary: dict) -> dict:
    """Read out mean %/mo, t, n_months exactly as each builder's write_report does:
    mean = dropna().mean()*100 ; t = summary['t_stat'] ; n = len(dropna())."""
    s = series.dropna()
    return {
        "mean_pct_per_month": float(s.mean() * 100) if len(s) else None,
        "t_stat": float(summary["t_stat"]),
        "n_months": int(len(s)),
    }


def run_mom6(maximal: pd.DataFrame, signal: pd.DataFrame) -> dict:
    cfg = build_mom6.load_config()
    out = {}
    for fam in FAMILIES:
        res = build_mom6.run_family(maximal, signal, fam, cfg)  # build_mom6.py:110
        out[fam] = _level(res["monthly"][f"mom6_{fam}"], res["summary"])
    return out


def run_str(maximal: pd.DataFrame) -> dict:
    out = {}
    for fam in FAMILIES:
        res = build_str.run_family(maximal, fam)  # build_str.py:126
        out[fam] = _level(res["monthly"][f"str_{fam}"], res["summary"])
    return out


def run_drf(maximal: pd.DataFrame, signals: pd.DataFrame) -> dict:
    out = {}
    for fam in FAMILIES:
        monthly, summaries = build_bbw_factors.run_family(maximal, signals, fam)  # build_bbw_factors.py:82
        out[fam] = _level(monthly["drf"][f"drf_{fam}"], summaries["drf"])
    return out


def main() -> None:
    _guard_no_holdout()

    # Load the mom6 signal once and the BBW signals once (family-agnostic inputs).
    mom6_signal = pd.read_parquet(MOM6_SIGNAL)
    var5 = pd.read_parquet(VAR_SIGNAL)
    gamma = pd.read_parquet(GAMMA_SIGNAL)
    bbw_signals = var5.merge(gamma, on=["cusip", "date"], how="outer")

    table: dict = {}
    for basis, panel_path in BASES.items():
        if not panel_path.exists():
            raise FileNotFoundError(f"base panel not found: {panel_path}")
        print(f"[{basis}] loading {panel_path.name}")
        maximal = pd.read_parquet(panel_path)

        print(f"[{basis}] drf  (BBW audited engine, raw+corr)")
        drf = run_drf(maximal, bbw_signals)
        print(f"[{basis}] str  (VW deciles P10-P1, signal_lag=0, raw+corr)")
        srev = run_str(maximal)
        print(f"[{basis}] mom6 (EW deciles P10-P1, Jostova skip + H=6, raw+corr)")
        mom6 = run_mom6(maximal, mom6_signal)

        table[basis] = {"drf": drf, "str": srev, "mom6": mom6}

        for anchor, res in (("drf", drf), ("str", srev), ("mom6", mom6)):
            for fam in FAMILIES:
                r = res[fam]
                print(f"    {anchor:5s} {fam:4s}: mean {r['mean_pct_per_month']:+.4f}%/mo, "
                      f"t {r['t_stat']:+.4f}, n {r['n_months']}")

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "thresholds_sha256": thresholds_sha256(),
        "purpose": "O5 D-Q3 descriptive-only anchor level comparison "
                   "(2x2 price_family x basis). NO pass/fail weight.",
        "method": "Reused each builder's own run_family (mom6/str/build_bbw_factors); "
                  "varied only base panel (basis) and price_family. No generic "
                  "uncorrected()/corrected() RunConfig forced onto the anchors.",
        "bases": {k: str(v.relative_to(REPO_ROOT)) for k, v in BASES.items()},
        "families": list(FAMILIES),
        "development_only": True,
        "reads_holdout": False,
        "table": table,
    }

    gs = git_short()
    out_dir = REPO_ROOT / "results" / "quant" / "descriptive" / f"run_{gs}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "anchor_descriptive.json"
    tmp = out_file.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.replace(out_file)
    print(f"\nWritten: {out_file}")


if __name__ == "__main__":
    main()
