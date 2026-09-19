"""
Accrual validation — Chunk 5 capstone (necessary + sufficient, per the
diagnostic-first protocol). Compares the clean-price and total-return panels:

  1. LEVELS-CORRECTED (the positive magnitude check at factor scale): CRF sign
     should turn positive, LRF should develop a premium, MKTB should rise toward
     ~0.4%/mo. (Bond-level AI magnitude is unit-tested in test_accrual.py.)
  2. DIFFERENTIALS-STATIC (necessary, not sufficient): the sign-invariant bias
     gates must not move materially — accrual is a level shift that cancels in
     differentials. Checked on the mom6 EP-EA gap and the BBW lead/lag
     correlation collapse.
  3. FALLBACK EXPOSURE (the queryable diagnostic): per BBW factor, how many
     30/360-fallback bonds (true basis ACT/*) land in the extreme score quintile
     each month — so a surprising factor-month can be cross-checked.

Output: data/development/headlines/accrual_validation.json

Usage:
  python scripts/run_accrual_validation.py [--total PATH] [--out PATH]
      defaults: data/development/monthly_panel_total_return.parquet,
                data/development/headlines/accrual_validation.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agents.quant.library.bbw_factors import run_bbw_factor, compose_crf, CRF_COMPONENTS  # noqa: E402
from agents.quant.library.characteristic_sort import _assign_groups  # noqa: E402
from agents.quant.library.lead_lag import inject_lead_lag  # noqa: E402
from agents.quant.library.market_factor import compute_market_factor  # noqa: E402
from agents.quant.library.overlap import run_with_holding_period  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402
from agents.quant.library.winsorize import winsorize_returns  # noqa: E402
from build_mom6 import mom6_rulebook  # noqa: E402
from scripts import basis_inputs  # noqa: E402

DEV = REPO_ROOT / "data" / "development"
CLEAN = DEV / "monthly_panel_maximal.parquet"
TOTAL = DEV / "monthly_panel_total_return.parquet"
OUT = DEV / "headlines" / "accrual_validation.json"
import yaml  # noqa: E402

from shared.licensed_inputs import require_licensed_input  # noqa: E402
with open(REPO_ROOT / "docs" / "thresholds.yaml") as _f:
    _CFG = yaml.safe_load(_f)
MOM6 = _CFG["signals"]["mom6"]
LAB = _CFG["bias_toggles"]["lab_filter"]
GATE = _CFG["validation"]["gate_thresholds"]["accrual_validation"]


def _rel(path: Path) -> str:
    """Repo-relative path when inside the repo, else the path as given (never raises)."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Accrual validation (clean vs total-return panel).")
    ap.add_argument("--total", type=Path, default=TOTAL,
                    help="total-return panel (default data/development/monthly_panel_total_return.parquet)")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="report path (default data/development/headlines/accrual_validation.json)")
    return ap.parse_args(argv or [])


def input_provenance(clean_path: Path, total_path: Path) -> dict:
    """The two compared panels, repo-relative path + sha256 each."""
    return {name: {"path": _rel(p), "sha256": basis_inputs.sha256(p)}
            for name, p in (("clean_panel", clean_path), ("total_panel", total_path))}


def _bbw_panel(panel_df, signals):
    cfg = RunConfig(PanelViewConfig("corr", False, False), ConstructionConfig(0, "none"), EvaluationConfig())
    p = view(panel_df, cfg, signals=signals).drop_duplicates(["cusip", "date"]).reset_index(drop=True)
    p["rev"] = p["ret"]
    return p


def bbw_means(p):
    comp, out = {}, {}
    # xret: crf_rev's control is xret (bbw_factors.py); mirror the
    # factor builder's column set (build_bbw_factors.py) so the CRF components run.
    sub = p[["cusip", "date", "ret", "size", "rating", "var_5pct", "gamma", "rev", "xret"]]
    for name in ["drf", "lrf"] + list(CRF_COMPONENTS):
        mr = run_bbw_factor(sub, name)["monthly_returns"]
        comp[name] = mr
        out[name] = float(mr["strategy_ret"].mean() * 100)
    out["crf"] = float(compose_crf(comp)["crf"].mean() * 100)
    return out, comp


def mom6_gap(panel_df, signal):
    cfg = RunConfig(PanelViewConfig("corr", False, False),
                    ConstructionConfig(int(MOM6["skip_months"]), "none"), EvaluationConfig())
    p = view(panel_df, cfg, signals=signal).drop_duplicates(["cusip", "date"]).reset_index(drop=True)
    base = p[["cusip", "date", "ret", "size", "mom6"]].copy()
    def prem(panel):
        mr = run_with_holding_period(panel, mom6_rulebook(MOM6), holding_period=int(MOM6["holding_months"]))
        return pd.Series(mr["strategy_ret"].values, index=pd.DatetimeIndex(mr["date"].values))
    ep = base.copy(); ep["ret"] = winsorize_returns(ep["ret"], level=LAB["level"], loc=LAB["loc"], mode="ex_post")
    ea = base.copy(); ea["ret"] = winsorize_returns(ea["ret"], dates=ea["date"], level=LAB["level"], loc=LAB["loc"], mode="ex_ante")
    return float((prem(ep) - prem(ea)).mean() * 100)


def leadlag_drf_corr(comp):
    drf = comp["drf"][["date", "strategy_ret"]].dropna().rename(columns={"strategy_ret": "v"})
    defective = inject_lead_lag(drf.rename(columns={"v": "strategy_ret"}), 1, ("2004-08", "2014-12"))
    s = pd.Timestamp("2004-08-31"); e = pd.Timestamp("2014-12-31")
    j = pd.concat([drf.set_index("date")["v"].rename("c"),
                   defective.set_index("date")["strategy_ret"].rename("d")], axis=1)
    j = j[(j.index >= s) & (j.index <= e)].dropna()
    return float(j["c"].corr(j["d"]))


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    total_path, out = Path(args.total), Path(args.out)
    if total_path != TOTAL and out == OUT:
        print(f"ERROR: --total {total_path} with the default --out would overwrite the default report {OUT}; "
              "pass --out", file=sys.stderr); sys.exit(2)
    for f in (CLEAN, total_path):
        if not f.exists():
            print(f"ERROR: missing {f}", file=sys.stderr); sys.exit(1)

    clean = pd.read_parquet(CLEAN)
    total = pd.read_parquet(total_path)
    signals = (pd.read_parquet(require_licensed_input(DEV / "signals" / "var_5pct.parquet", "var-5pct signal"))
               .merge(pd.read_parquet(require_licensed_input(DEV / "signals" / "gamma_illiq.parquet", "gamma-illiquidity signal")), on=["cusip", "date"], how="outer"))
    mom6_sig = pd.read_parquet(require_licensed_input(DEV / "signals" / "mom6.parquet", "mom6 signal panel"))

    pc, pt = _bbw_panel(clean, signals), _bbw_panel(total, signals)
    bc, comp_c = bbw_means(pc)
    bt, comp_t = bbw_means(pt)

    mktb_c = float(compute_market_factor(clean, "xret_corr")["mktb"].mean() * 100)
    mktb_t = float(compute_market_factor(total, "xret_corr")["mktb"].mean() * 100)

    # Fallback exposure: per month, fallback bonds in the rating-axis extreme
    # quintile (group 4 = CRF long leg, single-axis approx).
    fb = total[["cusip", "date", "day_count_fallback"]]
    pf = pt.merge(fb, on=["cusip", "date"], how="left").dropna(subset=["rating"])
    fb_counts = []
    for dt, d in pf.groupby("date", sort=True):
        q = _assign_groups(d, "rating", "cusip", 5).to_numpy()
        fb_counts.append(int(d["day_count_fallback"].to_numpy()[q == 4].sum()))
    fb_in_long = np.array(fb_counts)
    months_with_fb = int((fb_in_long > 0).sum())

    # Differentials-static.
    gap_c, gap_t = mom6_gap(clean, mom6_sig), mom6_gap(total, mom6_sig)
    ll_c, ll_t = leadlag_drf_corr(comp_c), leadlag_drf_corr(comp_t)

    gap_tol, corr_tol = float(GATE["gap_tol"]), float(GATE["corr_tol"])
    levels_corrected = (bt["crf"] > 0) and (bt["lrf"] > bc["lrf"]) and (mktb_t > mktb_c)
    diffs_static = (abs(gap_t - gap_c) < gap_tol) and (abs(ll_t - ll_c) < corr_tol)
    overall_pass = levels_corrected and diffs_static

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "inputs": input_provenance(CLEAN, total_path),
        "overall_pass": bool(overall_pass),
        "levels_corrected": {
            "crf_clean_pct": bc["crf"], "crf_total_pct": bt["crf"],
            "lrf_clean_pct": bc["lrf"], "lrf_total_pct": bt["lrf"],
            "drf_clean_pct": bc["drf"], "drf_total_pct": bt["drf"],
            "mktb_clean_pct": mktb_c, "mktb_total_pct": mktb_t,
            "pass": bool(levels_corrected),
            "criterion": "CRF sign turns positive, LRF premium appears, MKTB rises",
        },
        "differentials_static": {
            "mom6_ep_minus_ea_clean_pct": gap_c, "mom6_ep_minus_ea_total_pct": gap_t,
            "leadlag_drf_corr_clean": ll_c, "leadlag_drf_corr_total": ll_t,
            "pass": bool(diffs_static),
            "criterion": "sign-invariant bias gates unchanged (accrual cancels in differentials)",
        },
        "fallback_exposure": {
            "eligible_fallback_pct": float(total.loc[total["universe_eligible"].eq(True), "day_count_fallback"].mean() * 100),
            "months_with_fallback_in_crf_long_leg": months_with_fb,
            "max_fallback_bonds_in_a_long_leg_month": int(fb_in_long.max()),
            "note": "ACT/* bonds priced on 30/360; surfaced so a surprising CRF month is cross-checkable.",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print("Accrual validation (clean → total, corr family):")
    print(f"  LEVELS-CORRECTED  (pass={levels_corrected}):")
    print(f"    CRF : {bc['crf']:+.3f} → {bt['crf']:+.3f}%/mo   [sign flip to +]")
    print(f"    LRF : {bc['lrf']:+.3f} → {bt['lrf']:+.3f}%/mo   [premium appears]")
    print(f"    DRF : {bc['drf']:+.3f} → {bt['drf']:+.3f}%/mo")
    print(f"    MKTB: {mktb_c:+.3f} → {mktb_t:+.3f}%/mo   [rises toward ~0.4]")
    print(f"  DIFFERENTIALS-STATIC (pass={diffs_static}):")
    print(f"    mom6 EP-EA gap : {gap_c:+.3f} → {gap_t:+.3f}%/mo")
    print(f"    lead/lag DRF corr: {ll_c:.3f} → {ll_t:.3f}")
    print(f"  FALLBACK EXPOSURE: eligible {report['fallback_exposure']['eligible_fallback_pct']:.2f}%; "
          f"CRF-long-leg months with a fallback bond: {months_with_fb}")
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}  → {out}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main(sys.argv[1:])
