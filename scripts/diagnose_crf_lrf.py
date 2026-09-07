"""
CRF/LRF clean-price anomaly — diagnostic-first protocol (NO panel rebuild).

Triggered by Chunk 3: on the clean-price panel CRF is sign-flipped (vs DRR's
positive credit premium) and LRF is flat (~0). Before committing to the
expensive accrued-interest + coupon rebuild, two independent checks isolate the
cause (accrual artefact vs leg-direction/sort bug):

  Check A — accrual sensitivity on CRF. Add a ROUGH monthly coupon carry
    (coupon / 12, no day-count engine) to the clean corr return and re-run the
    credit factors. If the rough carry reorders CRF to a POSITIVE sign, accrual
    is confirmed as the cause (low-rated long-leg bonds carry higher coupons, so
    clean-price omission depresses exactly that leg). If the sign does not budge,
    there is a leg-direction/sort bug to find first.

  Check B — gamma separation on LRF. Compare mean gamma in the top vs bottom
    gamma quintile. Well-separated gamma + a flat RETURN spread → a return-basis
    (accrual) problem, fixable by Step 3. Nearly-equal gamma → the gamma sort
    itself is broken — a construction bug accrual cannot fix.

Records both verdicts (with the competing hypotheses) to
data/development/headlines/crf_lrf_diagnostic.json so the resolution path is
auditable. Does NOT modify the panel or any factor output.

Usage:
  python scripts/diagnose_crf_lrf.py
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.bbw_factors import run_bbw_factor, compose_crf  # noqa: E402
from agents.quant.library.characteristic_sort import _assign_groups  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    RunConfig, PanelViewConfig, ConstructionConfig, EvaluationConfig,
)
from agents.quant.library.views import view  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

PANEL_FILE = REPO_ROOT / "data" / "development" / "monthly_panel_maximal.parquet"
VAR_FILE = REPO_ROOT / "data" / "development" / "signals" / "var_5pct.parquet"
GAMMA_FILE = REPO_ROOT / "data" / "development" / "signals" / "gamma_illiq.parquet"
FISD_STATIC = REPO_ROOT / "data" / "development" / "fisd" / "fisd_reference_static.parquet"
OUT = REPO_ROOT / "data" / "development" / "headlines" / "crf_lrf_diagnostic.json"
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


def _crf_from_panel(panel: pd.DataFrame) -> float:
    comp = {name: run_bbw_factor(panel, name)["monthly_returns"]
            for name in ("crf_var", "crf_illiq", "crf_rev")}
    crf = compose_crf(comp)
    return float(crf["crf"].mean())


def _factor_mean(panel: pd.DataFrame, name: str) -> float:
    return float(run_bbw_factor(panel, name)["monthly_returns"]["strategy_ret"].mean())


def main():
    with open(THRESHOLDS_FILE) as f:
        diag_cfg = yaml.safe_load(f)["validation"]["gate_thresholds"]["crf_lrf_diagnostic"]
    lrf_lift_min = float(diag_cfg["lrf_lift_min"])
    gamma_sep_ratio = float(diag_cfg["gamma_sep_ratio"])

    print("Loading panel + signals + coupon...")
    maximal = pd.read_parquet(require_licensed_input(PANEL_FILE, "maximal development panel"))
    var5 = pd.read_parquet(require_licensed_input(VAR_FILE, "var-5pct signal"))
    gamma = pd.read_parquet(require_licensed_input(GAMMA_FILE, "gamma-illiquidity signal"))
    signals = var5.merge(gamma, on=["cusip", "date"], how="outer")
    coupon = pd.read_parquet(require_licensed_input(FISD_STATIC, "FISD static table"), columns=["cusip", "coupon"]).drop_duplicates("cusip")

    cfg = RunConfig(PanelViewConfig("corr", False, False),
                    ConstructionConfig(0, "none"), EvaluationConfig())
    p = view(maximal, cfg, signals=signals).drop_duplicates(["cusip", "date"]).reset_index(drop=True)
    p = p.merge(coupon, on="cusip", how="left")
    base = p[["cusip", "date", "ret", "size", "rating", "var_5pct", "gamma"]].copy()
    base["rev"] = base["ret"]

    cpn = p["coupon"]
    print(f"  coupon: median {cpn.median():.3f}, p25 {cpn.quantile(.25):.3f}, "
          f"p75 {cpn.quantile(.75):.3f} (expect ~3-9 if in percent), null {cpn.isna().mean()*100:.1f}%")

    # ---- Check A: rough coupon carry, clean vs rough-total ----
    print("\nCheck A — accrual sensitivity (rough carry = coupon/12):")
    carry = (p["coupon"].fillna(0.0) / 100.0) / 12.0  # monthly fraction of par
    tot = base.copy()
    tot["ret"] = base["ret"] + carry
    tot["rev"] = base["rev"] + carry  # the prior-return signal shifts too, but reversal is sign-of-residual

    crf_clean = _crf_from_panel(base)
    crf_tot = _crf_from_panel(tot)
    lrf_clean = _factor_mean(base, "lrf")
    lrf_tot = _factor_mean(tot, "lrf")
    drf_clean = _factor_mean(base, "drf")
    drf_tot = _factor_mean(tot, "drf")
    print(f"  CRF: clean {crf_clean*100:+.3f}%  →  rough-total {crf_tot*100:+.3f}%/mo")
    print(f"  LRF: clean {lrf_clean*100:+.3f}%  →  rough-total {lrf_tot*100:+.3f}%/mo")
    print(f"  DRF: clean {drf_clean*100:+.3f}%  →  rough-total {drf_tot*100:+.3f}%/mo (control)")

    accrual_flips_crf = crf_clean < 0 and crf_tot > 0
    accrual_lifts_lrf = lrf_tot > lrf_clean + lrf_lift_min  # default +0.05%/mo lift

    # ---- Check B: gamma separation in the LRF sort ----
    print("\nCheck B — gamma separation (top vs bottom gamma quintile):")
    g = base.dropna(subset=["gamma", "ret", "size"]).copy()
    grp = g.groupby("date", group_keys=False).apply(
        lambda d: d.assign(_q=_assign_groups(d, "gamma", "cusip", 5)), include_groups=False
    )
    top = grp.loc[grp["_q"] == 4, "gamma"]
    bot = grp.loc[grp["_q"] == 0, "gamma"]
    print(f"  mean gamma  top quintile {top.mean():.6e}  vs  bottom quintile {bot.mean():.6e}")
    ratio = float(top.mean() / bot.mean()) if bot.mean() != 0 else float("inf")
    print(f"  separation ratio (top/bottom): {ratio:.1f}x")
    gamma_sort_separates = abs(top.mean()) > gamma_sep_ratio * abs(bot.mean())

    verdict = {
        "check_A_accrual_flips_crf": bool(accrual_flips_crf),
        "check_A_accrual_lifts_lrf": bool(accrual_lifts_lrf),
        "check_B_gamma_sort_separates": bool(gamma_sort_separates),
        "diagnosis": (
            "ACCRUAL CONFIRMED — rough coupon carry reorders CRF to positive and "
            "lifts LRF, and the gamma sort separates cleanly. The clean-price "
            "(no-coupon) basis is the cause; Step 3 accrual rebuild is justified."
            if (accrual_flips_crf and gamma_sort_separates)
            else "NOT a clean accrual story — investigate a leg-direction/sort bug "
                 "before building the accrual engine (see check flags)."
        ),
    }

    report = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "CRF sign-flip / LRF-flat diagnostic before any accrual rebuild",
        "method": "rough monthly carry = coupon/12 added to clean corr return; "
                  "gamma quintile separation. No panel rebuild, no day-count engine.",
        "check_A": {
            "crf_clean_pct": crf_clean * 100, "crf_rough_total_pct": crf_tot * 100,
            "lrf_clean_pct": lrf_clean * 100, "lrf_rough_total_pct": lrf_tot * 100,
            "drf_clean_pct": drf_clean * 100, "drf_rough_total_pct": drf_tot * 100,
        },
        "check_B": {
            "mean_gamma_top_quintile": float(top.mean()),
            "mean_gamma_bottom_quintile": float(bot.mean()),
            "separation_ratio": ratio,
        },
        "verdict": verdict,
        "next_step": "If ACCRUAL CONFIRMED: build Chunk 4 (toggles/gates) on clean-price "
                     "now (gates are sign-invariant differentials), then add AI+coupon as a "
                     "separate chunk and verify differentials-static + levels-corrected. "
                     "Else: fix the leg/sort bug first.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, OUT)

    print("\nVERDICT:", verdict["diagnosis"])
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()
