#!/usr/bin/env python
"""System-level integration arms: a clean-variant integration null and an integrated known-error
positive control. Both reuse the tested engine and auditor machinery. Deterministic,
dev/synthetic only (no data/holdout/, no LLM).

Arm A -- clean-variant integration null: the zero-injection synthetic scenario is run through the
  SAME engine spine the T4b positive uses (lattice -> common support -> first-order DOE ->
  bootstrap). The instrument must NOT manufacture an effect on clean data: every toggle's DOE is
  within +/-vartheta and every bootstrap CI covers zero. It is the direct counterpart of the T4b
  planted-survivorship positive.

Arm B -- integrated known-error positive control: the BBW factors are built LIVE through the Quant
  executor (view -> run_bbw_factor) on the dev panel, the documented DRR lead/lag defect is injected,
  and the pre-registered gate grades the live output. This differs from the component-level control
  (`run_leadlag_gate.py`), which grades the pre-built factor parquet. NOTE: it is a Quant->Auditor integration,
  NOT the full Librarian->Quant->Auditor path -- BBW factors are driven by hardcoded rulebooks, not a
  Librarian extraction (documented limitation).

    ./.venv/bin/python scripts/run_integration_arms.py [--replicates 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.checks.bootstrap import run_bootstrap  # noqa: E402
from agents.auditor.data.synthetic_panel import build_scenario  # noqa: E402
from agents.auditor.schemas.toggle import TOGGLE_IDS  # noqa: E402
from agents.auditor.thresholds import load_vartheta  # noqa: E402
from agents.auditor.validation.known_error_control import evaluate_known_error_control  # noqa: E402
from agents.auditor.validation.layer_b_fixtures import run_scenario  # noqa: E402
from agents.quant.library.bbw_factors import compose_crf, run_bbw_factor  # noqa: E402
from agents.quant.library.lead_lag import inject_lead_lag  # noqa: E402
from agents.quant.library.run_config import (  # noqa: E402
    ConstructionConfig,
    EvaluationConfig,
    PanelViewConfig,
    RunConfig,
)
from agents.quant.library.views import view  # noqa: E402
from agents.reporter.format import to_bps  # noqa: E402
from scripts import basis_inputs  # noqa: E402

_DEFAULT_OUT = _REPO_ROOT / "results" / "scientist" / "integration_arms.json"


def default_out(basis: str | None = None) -> Path:
    """The default output without a basis; the consistent-basis location with one (never the default file)."""
    return _DEFAULT_OUT if basis is None else basis_inputs.basis_dir(basis, "scientist", "integration_arms.json")


# --- Arm A: clean-variant integration null -------------------------------------------------

def run_clean_null(*, seed: int = 2026, replicates: int = 200) -> dict:
    r = run_scenario(build_scenario(None, seed=seed))
    boot = run_bootstrap(r.lattice.cells, r.common, TOGGLE_IDS, n_replicates=replicates,
                         data_driven_block_months=6, min_effective_blocks=3, holding_period=1,
                         seed=seed + 1000)
    vartheta = load_vartheta()
    ci = boot.doe_ci()
    toggles, all_quiet, all_cover0 = {}, True, True
    for t in TOGGLE_IDS:
        doe = float(r.doe_first_order[t])
        lo, hi = ci[frozenset({t})]
        quiet = abs(doe) < vartheta
        covers0 = lo <= 0.0 <= hi
        all_quiet &= quiet
        all_cover0 &= covers0
        toggles[t] = {"doe_bp": to_bps(doe), "ci_bp": [to_bps(lo), to_bps(hi)],
                      "within_vartheta": bool(quiet), "ci_covers_zero": bool(covers0)}
    return {"arm": "clean_variant_integration_null", "seed": seed, "n_replicates": replicates,
            "vartheta_bp": to_bps(vartheta), "all_toggles_within_vartheta": bool(all_quiet),
            "all_ci_cover_zero": bool(all_cover0), "passed": bool(all_quiet and all_cover0),
            "toggles": toggles,
            "note": "clean synthetic panel through the T4b engine spine; the instrument "
                    "must manufacture no effect (counterpart of the planted-survivorship positive)."}


# --- Arm B: integrated known-error positive control (live Quant->Auditor) -------------------

_CRF_SUBFACTORS = ("crf_illiq", "crf_rev", "crf_var")   # CRF = compose_crf over these


def _prep_panel(maximal: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    """The corrected dev panel viewed for the BBW factor build, prepared exactly as the canonical
    build_bbw_factors.run_family: gamma_illiq -> gamma, and rev/xret aliased to the raw prior-month
    return (BBW's reversal signal)."""
    cfg = RunConfig(panel_view=PanelViewConfig(price_family="corr", stale_mask=False,
                                               include_terminal_rows=False),
                    construction=ConstructionConfig(signal_lag=0, expost_trim="none"),
                    evaluation=EvaluationConfig())
    panel = view(maximal, cfg, signals=signals).drop_duplicates(
        subset=["cusip", "date"]).reset_index(drop=True).rename(columns={"gamma_illiq": "gamma"})
    panel["rev"] = panel["ret"]
    panel["xret"] = panel["ret"]
    return panel


def _live_factor(panel: pd.DataFrame, name: str) -> pd.DataFrame:
    """One BBW factor built LIVE through the Quant executor (date + strategy_ret). CRF is not a
    single engine factor -- it is the composite (compose_crf over crf_illiq/crf_rev/crf_var)."""
    if name == "crf":
        comp = {s: run_bbw_factor(panel, s)["monthly_returns"] for s in _CRF_SUBFACTORS}
        return compose_crf(comp).rename(columns={"crf": "strategy_ret"})[["date", "strategy_ret"]]
    return run_bbw_factor(panel, name)["monthly_returns"][["date", "strategy_ret"]]


def _window_corr(correct: pd.DataFrame, other: pd.DataFrame, window) -> tuple[float, int]:
    start = pd.Timestamp(window[0]) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(window[1]) + pd.offsets.MonthEnd(0)
    j = pd.concat([correct.set_index("date")["strategy_ret"].rename("correct"),
                   other.set_index("date")["strategy_ret"].rename("other")], axis=1)
    j = j[(j.index >= start) & (j.index <= end)].dropna()
    return float(j["correct"].corr(j["other"])), int(len(j))


def _load_dev_inputs(basis: str | None = None):
    """(maximal, signals, registry): the default dev loader (load_dev_inputs) without a basis,
    else the basis loader."""
    if basis is None:
        from agents.auditor.ipca_differential.runner import load_dev_inputs
        return load_dev_inputs()
    return basis_inputs.load_basis_inputs(basis)


def run_integrated_leadlag_positive(basis: str | None = None) -> dict:
    maximal, signals, _registry = _load_dev_inputs(basis)
    panel = _prep_panel(maximal, signals)
    import yaml
    ll = yaml.safe_load((_REPO_ROOT / "docs" / "thresholds.yaml").read_text())["bias_toggles"]["lead_lag"]
    lead, lag = int(ll["drf_crf_lead_months"]), int(ll["lrf_lag_months"])
    specs = [("drf", +lead, ll["drf_crf_window"], "lead"),
             ("crf", +lead, ll["drf_crf_window"], "lead"),
             ("lrf", -lag, ll["lrf_window"], "lag")]
    arms = {}
    for name, shift, window, kind in specs:
        correct = _live_factor(panel, name)
        defective = inject_lead_lag(correct, shift_months=shift, window=tuple(window))
        restored = inject_lead_lag(defective[["date", "strategy_ret"]], shift_months=-shift,
                                   window=tuple(window))
        corr_defect, n = _window_corr(correct, defective, window)
        corr_restored, _ = _window_corr(correct, restored, window)
        arms[name] = {"kind": f"{kind} ({shift:+d} month)", "window": list(window),
                      "n_months_in_window": n,
                      "corr_correct_vs_defective": corr_defect,
                      "corr_correct_vs_restored": corr_restored}
    verdict = evaluate_known_error_control({"arms": arms})
    result = {"arm": "integrated_known_error_positive_control", "pipeline": "quant_to_auditor_live",
              "passed": bool(verdict.passed), "detail": verdict.detail, "arms": arms,
              "note": "BBW factors built live through the Quant executor (not the pre-built factor parquet); "
                      "Quant->Auditor integration, not the full Librarian->Quant->Auditor path."}
    if basis is not None:
        result["basis"] = basis
        result["basis_provenance"] = basis_inputs.basis_provenance(basis).to_dict()
    return result


def run_integration_arms(replicates: int = 200, basis: str | None = None) -> dict:
    """Arm A is synthetic (basis-free); ``basis`` selects Arm B's dev panel (None = the default dev loader)."""
    a = run_clean_null(replicates=replicates)
    b = run_integrated_leadlag_positive() if basis is None else run_integrated_leadlag_positive(basis=basis)
    return {"component": "system_integration_arms", "reads_holdout": False, "cost_usd": 0.0,
            "clean_variant_null": a, "integrated_known_error_positive_control": b}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="return basis of Arm B's dev panel (default: the dev loader load_dev_inputs)")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output JSON (default {_DEFAULT_OUT.relative_to(_REPO_ROOT)}; with --basis "
                         "results/consistent_basis/<basis>/scientist/integration_arms.json)")
    args = ap.parse_args(argv)
    if args.basis is None:
        result = run_integration_arms(replicates=args.replicates)
    else:
        result = run_integration_arms(replicates=args.replicates, basis=args.basis)
    out = args.out if args.out is not None else default_out(args.basis)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    a, b = result["clean_variant_null"], result["integrated_known_error_positive_control"]
    print(f"[clean-null] passed={a['passed']} (all toggles within vartheta + CI cover 0)")
    print(f"[leadlag-positive] passed={b['passed']} -- {b['detail']}")
    print(f"[integration_arms] -> {out}")
    return 0 if (a["passed"] and b["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
