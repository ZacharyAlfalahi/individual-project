"""
Real dev-panel driver for the Frozen-Loadings IPCA Differential (§12.7).

  python scripts/run_ipca_differential.py --smoke        # run (meas_err, drf), check engineering criteria
  python scripts/run_ipca_differential.py --all          # run all 9 runnable pairs

The smoke is ENGINEERING-ONLY (pre-registration §3): a surprising alpha SHIPS; only a DEFECT (a gate
misfiring, a leak, a broken invariant) fails it. All inputs are read from data/development/ — the
holdout is never touched. Verdict printed as JSON.

``--basis total_return`` (total-return sensitivity; design decision (2026-09-11)): the IPCA return leg and the fixed
anchor come from the default-flat total-return panel, while EVERY IPCA instrument (str_reversal, the
recomputed var_5pct / bond_vol / mom6, gamma_illiq, rating, maturity) is built from the
clean-price panel in the same cell state (clean-price instruments, total-return excess returns).
It is an unregistered sensitivity; ``--basis clean`` is the registered construction.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.thresholds import (  # noqa: E402
    load_ipca_execution_pairs,
    load_ipca_lambda,
    load_support_gate,
)
from agents.auditor.ipca_differential.fit_timing import time_one_fit  # noqa: E402
from agents.auditor.ipca_differential.panels import build_cell_feed, panel_states  # noqa: E402
from agents.auditor.ipca_differential.runner import (  # noqa: E402
    load_dev_inputs,
    run_runnable_pairs,
)
from scripts import basis_inputs  # noqa: E402

_BASIS_INSTRUMENT_NOTE = {
    "clean": ("registered construction: IPCA R = next-month clean xret / bond_vol; every instrument from the "
              "clean-price panel (var_5pct, bond_vol, mom6 recomputed from clean returns)."),
    "total_return": ("unregistered sensitivity (design decision (2026-09-11)): IPCA R = next-month default-flat total-return "
                     "xret / max(clean bond_vol, floor) and the fixed anchor on total return; EVERY instrument "
                     "(str_reversal, recomputed var_5pct / bond_vol / mom6, gamma_illiq, rating, "
                     "time_to_maturity) from the clean-price panel in the same cell state. A row enters only if "
                     "both its total-return and clean values are finite."),
}


def _load_inputs(basis: str | None):
    """(maximal, signals, registry): the default dev loader without a basis, else the basis loader."""
    return load_dev_inputs() if basis is None else basis_inputs.load_basis_inputs(basis)


def _characteristics_maximal(basis: str | None):
    """The clean-price maximal panel the instruments are built from on the total-return basis; None otherwise
    (the single-panel construction)."""
    return basis_inputs.load_basis_maximal("clean") if basis == "total_return" else None


def default_out_dir(basis: str | None = None) -> Path:
    """The default ``results/ipca_differential/run`` without a basis;
    ``basis_inputs.basis_dir(basis, "ipca_differential")`` with one."""
    if basis is None:
        return REPO_ROOT / "results" / "ipca_differential" / "run"
    return basis_inputs.basis_dir(basis, "ipca_differential")

_PROJ_KEYS = {
    "min_cross_section_n", "rank_fail_months", "max_condition_number", "pseudoinverse_used",
    "pseudoinverse_tolerance", "months_removed", "cell_refused", "n_valid_months",
}


def _json_roundtrips(d: dict) -> bool:
    try:
        json.loads(json.dumps(d, default=str))
        return True
    except (TypeError, ValueError):
        return False


def smoke_checks(result, timing_s: float, min_common_months: int) -> dict:
    """The pre-registered engineering-only pass criteria (never the alpha's size/sign)."""
    d = result.to_dict()
    cells = d["cells"]
    proj_ok = all(_PROJ_KEYS <= set(c["projection_diagnostics"]) for c in cells)
    return {
        "four_cells": len(cells) == 4,
        "projection_diagnostics_populated": proj_ok,
        "no_cell_refused": all(not c["projection_diagnostics"]["cell_refused"] for c in cells),
        "condition_numbers_finite": all(
            math.isfinite(c["projection_diagnostics"]["max_condition_number"]) for c in cells
        ),
        "frozen_hash_recorded": all(bool(c["frozen_state_hash"]) for c in cells),
        "support_clears_gate": d["common_support_n_months"] >= min_common_months,
        "bootstrap_computed": d["interaction_bracket_raw"]["interval_status"] == "computed",
        "corr_fields_present": (
            d["interaction_bracket_raw"]["name"].endswith("_corr")
            and all(k in d for k in ("data_margin_theta_n_corr", "data_margin_theta_b_corr", "total_corr"))
        ),
        "serialization_roundtrips": _json_roundtrips(d),
        "timing_measured": timing_s > 0.0,
    }


def _shipping_numbers(result) -> dict:
    d = result.to_dict()

    def eff(k: str) -> dict:
        return {"value": d[k]["value"], "interval": d[k]["interval"], "status": d[k]["interval_status"]}

    return {
        "bias": d["bias"], "anchor": d["anchor"], "is_focal": d["is_focal"],
        "common_support_n_months": d["common_support_n_months"],
        "cells": {c["label"]: c["value"] for c in d["cells"]},
        "interaction_bracket_raw": eff("interaction_bracket_raw"),
        "doe_interaction_effect": eff("doe_interaction_effect"),
        "data_margin_theta_n_corr": eff("data_margin_theta_n_corr"),
        "data_margin_theta_b_corr": eff("data_margin_theta_b_corr"),
        "total_corr": eff("total_corr"),
        "secondary_endtoend": d["secondary_endtoend"],
    }


def _basis_record(basis: str) -> dict:
    rec = {"basis_provenance": basis_inputs.basis_provenance(basis).to_dict(),
           "basis_instrument_note": _BASIS_INSTRUMENT_NOTE[basis]}
    if basis == "total_return":
        rec["instrument_panels"] = basis_inputs.basis_provenance("clean").to_dict()["panels"]
    return rec


def run_smoke(basis: str | None = None) -> int:
    pairs = load_ipca_execution_pairs()
    bias, anchor = pairs.smoke_pair
    print("[smoke] loading dev inputs (data/development/ only; holdout untouched) ...", flush=True)
    t0 = time.perf_counter()
    maximal, signals, reg = _load_inputs(basis)
    char_max = _characteristics_maximal(basis)
    print(f"[smoke] inputs loaded in {time.perf_counter() - t0:.1f}s; running ({bias}, {anchor}) ...", flush=True)

    # D-A52 timing criterion: time one production fit on the real P_N feed.
    lam = load_ipca_lambda()
    p_n, _ = panel_states(bias, maximal, signals)
    if char_max is None:
        feed_n = build_cell_feed(p_n, reg, "corr")
    else:
        c_n, _ = panel_states(bias, char_max, signals)
        feed_n = build_cell_feed(c_n, reg, "corr", return_panel=p_n)
    timing_s = time_one_fit(feed_n, lam, repeats=1)
    print(f"[smoke] one production fit: {timing_s:.3f}s  (T={len(feed_n.months)} months, "
          f"N_m median~{int(sorted(len(z) for z in feed_n.Z)[len(feed_n.Z)//2])})", flush=True)

    t1 = time.perf_counter()
    result = run_runnable_pairs(maximal, signals, reg, subset=[(bias, anchor)],
                                characteristics_maximal=char_max)[0]
    print(f"[smoke] differential computed in {time.perf_counter() - t1:.1f}s", flush=True)

    checks = smoke_checks(result, timing_s, load_support_gate().min_common_months)
    verdict = {
        "pair": [bias, anchor],
        "one_fit_seconds": round(timing_s, 4),
        "engineering_checks": checks,
        "PASS": all(checks.values()),
        "shipping_numbers": _shipping_numbers(result),
    }
    if basis is not None:
        verdict.update(_basis_record(basis))
    print(json.dumps(verdict, indent=2, default=str))
    print(f"\n[smoke] VERDICT: {'PASS' if verdict['PASS'] else 'FAIL — engineering check failed'}")
    return 0 if verdict["PASS"] else 1


def _config_hash() -> str:
    import hashlib
    import yaml
    block = yaml.safe_load((REPO_ROOT / "docs" / "thresholds.yaml").read_text())["auditor"]["ipca_differential"]
    return hashlib.sha256(yaml.safe_dump(block, sort_keys=True).encode()).hexdigest()[:16]


def _mdc_bp(ci) -> float | None:
    """Minimum detectable coupling ≈ conditional-bootstrap CI half-width, in bp/month."""
    return None if ci is None else round((ci[1] - ci[0]) / 2 * 1e4, 4)


def run_all(out_dir: Path | None = None, basis: str | None = None) -> int:
    from datetime import datetime, timezone

    from agents.auditor.ipca_differential.runner import run_pair_full

    pairs = load_ipca_execution_pairs()
    maximal, signals, reg = _load_inputs(basis)
    char_max = _characteristics_maximal(basis)
    print(f"[all] running {len(pairs.runnable_pairs())} runnable pairs "
          f"(recompute + §5.4 coupling stability) ...", flush=True)
    runnable, stability = [], []
    for bias, anchor in pairs.runnable_pairs():
        res, stab = run_pair_full(bias, anchor, maximal, signals, reg, characteristics_maximal=char_max)
        runnable.append(res.to_dict())
        stability.append(stab.to_dict()["coupling_stability_diagnostic"])
        print(f"  {bias}×{anchor}: I={res.interaction_bracket_raw.value * 1e4:+.3f}bp  "
              f"MDC={_mdc_bp(res.interaction_bracket_raw.interval)}bp  "
              f"sign_surv={stab.sign_survival}  usable={stab.n_usable_refits}", flush=True)

    refused = [
        {"bias": b, "anchor": a, "status": "refused",
         "reason": pairs.refused.reason, "note": pairs.refused.note}
        for (b, a) in pairs.refused_pairs()
    ]
    stab_by_pair = {s["bias"] + "×" + s["anchor"]: s for s in stability}
    matrix = {}
    for r in runnable:
        key = r["bias"] + "×" + r["anchor"]
        s = stab_by_pair[key]
        matrix[key] = {
            "I": r["interaction_bracket_raw"]["value"],
            "I_ci": r["interaction_bracket_raw"]["interval"],
            "mdc_bp": _mdc_bp(r["interaction_bracket_raw"]["interval"]),
            "data_margin_theta_n": r["data_margin_theta_n_corr"]["value"],
            "stability_sign_survival": s["sign_survival"],
            "stability_magnitude_survival": s["magnitude_survival"],
            "stability_n_usable_refits": s["n_usable_refits"],
            "n_months": r["common_support_n_months"], "is_focal": r["is_focal"],
        }
    run_log = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_tag": "ipca-differential-prereg",
        "thresholds_ipca_hash": _config_hash(),
        "window": "development 2002-2021 (holdout untouched)",
        "signal_propagation": "recompute var/vol/mom6 per panel state (D-A55); gamma_illiq daily-sourced, membership-only",
        "signal_recompute_note": "var_5pct/bond_vol/mom6 recomputed from each panel state's returns; gamma_illiq daily-sourced (membership-only)",
        "anchors": "str = value-weighted decile reversal sort on xret; mom6 = equal-weighted decile momentum with Jostova skip=1 + staggered holding=6 (canonical build_mom6); drf = bivariate var_5pct×rating (BBW)",
        "stability_diagnostic": "§5.4, R'=25 blocked-resample refits per pair; sign/magnitude survival reported (degenerate when I≡0)",
        "mdc_definition": "minimum detectable coupling ≈ conditional-bootstrap CI half-width in bp/month (D-A56)",
        "bootstrap_seed": 20260612, "stability_seed": 20260612,
        "n_runnable": len(runnable), "n_refused": len(refused),
    }
    if basis is not None:
        run_log.update(_basis_record(basis))
    payload = {"run_log": run_log, "matrix": matrix, "runnable": runnable, "refused": refused,
               "stability": stability}
    out_dir = out_dir or default_out_dir(basis)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_9pairs.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"run_log": run_log, "matrix": matrix}, indent=2, default=str))
    print(f"\n[all] wrote {out_dir / 'results_9pairs.json'}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the (meas_err, drf) smoke")
    ap.add_argument("--all", action="store_true", help="run all 9 runnable pairs")
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="return basis (default: the default dev loader). total_return = total-return IPCA "
                         "returns and anchor with clean-price instruments (unregistered sensitivity)")
    ap.add_argument("--out", type=Path, default=None,
                    help="--all output directory (default results/ipca_differential/run; with "
                         "--basis results/consistent_basis/<basis>/ipca_differential)")
    args = ap.parse_args(argv)
    if args.out is not None and not args.all:
        ap.error("--out applies to --all only (the smoke writes nothing)")
    if args.smoke:
        sys.exit(run_smoke() if args.basis is None else run_smoke(basis=args.basis))
    elif args.all:
        sys.exit(run_all(args.out) if args.basis is None else run_all(args.out, basis=args.basis))
    else:
        ap.error("choose --smoke or --all")


if __name__ == "__main__":
    main()
