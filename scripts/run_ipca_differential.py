"""
Real dev-panel driver for the Frozen-Loadings IPCA Differential (§12.7).

  python scripts/run_ipca_differential.py --smoke        # run (meas_err, drf), check engineering criteria
  python scripts/run_ipca_differential.py --all          # run all 9 runnable pairs (Step 4)

The smoke is ENGINEERING-ONLY (pre-registration §3): a surprising alpha SHIPS; only a DEFECT (a gate
misfiring, a leak, a broken invariant) fails it. All inputs are read from data/development/ — the
holdout is never touched. Verdict printed as JSON.
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


def run_smoke() -> int:
    pairs = load_ipca_execution_pairs()
    bias, anchor = pairs.smoke_pair
    print("[smoke] loading dev inputs (data/development/ only; holdout untouched) ...", flush=True)
    t0 = time.perf_counter()
    maximal, signals, reg = load_dev_inputs()
    print(f"[smoke] inputs loaded in {time.perf_counter() - t0:.1f}s; running ({bias}, {anchor}) ...", flush=True)

    # D-A52 timing criterion: time one production fit on the real P_N feed.
    lam = load_ipca_lambda()
    p_n, _ = panel_states(bias, maximal, signals)
    feed_n = build_cell_feed(p_n, reg, "corr")
    timing_s = time_one_fit(feed_n, lam, repeats=1)
    print(f"[smoke] one production fit: {timing_s:.3f}s  (T={len(feed_n.months)} months, "
          f"N_m median~{int(sorted(len(z) for z in feed_n.Z)[len(feed_n.Z)//2])})", flush=True)

    t1 = time.perf_counter()
    result = run_runnable_pairs(maximal, signals, reg, subset=[(bias, anchor)])[0]
    print(f"[smoke] differential computed in {time.perf_counter() - t1:.1f}s", flush=True)

    checks = smoke_checks(result, timing_s, load_support_gate().min_common_months)
    verdict = {
        "pair": [bias, anchor],
        "one_fit_seconds": round(timing_s, 4),
        "engineering_checks": checks,
        "PASS": all(checks.values()),
        "shipping_numbers": _shipping_numbers(result),
    }
    print(json.dumps(verdict, indent=2, default=str))
    print(f"\n[smoke] VERDICT: {'PASS' if verdict['PASS'] else 'DEFECT — fix, document, re-run'}")
    return 0 if verdict["PASS"] else 1


def _git_short() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _config_hash() -> str:
    import hashlib
    import yaml
    block = yaml.safe_load((REPO_ROOT / "docs" / "thresholds.yaml").read_text())["auditor"]["ipca_differential"]
    return hashlib.sha256(yaml.safe_dump(block, sort_keys=True).encode()).hexdigest()[:16]


def run_all(out_dir: Path | None = None) -> int:
    from datetime import datetime, timezone

    pairs = load_ipca_execution_pairs()
    maximal, signals, reg = load_dev_inputs()
    print(f"[all] running {len(pairs.runnable_pairs())} runnable pairs (recompute per panel state) ...", flush=True)
    results = run_runnable_pairs(maximal, signals, reg)
    runnable = [r.to_dict() for r in results]
    refused = [
        {"bias": b, "anchor": a, "status": "refused",
         "reason": pairs.refused.reason, "note": pairs.refused.note}
        for (b, a) in pairs.refused_pairs()
    ]
    # The 5x3 matrix (bracket I): 3 runnable bias-rows shown, 2 refused rows typed.
    matrix = {
        r["bias"] + "×" + r["anchor"]: {
            "I": r["interaction_bracket_raw"]["value"],
            "I_ci": r["interaction_bracket_raw"]["interval"],
            "data_margin_theta_n": r["data_margin_theta_n_corr"]["value"],
            "n_months": r["common_support_n_months"], "is_focal": r["is_focal"],
        }
        for r in runnable
    }
    run_log = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_short(),
        "prereg_tag": "ipca-differential-prereg",
        "thresholds_ipca_hash": _config_hash(),
        "window": "development 2002-2021 (holdout untouched)",
        "signal_propagation": "recompute var/vol/mom6 per panel state; gamma_illiq daily-sourced, membership-only",
        "signal_recompute_note": "verified numerically inert on the dev panel: stale_price masks 0 incremental bond-months (theta=30d), survivorship is membership-dominated",
        "n_runnable": len(runnable), "n_refused": len(refused),
    }
    payload = {"run_log": run_log, "matrix": matrix, "runnable": runnable, "refused": refused}
    out_dir = out_dir or (REPO_ROOT / "results" / "ipca_differential" / f"run_{_git_short()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_9pairs.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"run_log": run_log, "matrix": matrix}, indent=2, default=str))
    print(f"\n[all] wrote {out_dir / 'results_9pairs.json'}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run the (meas_err, drf) smoke")
    ap.add_argument("--all", action="store_true", help="run all 9 runnable pairs")
    args = ap.parse_args()
    if args.smoke:
        sys.exit(run_smoke())
    elif args.all:
        sys.exit(run_all())
    else:
        ap.error("choose --smoke or --all")


if __name__ == "__main__":
    main()
