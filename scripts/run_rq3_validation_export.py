"""RQ3 instrument-validation export — surface the recovery-of-planted-signal numbers
that certify the measurement instruments into one citeable results JSON.

These validations ALREADY PASS in the deterministic test suite; this driver runs the SAME shared
data-generating processes (``evaluation/rq3_validation/recovery_dgps.py``, extracted verbatim from
the test batteries) and library functions, and writes the recovered numbers to
``results/auditor/rq3_validation_<date>.json``.

Blocks:
  * **characteristic-sort recovery** — a planted long-short alpha is recovered within ±3 SE of the
    realised theoretical spread; a null DGP produces no spurious signal; a benchmark regression
    recovers a planted alpha/beta.
  * **IPCA recovery** — the planted factor subspace is recovered (max principal angle) at high and
    moderate SNR; the factors align (R²); a shuffled-date out-of-sample control collapses R².
  * **recovery sweep** — the §10.5 interval-coverage recovery curve over the pre-registered ρ grid.
  * **measurement-error recall** — the decimal-shift detector recovers injected ×10/×100 slips.
  * a **pointer** to the already-committed auditor Layer-B calibration (FPR / MDE / magnitude sweep
    / interaction recovery), which has its own exporter.

Deterministic (fixed seeds), offline, DEV-ONLY (synthetic panels only; no data files, no holdout).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.auditor.validation.recovery_sweep import (  # noqa: E402
    DEFAULT_RECOVERY_GRID,
    run_recovery_sweep,
)
from agents.quant.library.characteristic_sort import (  # noqa: E402
    regress_on_benchmark,
    run_characteristic_sort,
)
from agents.quant.library.ipca import (  # noqa: E402
    build_sufficient_stats,
    fit_ipca,
    fit_ipca_recursive,
    oos_total_r2,
)
from evaluation.rq3_validation.recovery_dgps import (  # noqa: E402
    build_quality_dgp,
    factor_alignment_r2,
    make_ipca_panel,
    max_principal_angle_deg,
)

_RESULTS_DIR = REPO_ROOT / "results" / "auditor"
_CALIBRATION_GLOB = "results/auditor/calibration/run_*/calibration.json"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Characteristic-sort recovery (planted long-short alpha, null, benchmark).
# ---------------------------------------------------------------------------

def characteristic_sort_recovery() -> dict:
    # Planted alpha (mirrors test_recovery_of_planted_long_short_alpha).
    n_bonds, n_months, alpha, sigma = 100, 121, 0.01, 0.04
    panel, qualities = build_quality_dgp(n_bonds, n_months, alpha, sigma, seed=2026)
    sorted_q = np.sort(qualities)
    theoretical = alpha * (sorted_q[-20:].mean() - sorted_q[:20].mean())
    rb = {"score": "score", "groups": 5, "weighting": "equal", "min_bonds": 100, "nw_lags": 0}
    res = run_characteristic_sort(panel, rb)
    recovered = float(res["summary"]["average"])
    t_stat = float(res["summary"]["t_stat"])
    n_run = int(res["summary"]["n_months"])
    se = math.sqrt(sigma**2 / 10.0 / n_run)
    planted = {
        "planted_alpha": alpha, "theoretical_spread": float(theoretical),
        "recovered_mean": recovered, "t_stat": t_stat, "n_months": n_run,
        "abs_error": float(abs(recovered - theoretical)), "three_se": 3.0 * se,
        "within_3se": bool(abs(recovered - theoretical) < 3.0 * se), "significant": bool(t_stat > 3.0),
    }

    # Null: score independent of next-ret (mirrors test_recovery_under_null_no_signal,
    # tests/unit/test_characteristic_sort_recovery.py — kept in sync by seed 2027 + fixed params).
    rng = np.random.default_rng(2027)
    bonds = [f"B{i:03d}" for i in range(n_bonds)]
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for j, d in enumerate(dates):
        scores = rng.normal(size=n_bonds)
        month_rets = np.zeros(n_bonds) if j == 0 else rng.normal(scale=sigma, size=n_bonds)
        for i, bid in enumerate(bonds):
            rows.append({"cusip": bid, "date": d, "ret": float(month_rets[i]),
                         "size": 100.0, "score": float(scores[i])})
    null_res = run_characteristic_sort(pd.DataFrame(rows), rb)
    null = {"t_stat": float(null_res["summary"]["t_stat"]),
            "recovered_mean": float(null_res["summary"]["average"]),
            "no_spurious_signal": bool(abs(float(null_res["summary"]["t_stat"])) < 3.0)}

    # Benchmark regression (mirrors test_recovery_benchmark_regression).
    T, sigma_eps, true_alpha, true_beta = 240, 0.02, 0.002, 0.5
    rng = np.random.default_rng(2029)
    bdates = pd.date_range("2000-01-31", periods=T, freq="ME")
    fac = rng.normal(scale=0.04, size=T)
    y = pd.Series(true_alpha + true_beta * fac + rng.normal(scale=sigma_eps, size=T), index=bdates)
    out = regress_on_benchmark(y, pd.DataFrame({"date": bdates, "MKT": fac}), nw_lags=0)
    bench = {"planted_alpha": true_alpha, "recovered_alpha": float(out["alpha"]),
             "planted_beta": true_beta, "recovered_beta": float(out["betas"]["MKT"]),
             "n_obs": int(out["n_obs"])}

    return {"planted_long_short_alpha": planted, "null_no_signal": null,
            "benchmark_regression": bench}


# ---------------------------------------------------------------------------
# IPCA recovery (subspace angle, factor alignment, shuffled-date null).
# ---------------------------------------------------------------------------

def ipca_recovery() -> dict:
    subspace = {}
    for snr, bound in ((40.0, 2.0), (4.0, 10.0)):
        rng = np.random.default_rng(int(snr) + 1)
        Z, R, months, truth = make_ipca_panel(rng, L=30, K=3, T=264, snr=snr)
        fit = fit_ipca(build_sufficient_stats(Z, R, months), K=3)
        angle = max_principal_angle_deg(fit.gamma_beta, truth["gamma_beta"])
        subspace[f"snr_{int(snr)}"] = {"max_principal_angle_deg": angle, "bound_deg": bound,
                                       "converged": bool(fit.converged), "within_bound": bool(angle < bound)}

    rng = np.random.default_rng(2)
    Z, R, months, truth = make_ipca_panel(rng, L=30, K=3, T=264, snr=40.0)
    fit = fit_ipca(build_sufficient_stats(Z, R, months), K=3)
    r2 = factor_alignment_r2(fit.factors, truth["F_true"])
    factor = {"factor_alignment_r2": r2, "bound": 0.95, "above_bound": bool(r2 > 0.95)}

    # Shuffled-date OOS null (mirrors test_oos_shuffle_control) — small panel, tractable.
    rng = np.random.default_rng(101)
    Z, R, months, _ = make_ipca_panel(rng, L=15, K=3, T=84, n_range=(400, 400), snr=20.0)
    rec = fit_ipca_recursive(build_sufficient_stats(Z, R, months), K=3, burn_in=36, tol=1e-4)
    intact = oos_total_r2(rec)
    perm = rng.permutation(rec.x_oos.shape[1])
    shuffled = oos_total_r2(rec._replace(x_oos=rec.x_oos[:, perm]))
    shuffle = {"intact_oos_r2": float(intact), "shuffled_oos_r2": float(shuffled),
               "intact_above_half": bool(intact > 0.5), "shuffled_collapses": bool(shuffled < 0.05 * intact)}

    return {"subspace_recovery": subspace, "factor_alignment": factor, "shuffled_date_null": shuffle}


# ---------------------------------------------------------------------------
# Recovery sweep (interval coverage) + measurement-error recall.
# ---------------------------------------------------------------------------

def recovery_sweep_block() -> dict:
    # A representative monotone sign-stable recovery curve over the pre-registered grid
    # (mirrors test_sweep_over_default_grid_sign_stable); the machinery is the §10.5 sweep.
    def eff(rho):
        e = -(1.0 - rho) * 0.1
        return e, e - 0.01, e + 0.01

    res = run_recovery_sweep(eff, headline_rule="FIXED_RECOVERY", headline_rho=0.4)
    out = res.to_dict()
    out["grid_is_preregistered"] = (tuple(res.grid) == tuple(DEFAULT_RECOVERY_GRID))
    out["note"] = ("representative synthetic recovery curve — demonstrates the §10.5 interval-"
                   "coverage sweep machinery over the pre-registered rho grid")
    return out


def meas_err_recall() -> dict:
    from apply_decimal_shift import apply_decimal_shift_vec, load_thresholds
    tc = load_thresholds()
    floor, ceiling = float(tc["price_floor"]), float(tc["price_ceiling"])
    # Inject known decimal slips that land ABOVE the ceiling so the detector must rescale:
    # idx1 = ×10 (100->1000, ÷10=100), idx3 = ×100 (100->10000, ÷100=100).
    truth = np.array([95.0, 100.0, 105.0, 100.0, 110.0])
    injected = truth.copy()
    injected[1] = 1000.0
    injected[3] = 10000.0
    res = apply_decimal_shift_vec(injected, floor, ceiling)
    detected = int(res.shift_applied[1]) + int(res.shift_applied[3])
    recovered_ok = bool(
        np.isclose(res.corrected[1], 100.0) and np.isclose(res.corrected[3], 100.0)
        and res.div10_mask[1] and res.div100_mask[3])
    untouched_ok = bool(not res.shift_applied[0] and np.isclose(res.corrected[0], 95.0))
    return {"n_injected": 2, "n_detected": detected, "recall": detected / 2.0,
            "values_recovered": recovered_ok, "clean_values_untouched": untouched_ok,
            "note": ("decimal-shift detector (×10 / ×100); the full 8-case measurement-error "
                     "injection suite is in tests/unit/test_meas_err_injection.py")}


def calibration_pointer() -> dict:
    hits = sorted(glob.glob(str(REPO_ROOT / _CALIBRATION_GLOB)))
    return {
        "component": "auditor.layer_b_calibration",
        "covers": "zero-injection FPR, signal/background magnitude sweep, MDE, interaction recovery",
        "exporter": "scripts/run_auditor_calibration.py",
        "artefact": (str(Path(hits[-1]).relative_to(REPO_ROOT)) if hits else None),
        "present": bool(hits),
    }


def build_validation_export() -> dict:
    return {
        "experiment": "rq3_instrument_validation",
        "characteristic_sort_recovery": characteristic_sort_recovery(),
        "ipca_recovery": ipca_recovery(),
        "recovery_sweep": recovery_sweep_block(),
        "measurement_error_recall": meas_err_recall(),
        "calibration_pointer": calibration_pointer(),
        "notes": {
            "anchor_triangulation": (
                "the §10.4 anchor triangulation is unit-verified in "
                "tests/unit/test_auditor_recovery_and_anchors.py; a live run reads the "
                "auditor.anchor_gate thresholds block, which is intentionally NOT added here "
                "(it would force a FROZEN_THRESHOLDS_SHA256 re-pin) — it needs the full-run "
                "observed effects to be meaningful."),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="output path (default results/auditor/rq3_validation_<date>.json)")
    args = ap.parse_args(argv)

    result = build_validation_export()
    stamp = datetime.now(timezone.utc)
    result["provenance"] = {
        "exporter": "scripts/run_rq3_validation_export.py",
        "run_timestamp": stamp.isoformat(),
        "git_commit": _git_commit(),
        "dev_only": True,
        "synthetic_dgp_only": True,
        "model_calls": 0,
        "spend_usd": 0.0,
    }

    out = Path(args.out) if args.out else _RESULTS_DIR / "rq3_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    cs = result["characteristic_sort_recovery"]["planted_long_short_alpha"]
    ip = result["ipca_recovery"]
    print(f"char-sort: recovered {cs['recovered_mean']:.5f} vs theoretical {cs['theoretical_spread']:.5f} "
          f"(within 3SE={cs['within_3se']}, t={cs['t_stat']:.2f})")
    print(f"IPCA: subspace angle snr40={ip['subspace_recovery']['snr_40']['max_principal_angle_deg']:.2f}°, "
          f"factor R²={ip['factor_alignment']['factor_alignment_r2']:.3f}, "
          f"shuffle collapses={ip['shuffled_date_null']['shuffled_collapses']}")
    print(f"meas-err recall: {result['measurement_error_recall']['recall']:.2f}")
    print(f"results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
