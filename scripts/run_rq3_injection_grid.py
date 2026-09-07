#!/usr/bin/env python
"""RQ3 per-correction synthetic-injection grid.

For each of the five corrections, export the engineering-tier + statistical-tier figures the
reported grid needs, in bp/mo, computed by reusing the existing (tested) auditor injection +
magnitude-sweep machinery. No auditor/calibration/synthetic module is modified.

Columns (all bp/mo except the ratio and coverage):
  * Injected            -- the noise-free planted long-short effect (build_scenario with
                           SyntheticSpec(noise_sd=0.0) -> doe_first_order[bias] x 10000). The
                           injection *magnitude* is a DGP-internal knob, not bp; multiplying it by
                           10000 would invent a bp unit that is not there, so we report the realised
                           noise-free effect instead (the planted truth).
  * Recovered           -- the estimated effect at the default magnitude (mean over seeds of the
                           magnitude-sweep effect x 10000). Recovered ~ Injected because the
                           synthetic estimator is unbiased on common support; reported as found.
  * Signal/background   -- injected effect / largest UNINJECTED toggle effect on the injected panel
                           (single_bias_fixture; dominance_ratio 5.0 == the "req >= 5" bar). On the
                           clean DGP the background is ~machine-zero, so we report "PASS (>=5),
                           background < 0.01 bp/mo" rather than a meaningless ~1e16 ratio.
  * Interval coverage   -- fraction of per-seed bootstrap CIs (at the default magnitude) that cover
                           the noise-free Injected truth. The one column that needs a fresh run.
  * MDE                 -- minimum detectable effect (target power 0.8) over the magnitude grid,
                           mapped to bp/mo at the threshold magnitude. A structural (magnitude-
                           independent) channel has no genuine threshold -> flagged, not printed 0.

Deterministic, $0, synthetic DGP only (no data/holdout/, no LLM).

    ./.venv/bin/python scripts/run_rq3_injection_grid.py [--seeds 25] [--replicates 200] [--out PATH]

Default output: results/auditor/rq3_injection_grid.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.data.synthetic_panel import (  # noqa: E402
    DEFAULT_MAGNITUDES,
    SyntheticSpec,
    build_scenario,
)
from agents.auditor.checks.bootstrap import run_bootstrap  # noqa: E402
from agents.auditor.validation.layer_b_fixtures import run_scenario, single_bias_fixture  # noqa: E402
from agents.auditor.schemas.toggle import TOGGLE_IDS  # noqa: E402
from agents.reporter.format import to_bps  # noqa: E402

_BACKGROUND_FLOOR_BP = 0.01   # below this the uninjected background is machine-zero; report as PASS

# Degeneracy floor for the per-seed bootstrap DOE CI half-width, in RETURN units (1e-12 return
# units == 1e-8 bp/mo). Rationale: orders of magnitude below any material effect (the RQ3
# materiality floor is theta = 10 bp/mo = 1e-3 return units), yet orders of magnitude ABOVE the
# ~1e-17 machine-precision scatter of a deterministic contrast. A CI narrower than this carries no
# bootstrap variability at all -- e.g. meas_err's per-bond time-constant shift hits the raw family
# only, so the ON-OFF contrast is deterministic, the CI collapses to ~1e-17 width, and
# covered-vs-not becomes a floating-point coin flip. Such seeds are classified `degenerate`
# instead of covered/not-covered.
_DEGENERATE_CI_HALFWIDTH_FLOOR = 1e-12


def _classify_coverage(lo: float, hi: float, truth: float) -> str:
    """Classify one seed's CI: `degenerate` when the half-width is below
    _DEGENERATE_CI_HALFWIDTH_FLOOR (no genuine bootstrap interval exists), else
    `covered`/`not_covered` on the usual containment test."""
    if (hi - lo) / 2.0 < _DEGENERATE_CI_HALFWIDTH_FLOOR:
        return "degenerate"
    return "covered" if lo <= truth <= hi else "not_covered"


def _injected_bp(bias: str, seeds: int) -> float:
    """The noise-free planted long-short effect in bp/mo (mean over seeds)."""
    vals = []
    for s in range(seeds):
        sc = build_scenario(bias, DEFAULT_MAGNITUDES[bias], seed=s, spec=SyntheticSpec(noise_sd=0.0))
        vals.append(run_scenario(sc).doe_first_order[bias])
    return to_bps(statistics.fmean(vals))


def _signal_background(bias: str) -> dict:
    fx = single_bias_fixture(bias, seed=0)
    inj_bp, bg_bp = to_bps(abs(fx.injected_effect)), to_bps(abs(fx.max_other_effect))
    ratio = (inj_bp / bg_bp) if bg_bp > 0 else float("inf")
    return {"dominates_ge5": bool(fx.dominates), "injected_bp": inj_bp,
            "background_bp": bg_bp, "ratio": ratio,
            "background_machine_zero": bg_bp < _BACKGROUND_FLOOR_BP}


_CAL_DIR = _REPO_ROOT / "results" / "auditor" / "calibration"


def _calibration_path() -> Path:
    """The selected local calibration artifact (run_<git>/calibration.json).
    Fail-loud when absent or ambiguous -- recovered/MDE are read, never recomputed."""
    candidates = sorted(_CAL_DIR.glob("run_*/calibration.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one calibration run under {_CAL_DIR}, found "
            f"{len(candidates)} -- run scripts/run_auditor_calibration.py first")
    return candidates[0]


def _committed_recovered_mde(bias: str) -> dict:
    """Recovered (at the default magnitude) and MDE, read from the COMMITTED, validated calibration
    run — the same magnitude-sweep machinery, already run and cited. Reading it here
    is $0 and avoids re-running the (expensive) full grid; the values carry their source."""
    cal_path = _calibration_path()
    cal = json.loads(cal_path.read_text())
    pb = cal["magnitude_sweep"]["per_bias"][bias]
    default = DEFAULT_MAGNITUDES[bias]
    rec = pb["per_magnitude"][str(default)]["mean_effect"]
    recovered_bp = to_bps(abs(rec))
    mde_mag = pb.get("mde")
    # structural == detected already at magnitude 0 (the channel is magnitude-independent, e.g.
    # stale_price). Survivorship has a mag>0 mde here, but its calibration notes flag its
    # magnitude-zero as not a valid null (the structural distress-drop is always active), so its
    # printed MDE measures sensitivity to distress severity, not a general detection floor.
    structural = (mde_mag == 0.0)
    mde_bp = None
    if mde_mag is not None and not structural:
        at_mde = pb["per_magnitude"].get(str(mde_mag))
        mde_bp = to_bps(abs(at_mde["mean_effect"])) if at_mde else None
    return {"recovered_bp": recovered_bp,
            "recovered_source": f"committed calibration {cal_path.parent.name}",
            "mde_magnitude": mde_mag, "mde_bp": mde_bp, "mde_structural": structural}


def _coverage(bias: str, seeds: int, replicates: int) -> dict:
    """Interval coverage at the default magnitude. For each seed: (i) the seed's OWN noise-free
    planted truth (the plant scales with that panel, so the truth is per-seed, not a single scalar),
    and (ii) the seed's NOISY estimate with a REAL block-bootstrap DOE confidence interval (the same
    `run_bootstrap` the auditor uses). Coverage = the fraction of noisy CIs that cover that seed's
    own noise-free truth. (`SweepRecord.ci` from magnitude_sweep is a degenerate point interval, not
    a bootstrap CI, so it cannot be used here.)

    DEGENERACY GUARD: a seed whose CI half-width is below _DEGENERATE_CI_HALFWIDTH_FLOOR is
    classified `degenerate` (deterministic contrast, no genuine interval) instead of
    covered/not-covered. If ANY seed is degenerate the bias reports
    interval_coverage="degenerate_deterministic" + n_degenerate + exact_recovery (max over the
    degenerate seeds of |truth - point estimate|, return units -- ~1e-17 == exact recovery at
    machine precision for meas_err)."""
    covered, valid, degenerate = 0, 0, 0
    exact_recovery_diffs = []
    for s in range(seeds):
        truth = run_scenario(build_scenario(bias, DEFAULT_MAGNITUDES[bias], seed=s,
                                            spec=SyntheticSpec(noise_sd=0.0))).doe_first_order[bias]
        run = run_scenario(build_scenario(bias, DEFAULT_MAGNITUDES[bias], seed=s))
        # Use the full TOGGLE_IDS so the CI is the saturated-DOE main effect for `bias` -- the SAME
        # estimand as the noise-free truth (bit-identical to a singleton toggle on a single-bias
        # panel, but self-consistent with the pre-registered magnitude_sweep/run_scenario path).
        boot = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, n_replicates=replicates,
                             data_driven_block_months=6, min_effective_blocks=3, holding_period=1,
                             seed=2000 + s)
        lo, hi = boot.doe_ci()[frozenset({bias})]
        cls = _classify_coverage(lo, hi, truth)
        if cls == "degenerate":
            degenerate += 1
            exact_recovery_diffs.append(abs(truth - run.doe_first_order[bias]))
            continue
        valid += 1
        if cls == "covered":
            covered += 1
    if degenerate:
        return {"interval_coverage": "degenerate_deterministic", "n_degenerate": degenerate,
                "exact_recovery": max(exact_recovery_diffs),
                "coverage_n_seeds": seeds, "coverage_n_replicates": replicates}
    return {"interval_coverage": (covered / valid) if valid else None,
            "coverage_n_seeds": seeds, "coverage_n_replicates": replicates}


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_injection_grid(seeds: int = 20, replicates: int = 200) -> dict:
    grid = {}
    for bias in TOGGLE_IDS:
        injected_bp = _injected_bp(bias, seeds=3)
        sig = _signal_background(bias)
        rec_mde = _committed_recovered_mde(bias)
        cov = _coverage(bias, seeds, replicates)
        grid[bias] = {"injected_bp": injected_bp, "signal_background": sig, **rec_mde, **cov}
    return {"component": "rq3_injection_grid", "synthetic_dgp_only": True,
            "reads_real_data": False, "touches_holdout": False, "cost_usd": 0.0,
            "unit": "bp_per_month (effect x 10000; injection magnitude is DGP-internal, not bp)",
            "git_commit": _git_commit(), "per_correction": grid}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=25)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=_REPO_ROOT / "results" / "auditor" / "rq3_injection_grid.json")
    args = ap.parse_args(argv)
    result = run_injection_grid(seeds=args.seeds, replicates=args.replicates)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    for b, r in result["per_correction"].items():
        sg = r["signal_background"]
        cov = r["interval_coverage"]
        cov_s = cov if isinstance(cov, str) else f"{(cov or 0):.2f}"
        print(f"{b:14s} injected={r['injected_bp']:8.1f}bp recovered="
              f"{(r['recovered_bp'] or 0):8.1f}bp sig/bg={'PASS>=5' if sg['dominates_ge5'] else 'FAIL'}"
              f" cov={cov_s} mde="
              f"{('struct' if r['mde_structural'] else str(round(r['mde_bp'] or 0,1))+'bp')}")
    print(f"[rq3_injection_grid] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
