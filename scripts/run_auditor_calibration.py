"""
run_auditor_calibration.py — RQ3 item 6: run the Auditor's Layer-B statistical
calibration (confirmatory gate 11, design §10.2) at the PRE-REGISTERED REPORTED
bootstrap replicate count and record the artefacts.

This is the runner the calibration module has lacked: the four routines in
`agents/auditor/validation/layer_b_calibration.py` were previously exercised only
by `tests/unit/test_auditor_calibration.py` at CI-cheap sizes (n_replicates 120).
The reported run scales the bootstrap replicates up to the pre-registered count
committed to `docs/thresholds.yaml` (`auditor.bootstrap.n_replicates`, §14: the
bootstrap replicate count is the first conditional cut — NEVER the lattice or the
injection calibration, so the lattice, the seeds and the magnitude grid are held
fixed and only the replicate count is scaled).

Synthetic DGP only (`agents/auditor/data/synthetic_panel.py`). No real data, no
holdout. It does NOT modify `agents/auditor/validation/` or `agents/quant/library/`.

Determinism: every routine derives its per-scenario DGP seed and per-scenario
bootstrap seed internally and deterministically (build_scenario(seed=s);
bootstrap seed = 1000+s / 2000+s / 3000+seed), exactly as the unit tests rely on.
Given a fixed replicate count and fixed grids, the run is reproducible.

The four operating characteristics measured (§10.2):
  * false_positive_rate      — specificity under zero injection
  * magnitude_sweep + MDE    — monotone, correctly-signed recovery + minimum
                               detectable effect per bias
  * interaction_recovery     — end-to-end recovery of the 3 pre-registered pairs

A RED gate is a finding, not a bug to tune away: nothing here is nudged to pass.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # runnable as `python scripts/run_auditor_calibration.py`

from agents.auditor.data.synthetic_panel import (
    DEFAULT_MAGNITUDES,
    INTERACTION_MECHANISMS,
)
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.thresholds import load_bootstrap_config
from agents.auditor.validation.layer_b_calibration import (
    _BOOT,
    detection_rate,
    false_positive_rate,
    interaction_recovery,
    magnitude_sweep,
    minimum_detectable_effect,
)

ALPHA = 0.05

# Per-bias magnitude grid: 0.0 (specificity-within-sweep) + a low / default / high
# bracket anchored on each toggle's DEFAULT injection magnitude. The meas_err row
# reproduces the unit test's grid ([0.0, 0.01, 0.02, 0.04]) and the others
# generalise it (default, half-default, double-default).
MAGNITUDE_GRIDS: dict[str, list[float]] = {
    b: [0.0, round(m / 2, 6), m, round(m * 2, 6)]
    for b, m in DEFAULT_MAGNITUDES.items()
}

# Detection-rate target for the minimum-detectable-effect. 0.8 is the conventional
# power floor (the module's own default); 0.6 mirrors the unit test's looser probe
# and is reported alongside for context.
MDE_TARGET = 0.8
MDE_TARGET_LOOSE = 0.6

N_SEEDS_FPR = 8   # mirrors test_zero_injection_false_positive_rate_is_controlled
N_SEEDS_SWEEP = 3  # mirrors test_meas_err_recovery_is_monotone_and_signed

# Whether magnitude=0.0 is a GENUINE within-sweep null for each toggle. The sweep's
# `magnitude` parameter scales only ONE channel of each injector; a toggle is a real
# no-op at mag=0.0 iff that channel is the ENTIRE injection. This is a property of the
# injectors in synthetic_panel.py (verified empirically: at mag=0.0, meas_err is
# bit-identical to build_scenario(None), whereas stale_price still plants 1296 stale
# rows and survivorship still drops 477 rows + flags 18 defaults). For the STRUCTURAL
# injectors, "detection" at mag=0.0 is the instrument correctly flagging a real
# structural bias — NOT a false positive. The pre-registered specificity gate is the
# zero-injection FPR (metric 1, build_scenario(None)), which is genuinely clean.
MAGNITUDE_ZERO_IS_VALID_NULL: dict[str, tuple[bool, str]] = {
    "meas_err": (True,
        "pure return-channel injector (ret_raw += mag*q); mag=0.0 is bit-identical "
        "to the clean null, so the toggle is a genuine no-op."),
    "stale_price": (False,
        "magnitude scales only the return inflation; the staleness structure "
        "(last_trade_date shifted 45d on long-leg bonds) is magnitude-independent, so "
        "at mag=0.0 the stale-mask toggle still changes portfolio membership — a REAL "
        "structural effect, not a false positive. Specificity is gated by metric 1 (FPR)."),
    "survivorship": (False,
        "magnitude scales only the crater depth (-mag); the distress-exit flag and "
        "post-default row-dropping are magnitude-independent, so at mag=0.0 the "
        "survivorship toggle still has a small real effect. Specificity is gated by metric 1."),
    "lib_gap": (True,
        "at mag=0.0 the panel's return is pure noise uncorrelated with the score, so "
        "the signal-lag toggle captures nothing either way — a genuine no-op."),
    "lab_trim": (True,
        "at mag=0.0 the planted extreme is exactly -0.5 (AT the truncation bound, not "
        "beyond it), so trim on/off is a no-op."),
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _provenance() -> dict:
    short = _git("rev-parse", "--short", "HEAD")
    full = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    thresholds_path = REPO_ROOT / "docs" / "thresholds.yaml"
    thresholds_sha = hashlib.sha256(thresholds_path.read_bytes()).hexdigest()
    return {
        "git_commit_short": short,
        "git_commit_full": full,
        "working_tree_dirty": dirty,
        "thresholds_file": str(thresholds_path.relative_to(REPO_ROOT)),
        "thresholds_sha256": thresholds_sha,
    }


def _resolve_boot() -> tuple[dict, dict]:
    """The reported bootstrap config: the calibration module's own _BOOT (the
    synthetic-scale block/holding settings the module and tests use) with the
    replicate count scaled to the pre-registered reported value. We READ that
    value from thresholds.yaml and assert the runner is using it — so the number
    is the pre-registered constant, not a literal typed here."""
    pre = load_bootstrap_config()  # raises if auditor.bootstrap is unregistered
    boot = {**_BOOT, "n_replicates": pre.n_replicates}
    meta = {
        "n_replicates": pre.n_replicates,
        "n_replicates_source": "docs/thresholds.yaml:auditor.bootstrap.n_replicates",
        "module_default_n_replicates": _BOOT["n_replicates"],
        "data_driven_block_months": boot["data_driven_block_months"],
        "min_effective_blocks": boot["min_effective_blocks"],
        "holding_period": boot["holding_period"],
        "block_holding_source": (
            "agents/auditor/validation/layer_b_calibration.py:_BOOT "
            "(synthetic-scale; the 72-month DGP is far shorter than the ~240-month "
            "development window, so the module deliberately uses B_min=3 here, not "
            "the real-data auditor.bootstrap.min_effective_blocks=10)"
        ),
        "preregistered_bootstrap_block": {
            "n_replicates": pre.n_replicates,
            "min_effective_blocks": pre.min_effective_blocks,
            "block_length_months": pre.block_length_months,
        },
    }
    return boot, meta


# ---------------------------------------------------------------------------
# Metric 1 — false-positive rate under zero injection (specificity)
# ---------------------------------------------------------------------------

def run_fpr(boot: dict) -> dict:
    rep = false_positive_rate(n_seeds=N_SEEDS_FPR, alpha=ALPHA, boot=boot)
    per_toggle = {t: rep.per_toggle_false_positives[t] for t in TOGGLE_IDS}
    per_toggle_rate = {t: per_toggle[t] / rep.n_seeds for t in TOGGLE_IDS}
    # Targets (§10.2): per-toggle FPR should sit near nominal alpha; the test gate
    # is per-toggle count <= n_seeds//2 AND family-wise <= 0.5. A clean panel that
    # recovers ~0 on common support should give FPR ~0.
    no_toggle_majority = all(c <= rep.n_seeds // 2 for c in per_toggle.values())
    family_wise_ok = rep.family_wise_fpr <= 0.5
    per_toggle_near_nominal = all(r <= ALPHA + 1e-9 or c == 0
                                  for r, c in zip(per_toggle_rate.values(),
                                                  per_toggle.values()))
    return {
        "n_seeds": rep.n_seeds,
        "alpha_nominal": rep.alpha,
        "per_toggle_false_positive_count": per_toggle,
        "per_toggle_false_positive_rate": per_toggle_rate,
        "any_false_positive_seeds": rep.any_false_positive_seeds,
        "family_wise_fpr": rep.family_wise_fpr,
        "targets": {
            "family_wise_fpr_leq_0.5": family_wise_ok,
            "no_single_toggle_on_majority_of_seeds": no_toggle_majority,
            "per_toggle_rate_at_or_below_nominal_alpha": per_toggle_near_nominal,
        },
        "pass": bool(family_wise_ok and no_toggle_majority),
    }


# ---------------------------------------------------------------------------
# Metric 2/3 — magnitude sweep + minimum detectable effect (per bias)
# ---------------------------------------------------------------------------

def run_sweep(boot: dict) -> dict:
    out: dict[str, dict] = {}
    for bias in TOGGLE_IDS:
        grid = MAGNITUDE_GRIDS[bias]
        records = magnitude_sweep(
            bias, grid, n_seeds=N_SEEDS_SWEEP, alpha=ALPHA, boot=boot
        )
        per_mag = {}
        for mag in grid:
            rs = [r for r in records if r.magnitude == mag]
            mean_eff = sum(r.effect for r in rs) / len(rs)
            mean_abs = sum(abs(r.effect) for r in rs) / len(rs)
            per_mag[str(mag)] = {
                "mean_effect": mean_eff,
                "mean_abs_effect": mean_abs,
                "detection_rate": detection_rate(records, mag),
                "n_seeds": len(rs),
            }
        nonzero = [m for m in grid if m > 0.0]
        abs_by_mag = [per_mag[str(m)]["mean_abs_effect"] for m in nonzero]
        monotone = all(a < b for a, b in zip(abs_by_mag, abs_by_mag[1:]))
        # Sign consistency: the sign of the recovered effect at the largest
        # magnitude should be shared by every seed at that magnitude.
        top = max(grid)
        top_signs = {(-1 if r.effect < 0 else 1)
                     for r in records if r.magnitude == top}
        signed = len(top_signs) == 1
        zero_det = detection_rate(records, 0.0)
        mde = minimum_detectable_effect(records, target=MDE_TARGET)
        mde_loose = minimum_detectable_effect(records, target=MDE_TARGET_LOOSE)
        default_mag = DEFAULT_MAGNITUDES[bias]
        valid_null, null_reason = MAGNITUDE_ZERO_IS_VALID_NULL[bias]
        mde_ok = mde is not None and mde <= default_mag + 1e-12
        # The within-sweep zero-magnitude check is a specificity probe ONLY where
        # mag=0.0 is a genuine null (return-channel injectors). For the structural
        # injectors it is not applicable — specificity is gated by metric 1 (FPR).
        zero_null_ok = (zero_det <= 0.5) if valid_null else None
        targets = {
            "monotone_growth": monotone,
            "correctly_signed": signed,
            "mde_exists_at_target_0.8": mde is not None,
            "mde_at_or_below_default_magnitude": mde_ok,
            "within_sweep_zero_null_leq_0.5": zero_null_ok,  # None = not applicable
        }
        # Pre-registered per-bias pass = monotone + correctly signed + MDE at/below the
        # default injection magnitude, PLUS the within-sweep zero null where applicable.
        bias_pass = monotone and signed and mde_ok and (zero_null_ok is not False)
        out[bias] = {
            "grid": grid,
            "default_magnitude": default_mag,
            "per_magnitude": per_mag,
            "monotone_abs_effect": monotone,
            "sign_consistent_at_max": signed,
            "zero_injection_detection_rate": zero_det,   # raw, always reported
            "magnitude_zero_is_valid_null": valid_null,
            "magnitude_zero_null_reason": null_reason,
            "mde_target": MDE_TARGET,
            "mde": mde,
            "mde_loose_target": MDE_TARGET_LOOSE,
            "mde_loose": mde_loose,
            "targets": targets,
            "pass": bool(bias_pass),
        }
    overall = all(v["pass"] for v in out.values())
    return {"per_bias": out, "pass": bool(overall)}


# ---------------------------------------------------------------------------
# Metric 4 — end-to-end interaction recovery (the 3 pre-registered mechanisms)
# ---------------------------------------------------------------------------

def run_interactions(boot: dict) -> dict:
    out = []
    for pair in INTERACTION_MECHANISMS:
        rec = interaction_recovery(pair, seed=0, alpha=ALPHA, boot=boot)
        d = asdict(rec)
        d["pair"] = list(rec.pair)
        d["interaction_ci"] = list(rec.interaction_ci)
        mains_present = all(abs(v) > 1e-4 for v in rec.main_effects.values())
        lo, hi = rec.interaction_ci
        finite_interval = lo <= hi
        d["targets"] = {
            "both_main_effects_present": mains_present,
            "efficiency_residual_ok": rec.efficiency_residual_ok,
            "finite_interaction_interval": finite_interval,
        }
        d["pass"] = bool(
            mains_present and rec.efficiency_residual_ok and finite_interval
        )
        out.append(d)
    return {"mechanisms": out, "pass": bool(all(m["pass"] for m in out))}


def main() -> None:
    prov = _provenance()
    boot, boot_meta = _resolve_boot()

    print(f"[calibration] commit {prov['git_commit_short']} "
          f"(dirty={prov['working_tree_dirty']})")
    print(f"[calibration] n_replicates={boot_meta['n_replicates']} "
          f"(source: {boot_meta['n_replicates_source']})")
    print(f"[calibration] boot={boot}")

    t0 = time.time()
    print("[calibration] 1/4 false-positive rate (zero injection) ...")
    fpr = run_fpr(boot)
    t_fpr = time.time()
    print(f"           ... {t_fpr - t0:.1f}s  family_wise_fpr={fpr['family_wise_fpr']}")

    print("[calibration] 2/4 magnitude sweep + MDE (5 biases) ...")
    sweep = run_sweep(boot)
    t_sweep = time.time()
    print(f"           ... {t_sweep - t_fpr:.1f}s")

    print("[calibration] 3/4 interaction recovery (3 mechanisms) ...")
    inter = run_interactions(boot)
    t_inter = time.time()
    print(f"           ... {t_inter - t_sweep:.1f}s")

    wall = time.time() - t0
    all_pass = fpr["pass"] and sweep["pass"] and inter["pass"]

    payload = {
        "component": "auditor.layer_b_calibration",
        "gate": "confirmatory gate 11 (design §10.2)",
        "synthetic_dgp_only": True,
        "notes": {
            "specificity_gate": (
                "The pre-registered specificity target is the zero-injection FPR "
                "(metric 1, build_scenario(None) — a genuinely clean panel). The "
                "within-sweep zero-magnitude detection is a SECONDARY probe, valid "
                "only for return-channel injectors (meas_err, lib_gap, lab_trim). For "
                "the structural injectors (stale_price, survivorship) the magnitude "
                "parameter scales only the return channel, so mag=0.0 is NOT a null and "
                "a 'detection' there is a true positive on a real structural bias — see "
                "each bias's magnitude_zero_null_reason."
            ),
            "nothing_tuned": (
                "No DGP, magnitude, replicate count, or instrument parameter was "
                "adjusted to pass. Raw zero-injection detection rates are reported "
                "verbatim for every bias."
            ),
        },
        "provenance": prov,
        "bootstrap": boot_meta,
        "alpha": ALPHA,
        "n_seeds": {"fpr": N_SEEDS_FPR, "magnitude_sweep": N_SEEDS_SWEEP,
                    "interaction": 1},
        "wall_clock_seconds": {
            "false_positive_rate": t_fpr - t0,
            "magnitude_sweep": t_sweep - t_fpr,
            "interaction_recovery": t_inter - t_sweep,
            "total": wall,
        },
        "false_positive_rate": fpr,
        "magnitude_sweep": sweep,
        "interaction_recovery": inter,
        "all_gates_pass": bool(all_pass),
    }

    out_dir = REPO_ROOT / "results" / "auditor" / "calibration" / f"run_{prov['git_commit_short']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))

    print(f"[calibration] wall-clock total {wall:.1f}s")
    print(f"[calibration] FPR pass={fpr['pass']}  "
          f"sweep pass={sweep['pass']}  interaction pass={inter['pass']}")
    print(f"[calibration] ALL GATES PASS = {all_pass}")
    print(f"[calibration] wrote {out_path}")


if __name__ == "__main__":
    main()
