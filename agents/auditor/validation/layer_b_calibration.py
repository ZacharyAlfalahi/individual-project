"""
layer_b_calibration.py — Layer B statistical calibration (§10.2).

Recall shows the instrument responds (the engineering fixtures); calibration shows
it is CALIBRATED, and only calibration licenses RQ3's quantitative claims. This
module measures, over repeated seeds:

  * the FALSE-POSITIVE RATE under zero injection — the specificity evidence most
    likely to be dropped under deadline pressure, and the one that makes the
    specificity claim non-vacuous (§10.2). A false positive is a toggle whose
    bootstrap DOE interval excludes zero on a clean panel.
  * the MAGNITUDE-SWEEP response — the recovered effect as a function of the
    planted magnitude (monotone, correctly signed), with detection per seed.
  * the MINIMUM DETECTABLE EFFECT per bias — the smallest magnitude detected at a
    target rate.
  * end-to-end recovery of the three pre-registered INTERACTION mechanisms.

Everything runs on the metrics-only cheap path: one lattice per scenario, then the
bootstrap resamples the return matrix (never the engine).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..checks.bootstrap import run_bootstrap
from ..checks.shapley import shapley_result
from ..data.synthetic_panel import build_interaction_scenario, build_scenario
from ..schemas.toggle import TOGGLE_IDS, ToggleId
from .layer_b_fixtures import run_scenario

# Small defaults keep the calibration affordable in tests; the reported run scales
# these up (§14: bootstrap replicates are the first conditional cut, never the
# lattice or the injection calibration).
_BOOT = dict(
    n_replicates=200,
    data_driven_block_months=6,
    min_effective_blocks=3,
    holding_period=1,
)


def _bootstrap_for(scenario, *, seed, boot=None):
    """Run a scenario -> lattice -> bootstrap, returning (run, BootstrapResult)."""
    run = run_scenario(scenario)
    res = run_bootstrap(
        run.lattice.cells, run.common, TOGGLE_IDS, seed=seed, **(boot or _BOOT)
    )
    return run, res


def _detected(ci: tuple[float, float]) -> bool:
    lo, hi = ci
    return not (lo <= 0.0 <= hi)


# ---------------------------------------------------------------------------
# False-positive rate under zero injection (specificity)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FprReport:
    n_seeds: int
    alpha: float
    per_toggle_false_positives: dict[ToggleId, int]
    any_false_positive_seeds: int

    @property
    def family_wise_fpr(self) -> float:
        return self.any_false_positive_seeds / self.n_seeds if self.n_seeds else 0.0


def false_positive_rate(
    n_seeds: int = 8, *, alpha: float = 0.05, boot: dict | None = None
) -> FprReport:
    """Run zero-injection scenarios over `n_seeds` and count spurious detections
    (a toggle whose DOE CI excludes zero on a clean panel)."""
    per_toggle = {t: 0 for t in TOGGLE_IDS}
    any_fp = 0
    for s in range(n_seeds):
        scenario = build_scenario(None, seed=s)
        _, res = _bootstrap_for(scenario, seed=1000 + s, boot=boot)
        ci = res.doe_ci(alpha)
        seed_had_fp = False
        for t in TOGGLE_IDS:
            if _detected(ci[frozenset({t})]):
                per_toggle[t] += 1
                seed_had_fp = True
        any_fp += int(seed_had_fp)
    return FprReport(n_seeds, alpha, per_toggle, any_fp)


# ---------------------------------------------------------------------------
# Magnitude sweep + minimum detectable effect
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepRecord:
    bias: ToggleId
    magnitude: float
    seed: int
    effect: float
    ci: tuple[float, float]
    detected: bool


def magnitude_sweep(
    bias: ToggleId,
    magnitudes: list[float],
    n_seeds: int = 3,
    *,
    alpha: float = 0.05,
    boot: dict | None = None,
) -> list[SweepRecord]:
    """Recover `bias`'s first-order effect across planted magnitudes and seeds."""
    records: list[SweepRecord] = []
    for mag in magnitudes:
        for s in range(n_seeds):
            scenario = build_scenario(bias, mag, seed=s)
            run, res = _bootstrap_for(scenario, seed=2000 + s, boot=boot)
            effect = run.doe_first_order[bias]
            ci = res.doe_ci(alpha)[frozenset({bias})]
            records.append(SweepRecord(bias, mag, s, effect, ci, _detected(ci)))
    return records


def detection_rate(records: list[SweepRecord], magnitude: float) -> float:
    at_mag = [r for r in records if r.magnitude == magnitude]
    if not at_mag:
        return 0.0
    return sum(r.detected for r in at_mag) / len(at_mag)


def minimum_detectable_effect(
    records: list[SweepRecord], *, target: float = 0.8
) -> float | None:
    """The smallest swept magnitude whose detection rate reaches `target`."""
    mags = sorted({r.magnitude for r in records})
    for mag in mags:
        if detection_rate(records, mag) >= target:
            return mag
    return None


# ---------------------------------------------------------------------------
# Interaction mechanisms (end-to-end)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InteractionRecord:
    pair: tuple[str, str]
    interaction_effect: float
    interaction_ci: tuple[float, float]
    main_effects: dict[str, float]
    efficiency_residual_ok: bool


def interaction_recovery(
    pair: tuple[str, str],
    *,
    seed: int = 0,
    alpha: float = 0.05,
    boot: dict | None = None,
) -> InteractionRecord:
    """Run one interaction mechanism through the full lattice and recover the
    two-way DOE effect with its bootstrap interval, plus the two main effects."""
    scenario = build_interaction_scenario(pair, seed=seed)
    run, res = _bootstrap_for(scenario, seed=3000 + seed, boot=boot)
    a, b = pair
    inter_key = frozenset({a, b})
    inter = run.basis.doe[inter_key]
    inter_ci = res.doe_ci(alpha)[inter_key]
    mains = {a: run.basis.doe[frozenset({a})], b: run.basis.doe[frozenset({b})]}
    # shapley_result asserts efficiency (Σφ_i == Y(N)-Y(∅)) internally and raises
    # on violation; reaching here with a small residual confirms it end-to-end.
    shap = shapley_result(run.Y, TOGGLE_IDS, percentage_denominator_min=0.0)
    return InteractionRecord(
        pair=(a, b),
        interaction_effect=inter,
        interaction_ci=inter_ci,
        main_effects=mains,
        efficiency_residual_ok=bool(abs(shap.efficiency_residual) < 1e-9),
    )
