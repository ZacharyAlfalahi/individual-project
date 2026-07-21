"""
layer_b_fixtures.py — Layer B engineering fixtures (§10.2).

End-to-end recovery: inject a KNOWN single bias into a clean synthetic panel, run
the FULL 2^k lattice, decompose on common support, and assert the injected toggle's
first-order DOE effect FIRES (is clearly nonzero and dominates the others). This is
an ENGINEERING GATE — it shows each check responds. It does NOT license the
quantitative claims; that is the statistical calibration (Layer B, step 11).

`run_scenario` is the shared driver used by both the fixtures and the calibration
sweep: preflight (all toggles runnable) -> lattice -> common support -> saturated
decomposition -> first-order DOE effect per toggle.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..checks.algebra import saturated_basis
from ..checks.lattice import run_lattice
from ..checks.preflight import derive_scope
from ..checks.support import common_support, primary_metric_vector
from ..data.synthetic_panel import Scenario, build_scenario
from ..schemas.decomposition import SaturatedBasis
from ..schemas.lattice_types import LatticeResult
from ..schemas.toggle import TOGGLE_IDS, ToggleFacts, ToggleId


@dataclass(frozen=True, eq=False)
class ScenarioRun:
    lattice: LatticeResult
    common: pd.DatetimeIndex
    Y: dict
    basis: SaturatedBasis
    doe_first_order: dict[ToggleId, float]


def run_scenario(
    scenario: Scenario,
    *,
    metric: str = "average",
    lib_gap_lags: tuple[int, int] = (0, 1),
) -> ScenarioRun:
    """Run one scenario through the full instrument spine and return the
    first-order DOE effects on common support."""
    facts = [ToggleFacts(t, runnable=True) for t in TOGGLE_IDS]
    pf = derive_scope(scenario.strategy.strategy_label, facts)
    lat = run_lattice(
        scenario.strategy,
        pf.runnable_toggles,
        pf.fixed_states,
        scenario.panel,
        signals=scenario.signals,
        lib_gap_lags=lib_gap_lags,
    )
    common = common_support(lat.cells)
    Y = primary_metric_vector(lat.cells, common, metric)
    basis = saturated_basis(Y, pf.runnable_toggles)
    doe1 = {t: basis.doe[frozenset({t})] for t in pf.runnable_toggles}
    return ScenarioRun(lattice=lat, common=common, Y=Y, basis=basis, doe_first_order=doe1)


@dataclass(frozen=True)
class FixtureVerdict:
    bias: ToggleId
    injected_effect: float
    max_other_effect: float
    fired: bool
    dominates: bool


def single_bias_fixture(
    bias: ToggleId,
    *,
    magnitude: float | None = None,
    seed: int = 0,
    metric: str = "average",
    fire_threshold: float = 1e-3,
    dominance_ratio: float = 5.0,
) -> FixtureVerdict:
    """Inject `bias` alone and verify its first-order effect fires and dominates.

    `fired`      = |E_bias| exceeds `fire_threshold`.
    `dominates`  = |E_bias| exceeds `dominance_ratio` x the largest un-injected effect.
    """
    scenario = build_scenario(bias, magnitude, seed=seed)
    run = run_scenario(scenario, metric=metric)
    injected = abs(run.doe_first_order[bias])
    others = [abs(v) for t, v in run.doe_first_order.items() if t != bias]
    max_other = max(others) if others else 0.0
    return FixtureVerdict(
        bias=bias,
        injected_effect=injected,
        max_other_effect=max_other,
        fired=injected > fire_threshold,
        dominates=injected > dominance_ratio * max_other,
    )
