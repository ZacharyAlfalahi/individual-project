"""Inc2-D — the compression-adequacy statistic D (§8.1)."""

from __future__ import annotations

import math

from agents.auditor.checks.bootstrap import run_bootstrap
from agents.auditor.checks.compression import (
    compression_statistic,
    run_compression,
)
from agents.auditor.checks.support import primary_metric_vector
from agents.auditor.checks.algebra import saturated_basis
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.validation.layer_b_fixtures import run_scenario

_BOOT = dict(
    n_replicates=150, data_driven_block_months=6,
    min_effective_blocks=3, holding_period=1,
)


# --------------------------------------------------------------------------
# The statistic on hand-built DOE maps
# --------------------------------------------------------------------------

def test_D_zero_when_only_first_and_second_order():
    doe = {
        frozenset(): 0.0,
        frozenset({"a"}): 0.5, frozenset({"b"}): 0.3,
        frozenset({"a", "b"}): 0.2,
    }
    assert compression_statistic(doe) == 0.0


def test_D_positive_with_third_order_energy():
    doe = {
        frozenset({"a"}): 1.0, frozenset({"b"}): 0.0, frozenset({"c"}): 0.0,
        frozenset({"a", "b", "c"}): 1.0,  # equal 1st and 3rd order energy
    }
    # num = 1^2, den = 1^2 + 1^2 = 2 => D = 0.5
    assert math.isclose(compression_statistic(doe), 0.5)


def test_D_zero_when_no_effect_energy():
    doe = {frozenset(): 3.0, frozenset({"a"}): 0.0, frozenset({"a", "b", "c"}): 0.0}
    assert compression_statistic(doe) == 0.0  # denom below floor => 0


# --------------------------------------------------------------------------
# Bootstrap distribution + verdict on a real lattice
# --------------------------------------------------------------------------

def test_clean_panel_compression_is_adequate():
    scenario = build_scenario(None, seed=0)
    run = run_scenario(scenario, metric="average")
    res = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, seed=1, **_BOOT)
    Y = primary_metric_vector(run.lattice.cells, run.common, "average")
    doe = saturated_basis(Y, TOGGLE_IDS).doe
    comp = run_compression(doe, res, d_max=0.2)
    # clean panel => negligible higher-order energy => D small, adequate
    assert comp.d_point < 0.2
    assert comp.adequate
    d = comp.to_dict()
    assert d["compression_adequate"] is True


def test_verdict_flips_when_d_max_below_point():
    scenario = build_scenario("meas_err", seed=1)
    run = run_scenario(scenario, metric="average")
    res = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, seed=1, **_BOOT)
    Y = primary_metric_vector(run.lattice.cells, run.common, "average")
    doe = saturated_basis(Y, TOGGLE_IDS).doe
    # an impossibly strict D_max (negative) can never be met => not adequate
    comp = run_compression(doe, res, d_max=-1.0)
    assert not comp.adequate
