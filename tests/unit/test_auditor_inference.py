"""Inc2-A — stats primitives and inference routing (§7.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agents.auditor.checks.bootstrap import run_bootstrap
from agents.auditor.checks.inference import MEAN_METRICS, infer_doe_effects
from agents.auditor.checks.stats import normal_cdf, normal_ppf, two_sided_p
from agents.auditor.confirmatory import confirmatory_coordinates
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.validation.layer_b_fixtures import run_scenario

_BOOT = dict(
    n_replicates=200, data_driven_block_months=6,
    min_effective_blocks=3, holding_period=1,
)


# --------------------------------------------------------------------------
# stats primitives
# --------------------------------------------------------------------------

def test_normal_cdf_known_values():
    assert math.isclose(normal_cdf(0.0), 0.5, abs_tol=1e-12)
    assert math.isclose(normal_cdf(1.96), 0.975, abs_tol=1e-3)
    assert math.isclose(normal_cdf(-1.96), 0.025, abs_tol=1e-3)


def test_normal_ppf_inverts_cdf():
    for p in (0.01, 0.1, 0.5, 0.9, 0.975, 0.999):
        assert math.isclose(normal_cdf(normal_ppf(p)), p, abs_tol=1e-6)


def test_normal_ppf_domain():
    with pytest.raises(ValueError):
        normal_ppf(0.0)


def test_two_sided_p_of_196_is_005():
    assert math.isclose(two_sided_p(1.96), 0.05, abs_tol=1e-3)
    assert two_sided_p(0.0) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Routing: HAC for the mean metric, bootstrap for a nonlinear metric
# --------------------------------------------------------------------------

def _run(bias, metric, seed=0):
    scenario = build_scenario(bias, seed=seed)
    run = run_scenario(scenario, metric=metric)
    res = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, seed=5, **_BOOT)
    coords = confirmatory_coordinates(TOGGLE_IDS)
    return infer_doe_effects(
        run.lattice.cells, run.common, TOGGLE_IDS, res,
        metric=metric, coordinates=coords,
    )


def test_mean_metric_routes_to_hac():
    inf = _run("meas_err", "average")
    r = inf[frozenset({"meas_err"})]
    assert r.method == "HAC"
    assert r.t_stat is not None and r.n_obs is not None


def test_nonlinear_metric_routes_to_bootstrap():
    inf = _run("meas_err", "sharpe")
    r = inf[frozenset({"meas_err"})]
    assert r.method == "bootstrap"
    assert r.t_stat is None


def test_confirmatory_coordinates_shape():
    coords = confirmatory_coordinates(TOGGLE_IDS)
    singletons = [c for c in coords if len(c) == 1]
    pairs = [c for c in coords if len(c) == 2]
    assert len(singletons) == 5 and len(pairs) == 3  # 5 main + 3 validated pairs


def test_reduced_lattice_drops_coordinates_needing_absent_toggles():
    # survivorship non-runnable => its main effect and both interactions containing
    # it (surv×meas) drop; meas×stale and lib×trim survive.
    coords = confirmatory_coordinates(("meas_err", "stale_price", "lib_gap", "lab_trim"))
    assert frozenset({"survivorship"}) not in coords
    assert frozenset({"survivorship", "meas_err"}) not in coords
    assert frozenset({"meas_err", "stale_price"}) in coords


def test_injected_effect_is_significant_under_hac():
    inf = _run("meas_err", "average", seed=1)
    r = inf[frozenset({"meas_err"})]
    # a planted effect should be detected: small p, CI excludes 0
    assert r.p_value < 0.05
    assert not (r.ci_low <= 0.0 <= r.ci_high)


def test_hac_point_equals_mean_of_monthly_doe_series():
    # The HAC point estimate must equal the common-support DOE effect exactly.
    scenario = build_scenario("stale_price", seed=2)
    run = run_scenario(scenario, metric="average")
    res = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, seed=1, **_BOOT)
    inf = infer_doe_effects(
        run.lattice.cells, run.common, TOGGLE_IDS, res, metric="average",
        coordinates=[frozenset({"stale_price"})],
    )
    r = inf[frozenset({"stale_price"})]
    assert np.isclose(r.point, run.basis.doe[frozenset({"stale_price"})])


def test_mean_metrics_set_contents():
    assert "average" in MEAN_METRICS and "sharpe" not in MEAN_METRICS


# --------------------------------------------------------------------------
# inert coordinates (identically-zero contrasts are not tested)
# --------------------------------------------------------------------------

def _doctored_cells_with_inert_stale_price(seed: int):
    """The meas_err scenario with stale_price made an exact no-op: every cell with stale_price ON carries
    the returns of its stale_price-OFF twin, so every DOE contrast involving stale_price is zero up to
    round-off — the exact no-op case an invariance gate checks for."""
    import dataclasses
    scenario = build_scenario("meas_err", seed=seed)
    run = run_scenario(scenario, metric="average")
    by_key = {c.on_set: c for c in run.lattice.cells}
    cells = [dataclasses.replace(c, returns=by_key[c.on_set - {"stale_price"}].returns)
             if "stale_price" in c.on_set else c for c in run.lattice.cells]
    return cells, run.common


def test_inert_coordinate_is_not_tested_under_hac():
    cells, common = _doctored_cells_with_inert_stale_price(seed=3)
    res = run_bootstrap(cells, common, TOGGLE_IDS, seed=1, **_BOOT)
    coords = [frozenset({"stale_price"}), frozenset({"meas_err"}), frozenset({"meas_err", "stale_price"})]
    inf = infer_doe_effects(cells, common, TOGGLE_IDS, res, metric="average", coordinates=coords)
    sp = inf[frozenset({"stale_price"})]
    assert sp.inert and sp.p_value == 1.0 and sp.t_stat == 0.0 and abs(sp.point) < 1e-12
    assert inf[frozenset({"meas_err", "stale_price"})].inert
    me = inf[frozenset({"meas_err"})]                      # the planted effect is still detected
    assert not me.inert and me.p_value < 0.05 and me.t_stat != 0.0


def test_inert_coordinate_is_not_tested_under_bootstrap():
    cells, common = _doctored_cells_with_inert_stale_price(seed=4)
    res = run_bootstrap(cells, common, TOGGLE_IDS, seed=1, **_BOOT)
    inf = infer_doe_effects(cells, common, TOGGLE_IDS, res, metric="sharpe",
                            coordinates=[frozenset({"stale_price"}), frozenset({"meas_err"})])
    assert inf[frozenset({"stale_price"})].inert and inf[frozenset({"stale_price"})].p_value == 1.0
    assert not inf[frozenset({"meas_err"})].inert


def test_guard_leaves_every_non_inert_coordinate_untouched():
    # Every non-inert HAC coordinate keeps the two-sided normal p of its t; every inert one has p = 1, t = 0;
    # the planted effect is never inert.
    inf = _run("meas_err", "average", seed=1)
    assert not inf[frozenset({"meas_err"})].inert
    for r in inf.values():
        if r.inert:
            assert r.p_value == 1.0 and r.t_stat == 0.0
        else:
            assert math.isclose(r.p_value, two_sided_p(r.t_stat), rel_tol=0, abs_tol=1e-15)
