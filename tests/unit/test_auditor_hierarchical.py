"""Inc2-F — the hierarchical prevalence model (§8.2)."""

from __future__ import annotations

import numpy as np

from agents.auditor.checks.hierarchical import (
    HierarchyPriors,
    StrategyEffect,
    StrategyVector,
    fit_coordinate_hierarchy,
    fit_hierarchy,
    fit_joint_hierarchy,
)

PRIORS = HierarchyPriors(mu_scale=1.0, tau_scale=0.1)


def _effects(values, var=1e-4, no_ops=0):
    out = [StrategyEffect(f"s{i}", v, var, False) for i, v in enumerate(values)]
    for k in range(no_ops):
        out.append(StrategyEffect(f"noop{k}", 0.0, 1e-12, True))
    return out


# --------------------------------------------------------------------------
# Magnitude recovery
# --------------------------------------------------------------------------

def test_mu_recovers_common_mean():
    # All strategies near 0.05 => μ posterior near 0.05.
    eff = _effects([0.048, 0.052, 0.05, 0.049, 0.051])
    res = fit_coordinate_hierarchy(eff, vartheta=0.01, priors=PRIORS, seed=1)
    assert 0.03 < res.mu_mean < 0.07
    assert res.mu_ci_low < res.mu_mean < res.mu_ci_high


def test_wide_tau_when_effects_heterogeneous():
    homogeneous = fit_coordinate_hierarchy(
        _effects([0.05, 0.05, 0.05, 0.05, 0.05]), vartheta=0.01, priors=PRIORS, seed=2
    )
    heterogeneous = fit_coordinate_hierarchy(
        _effects([-0.1, 0.2, 0.0, 0.15, -0.05]), vartheta=0.01, priors=PRIORS, seed=2
    )
    assert heterogeneous.tau_mean > homogeneous.tau_mean


# --------------------------------------------------------------------------
# Prevalence denominator (D-A31): no-ops count in the denominator, not the numerator
# --------------------------------------------------------------------------

def test_prevalence_large_when_all_susceptible_are_material():
    # 5 strategies all with a big effect vs ϑ => prevalence near 1.
    res = fit_coordinate_hierarchy(
        _effects([0.3, 0.3, 0.3, 0.3, 0.3], var=1e-5), vartheta=0.05, priors=PRIORS, seed=3
    )
    assert res.prevalence > 0.9
    assert res.n_runnable == 5 and res.n_susceptible == 5


def test_no_ops_inflate_denominator_not_numerator():
    # 1 material strategy + 4 proven no-ops => prevalence ≈ 1/5 = 0.2 (D-A31 limit case).
    eff = _effects([0.3], var=1e-6, no_ops=4)
    res = fit_coordinate_hierarchy(eff, vartheta=0.05, priors=PRIORS, seed=4)
    assert res.n_runnable == 5 and res.n_susceptible == 1
    assert abs(res.prevalence - 0.2) < 0.08  # ~0.2, not ~1.0


def test_all_no_ops_gives_zero_prevalence():
    eff = _effects([], no_ops=3)
    res = fit_coordinate_hierarchy(eff, vartheta=0.05, priors=PRIORS, seed=5)
    assert res.prevalence == 0.0
    assert res.n_susceptible == 0


def test_prevalence_small_when_effects_below_threshold():
    # Effects all well below ϑ => low prevalence.
    res = fit_coordinate_hierarchy(
        _effects([0.001, 0.002, 0.0, 0.001, -0.001], var=1e-8),
        vartheta=0.05, priors=PRIORS, seed=6,
    )
    assert res.prevalence < 0.3


# --------------------------------------------------------------------------
# Corpus wrapper + serialisation
# --------------------------------------------------------------------------

def test_fit_hierarchy_over_multiple_coordinates():
    corpus = {
        "meas_err": _effects([0.1, 0.12, 0.11, 0.09, 0.1], var=1e-5),
        "lib_gap": _effects([0.0, 0.001, 0.0, 0.0, 0.0], var=1e-8),
    }
    out = fit_hierarchy(corpus, vartheta=0.05, priors=PRIORS, seed=0)
    assert set(out) == {"meas_err", "lib_gap"}
    assert out["meas_err"].prevalence > out["lib_gap"].prevalence
    d = out["meas_err"].to_dict()
    assert "conditional on susceptibility" in d["label_note"]


# --------------------------------------------------------------------------
# fit_joint_hierarchy — the FULL measurement covariance (D-A32)
# --------------------------------------------------------------------------

EPS = 1e-8  # eigenvalue floor (threshold-sourced in production; explicit in tests)


def _diag_vectors(values, var=1e-4, no_ops=0, coord="c0"):
    """One-coordinate StrategyVectors with a diagonal (here 1×1) measurement covariance —
    the joint model must reduce to the scalar model on this input."""
    out = [StrategyVector(f"s{i}", (coord,), np.array([v], float),
                          np.array([[var]], float), np.array([False]))
           for i, v in enumerate(values)]
    for k in range(no_ops):
        out.append(StrategyVector(f"noop{k}", (coord,), np.array([0.0]),
                                  np.array([[1e-12]]), np.array([True])))
    return out


def _two_coord_vectors(est_pairs, cov):
    return [StrategyVector(f"s{i}", ("a", "b"), np.array(e, float),
                           np.array(cov, float), np.array([False, False]))
            for i, e in enumerate(est_pairs)]


def test_joint_reduces_to_scalar_on_diagonal_input():
    values = [0.30, 0.31, 0.29, 0.30, 0.30]
    scalar = fit_coordinate_hierarchy(_effects(values, var=1e-5), vartheta=0.05, priors=PRIORS, seed=1)
    joint = fit_joint_hierarchy(_diag_vectors(values, 1e-5),
                                vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=1).coordinates["c0"]
    assert joint.n_runnable == scalar.n_runnable == 5
    assert abs(joint.mu_mean - scalar.mu_mean) < 0.02
    assert abs(joint.prevalence - scalar.prevalence) < 0.05


def test_joint_no_ops_inflate_denominator_not_numerator():
    res = fit_joint_hierarchy(_diag_vectors([0.3], var=1e-6, no_ops=4),
                              vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=4).coordinates["c0"]
    assert res.n_runnable == 5 and res.n_susceptible == 1
    assert abs(res.prevalence - 0.2) < 0.08  # ~0.2, not ~1.0 (D-A31)


def test_joint_all_no_ops_zero_prevalence():
    res = fit_joint_hierarchy(_diag_vectors([], no_ops=3),
                              vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=5).coordinates["c0"]
    assert res.prevalence == 0.0 and res.n_susceptible == 0


def test_off_diagonal_covariance_actually_moves_the_estimate():
    # Same diagonal variances; only the off-diagonals differ. If V̂'s off-diagonals were
    # ignored (the scalar model), the two fits would be identical.
    ests = [[0.06, 0.05], [0.04, 0.06], [0.05, 0.04], [0.06, 0.05]]
    var = 4e-4
    full = [[var, 0.9 * var], [0.9 * var, var]]
    diag = [[var, 0.0], [0.0, var]]
    r_full = fit_joint_hierarchy(_two_coord_vectors(ests, full),
                                 vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=7).coordinates
    r_diag = fit_joint_hierarchy(_two_coord_vectors(ests, diag),
                                 vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=7).coordinates
    moved = (abs(r_full["a"].mu_mean - r_diag["a"].mu_mean) > 1e-6
             or abs(r_full["a"].prevalence - r_diag["a"].prevalence) > 1e-6)
    assert moved


def test_joint_handles_variable_k_strategies():
    # s1's coordinate `b` is structurally absent (stripped upstream, NOT a no-op), so it
    # must lower b's denominator without ever being counted for s1.
    s0 = StrategyVector("s0", ("a", "b"), np.array([0.2, 0.2]),
                        np.eye(2) * 1e-4, np.array([False, False]))
    s1 = StrategyVector("s1", ("a",), np.array([0.2]),
                        np.array([[1e-4]]), np.array([False]))
    out = fit_joint_hierarchy([s0, s1], vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=2).coordinates
    assert out["a"].n_runnable == 2 and out["a"].n_susceptible == 2
    assert out["b"].n_runnable == 1 and out["b"].n_susceptible == 1


def test_joint_is_deterministic_by_seed():
    svs = _diag_vectors([0.048, 0.052, 0.05, 0.049, 0.051], 1e-4)
    a = fit_joint_hierarchy(svs, vartheta=0.01, epsilon=EPS, priors=PRIORS, seed=3).coordinates["c0"]
    b = fit_joint_hierarchy(svs, vartheta=0.01, epsilon=EPS, priors=PRIORS, seed=3).coordinates["c0"]
    assert a.mu_mean == b.mu_mean and a.prevalence == b.prevalence


def test_prevalence_sweep_matches_headline_and_is_monotone():
    grid = (0.0005, 0.001, 0.0015, 0.002)
    svs = _diag_vectors([0.05, 0.06, 0.05, 0.055, 0.05], var=1e-5)
    res = fit_joint_hierarchy(svs, vartheta=0.001, epsilon=EPS, priors=PRIORS, seed=3,
                              vartheta_grid=grid).coordinates["c0"]
    sweep = dict(res.prevalence_sweep)
    assert set(sweep) == set(grid)
    assert abs(sweep[0.001] - res.prevalence) < 1e-12       # headline entry == primary π̃
    vals = [sweep[v] for v in grid]
    assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1))  # non-increasing
    assert "prevalence_sweep" in res.to_dict()


def test_scalar_prevalence_sweep_matches_headline():
    grid = (0.0005, 0.001, 0.0015, 0.002)
    res = fit_coordinate_hierarchy(_effects([0.05, 0.06, 0.05, 0.055, 0.05], var=1e-5),
                                   vartheta=0.001, priors=PRIORS, seed=3, vartheta_grid=grid)
    sweep = dict(res.prevalence_sweep)
    assert set(sweep) == set(grid)
    assert abs(sweep[0.001] - res.prevalence) < 1e-12


def test_prevalence_sweep_empty_without_grid():
    res = fit_joint_hierarchy(_diag_vectors([0.05, 0.05, 0.05]), vartheta=0.001,
                              epsilon=EPS, priors=PRIORS, seed=1).coordinates["c0"]
    assert res.prevalence_sweep == ()
    assert "prevalence_sweep" not in res.to_dict()


def test_ill_conditioned_covariance_is_floored_and_recorded():
    # A near-collinear V̂ (two co-moving corrections). The pre-registered eigenvalue floor
    # (D-A34) must clip and RECORD it — not silently invert an ill-conditioned matrix — and
    # still yield a finite fit. Without the floor, np.linalg.inv would return an unbounded
    # precision and over-certain effects.
    var = 4e-4
    near_singular = [[var, 0.999999 * var], [0.999999 * var, var]]
    ests = [[0.06, 0.06], [0.05, 0.05], [0.055, 0.055], [0.05, 0.05]]
    fit = fit_joint_hierarchy(_two_coord_vectors(ests, near_singular),
                              vartheta=0.05, epsilon=EPS, priors=PRIORS, seed=8)
    # every strategy's regularisation is recorded, and at least one eigenvalue was clipped.
    assert set(fit.regularisations) == {"s0", "s1", "s2", "s3"}
    assert any(r.n_eigenvalues_clipped >= 1 for r in fit.regularisations.values())
    for coord in ("a", "b"):
        h = fit.coordinates[coord]
        assert np.isfinite(h.mu_mean) and np.isfinite(h.tau_mean) and np.isfinite(h.prevalence)
