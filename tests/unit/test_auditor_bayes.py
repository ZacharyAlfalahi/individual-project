"""Inc2-C — Bayesian normal approximation (§7.3)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.auditor.checks.bayes import (
    assemble_theta,
    regularise_covariance,
    run_bayes,
)
from agents.auditor.checks.bootstrap import run_bootstrap
from agents.auditor.checks.support import primary_metric_vector
from agents.auditor.checks.algebra import saturated_basis
from agents.auditor.confirmatory import confirmatory_coordinates
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.validation.layer_b_fixtures import run_scenario

_BOOT = dict(
    n_replicates=300, data_driven_block_months=6,
    min_effective_blocks=3, holding_period=1,
)


# --------------------------------------------------------------------------
# Covariance regularisation (§7.3.3)
# --------------------------------------------------------------------------

def test_regularisation_records_clipped_eigenvalues():
    # A rank-deficient covariance (one zero eigenvalue) must be clipped and recorded.
    V = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular
    V_reg, V_inv, reg = regularise_covariance(V, epsilon=1e-6)
    assert reg.n_eigenvalues_clipped == 1
    assert reg.min_eigenvalue_before < 1e-9
    # regularised matrix is now positive definite
    assert np.linalg.eigvalsh(V_reg).min() >= 1e-6 - 1e-12
    # V_inv is finite and a genuine inverse of V_reg (no unguarded inversion)
    assert np.all(np.isfinite(V_inv))
    assert np.allclose(V_reg @ V_inv, np.eye(2), atol=1e-8)


def test_regularisation_clips_at_epsilon_boundary():
    # An eigenvalue exactly == epsilon must be treated as clipped (<=), so V_inv
    # stays finite even at the boundary (the S3 zero-eigenvalue-at-epsilon=0 trap).
    V_reg, V_inv, reg = regularise_covariance(np.diag([0.0, 2.0]), epsilon=1e-9)
    assert reg.n_eigenvalues_clipped == 1
    assert np.all(np.isfinite(V_inv))


def test_regularisation_leaves_pd_matrix_alone():
    V = np.diag([0.5, 0.3])
    _, _, reg = regularise_covariance(V, epsilon=1e-8)
    assert reg.n_eigenvalues_clipped == 0


# --------------------------------------------------------------------------
# Conjugate posterior behaviour
# --------------------------------------------------------------------------

def _theta_and_draws(bias, seed=0):
    scenario = build_scenario(bias, seed=seed)
    run = run_scenario(scenario, metric="average")
    res = run_bootstrap(run.lattice.cells, run.common, TOGGLE_IDS, seed=3, **_BOOT)
    coords = confirmatory_coordinates(TOGGLE_IDS)
    Y = primary_metric_vector(run.lattice.cells, run.common, "average")
    doe = saturated_basis(Y, TOGGLE_IDS).doe
    theta_hat, draws = assemble_theta(res, coords, doe)
    return coords, theta_hat, draws


def test_posterior_has_all_coordinates_and_probabilities():
    coords, theta_hat, draws = _theta_and_draws("meas_err", seed=1)
    result = run_bayes(theta_hat, draws, coords, prior_scale=0.1, vartheta=0.005, epsilon=1e-8)
    assert set(result.posteriors) == set(coords)
    for T, post in result.posteriors.items():
        assert 0.0 <= post.p_negative <= 1.0
        assert abs(post.p_negative + post.p_positive - 1.0) < 1e-9
        assert 0.0 <= post.p_material <= 1.0
        assert post.ci_low <= post.mean <= post.ci_high


def test_injected_effect_has_high_directional_probability():
    coords, theta_hat, draws = _theta_and_draws("meas_err", seed=2)
    result = run_bayes(theta_hat, draws, coords, prior_scale=0.1, vartheta=0.002, epsilon=1e-8)
    post = result.posteriors[frozenset({"meas_err"})]
    # a clearly nonzero effect => posterior mass concentrates on one side
    assert max(post.p_negative, post.p_positive) > 0.95
    assert post.p_material > 0.5


def test_tighter_prior_shrinks_posterior_mean_toward_zero():
    coords, theta_hat, draws = _theta_and_draws("meas_err", seed=3)
    loose = run_bayes(theta_hat, draws, coords, prior_scale=1.0, vartheta=0.005, epsilon=1e-8)
    tight = run_bayes(theta_hat, draws, coords, prior_scale=1e-3, vartheta=0.005, epsilon=1e-8)
    T = frozenset({"meas_err"})
    assert abs(tight.posteriors[T].mean) < abs(loose.posteriors[T].mean)


def test_regularisation_and_sensitivity_recorded():
    coords, theta_hat, draws = _theta_and_draws("stale_price", seed=4)
    result = run_bayes(theta_hat, draws, coords, prior_scale=0.1, vartheta=0.005, epsilon=1e-8)
    d = result.to_dict()
    assert "regularisation" in d
    assert d["diagonal_sensitivity_max_mean_shift"] >= 0.0
    assert set(d["posteriors"])  # non-empty labelled posteriors


def test_reduced_lattice_theta_has_variable_dimension():
    coords = confirmatory_coordinates(("meas_err", "stale_price", "lib_gap", "lab_trim"))
    # survivorship dropped => 4 main + 2 pairs (meas×stale, lib×trim) = 6 coords
    assert len(coords) == 6


def test_run_bayes_refuses_empty_coordinates():
    with pytest.raises(ValueError, match="no confirmatory coordinates"):
        run_bayes(np.array([]), np.empty((10, 0)), [], prior_scale=0.1,
                  vartheta=0.005, epsilon=1e-8)


# --------------------------------------------------------------------------
# S3 — rank-deficient covariance must not crash; loader guards ε>0, prior_scale>0
# --------------------------------------------------------------------------

def test_run_bayes_survives_rank_deficient_draws():
    # Two coordinates whose bootstrap draws are perfectly collinear => a singular
    # covariance. With ε > 0 the eigen floor makes it PD and run_bayes must not raise.
    coords = [frozenset({"a"}), frozenset({"b"})]
    base = np.random.default_rng(0).normal(0.0, 0.01, size=400)
    draws = np.column_stack([base, base])       # identical columns => rank 1
    theta_hat = np.array([float(base.mean()), float(base.mean())])
    result = run_bayes(theta_hat, draws, coords, prior_scale=0.1,
                       vartheta=0.005, epsilon=1e-8)
    assert set(result.posteriors) == set(coords)
    assert result.regularisation.n_eigenvalues_clipped >= 1
    for post in result.posteriors.values():
        assert np.isfinite(post.mean) and np.isfinite(post.sd)


def test_load_bayes_params_rejects_nonpositive_epsilon(tmp_path):
    import textwrap
    from agents.auditor.thresholds import AuditorThresholdError, load_bayes_params
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent("""
        auditor:
          bayes:
            prior_scale: 0.1
            epsilon: 0.0
    """))
    with pytest.raises(AuditorThresholdError, match="strictly positive"):
        load_bayes_params(p)


def test_load_bayes_params_rejects_nonpositive_prior_scale(tmp_path):
    import textwrap
    from agents.auditor.thresholds import AuditorThresholdError, load_bayes_params
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent("""
        auditor:
          bayes:
            prior_scale: -1.0
            epsilon: 1.0e-8
    """))
    with pytest.raises(AuditorThresholdError, match="strictly positive"):
        load_bayes_params(p)
