"""WS-A (P3) — closed-form unit tests for the scalar posterior layer, plus the
d=1 consistency pin against the Auditor's pre-registered `_posterior` (the
reuse claim is tested, not asserted)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from shared.stats.posterior import (
    PosteriorSummary,
    alpha_se_from_t,
    posterior_summary,
)

PRIORS = {"wide": 0.02, "moderate": 0.005, "sceptical": 0.0025}


def _row(summary: PosteriorSummary, label: str):
    return next(p for p in summary.posteriors if p.label == label)


# ---------------------------------------------------------------------------
# Closed-form limits.
# ---------------------------------------------------------------------------

def test_diffuse_prior_recovers_the_likelihood():
    s = posterior_summary(0.004, 0.002, {"diffuse": 1e6})
    row = s.posteriors[0]
    assert row.post_mean == pytest.approx(0.004, rel=1e-6)
    assert row.post_sd == pytest.approx(0.002, rel=1e-6)


def test_infinite_noise_recovers_the_prior():
    s = posterior_summary(0.004, 1e6, {"sceptical": 0.0025})
    row = s.posteriors[0]
    assert row.post_mean == pytest.approx(0.0, abs=1e-12)
    assert row.post_sd == pytest.approx(0.0025, rel=1e-6)
    assert row.p_positive == pytest.approx(0.5, abs=1e-9)


def test_zero_point_is_exactly_agnostic():
    s = posterior_summary(0.0, 0.003, PRIORS)
    for row in s.posteriors:
        assert row.p_positive == pytest.approx(0.5, abs=1e-12)
        assert row.post_mean == 0.0


def test_hand_computed_case():
    # point 1%/mo, se 0.5%/mo, prior sigma 1%/mo:
    #   precision = 1/0.005^2 + 1/0.01^2 = 40000 + 10000 = 50000
    #   var = 2e-5, mean = 2e-5 * (0.01 / 2.5e-5) = 0.008, sd = sqrt(2e-5)
    s = posterior_summary(0.01, 0.005, {"m": 0.01})
    row = s.posteriors[0]
    assert row.post_mean == pytest.approx(0.008, rel=1e-9)
    assert row.post_sd == pytest.approx(math.sqrt(2e-5), rel=1e-9)
    assert row.p_positive == pytest.approx(0.96319, abs=5e-4)
    assert row.ci_low == pytest.approx(0.008 - 1.959964 * math.sqrt(2e-5), rel=1e-5)


def test_tighter_priors_shrink_harder():
    s = posterior_summary(0.01, 0.005, PRIORS)
    wide, moderate, sceptical = (_row(s, k).post_mean for k in ("wide", "moderate", "sceptical"))
    assert wide > moderate > sceptical > 0.0
    # And every posterior mean sits between 0 (the prior mean) and the point.
    assert 0.0 < sceptical and wide < 0.01


# ---------------------------------------------------------------------------
# d=1 consistency with the Auditor's pre-registered posterior.
# ---------------------------------------------------------------------------

def test_matches_auditor_posterior_at_d1():
    from agents.auditor.checks.bayes import _posterior

    point, se, sigma = 0.0062, 0.0031, 0.02
    mu, cov = _posterior(np.array([point]), np.array([[1.0 / se**2]]), sigma)
    row = posterior_summary(point, se, {"wide": sigma}).posteriors[0]
    assert row.post_mean == pytest.approx(float(mu[0]), rel=1e-12)
    assert row.post_sd == pytest.approx(math.sqrt(float(cov[0, 0])), rel=1e-12)


# ---------------------------------------------------------------------------
# The SE seam from the G3 output.
# ---------------------------------------------------------------------------

def test_alpha_se_from_t():
    assert alpha_se_from_t(0.01, 2.5) == pytest.approx(0.004)
    assert alpha_se_from_t(-0.01, -2.5) == pytest.approx(0.004)
    with pytest.raises(ValueError):
        alpha_se_from_t(0.01, 0.0)
    with pytest.raises(ValueError):
        alpha_se_from_t(float("nan"), 2.0)


# ---------------------------------------------------------------------------
# Fail-loud inputs.
# ---------------------------------------------------------------------------

def test_fail_loud_inputs():
    with pytest.raises(ValueError):
        posterior_summary(0.01, 0.0, PRIORS)              # se must be > 0
    with pytest.raises(ValueError):
        posterior_summary(0.01, 0.005, {})                # priors non-empty
    with pytest.raises(ValueError):
        posterior_summary(0.01, 0.005, {"bad": 0.0})      # sigma must be > 0
    with pytest.raises(ValueError):
        posterior_summary(float("inf"), 0.005, PRIORS)    # point finite


def test_to_dict_round_trips_numerically():
    s = posterior_summary(0.01, 0.005, PRIORS)
    d = s.to_dict()
    assert d["point"] == 0.01 and len(d["posteriors"]) == 3
    assert d["posteriors"][0]["label"] == "wide"
