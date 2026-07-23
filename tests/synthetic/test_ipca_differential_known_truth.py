"""
Gate §6.5 — known-truth. Under noiseless data, full column rank of B_t every month, the production
normalisation and no omitted intercept, the frozen projection recovers the true f_t EXACTLY. This
proves the evaluation stage computes what it claims (and cannot fail for an algebraically legitimate
reason). Tested both on the bare projection primitive and through the full evaluation path.
"""

from __future__ import annotations

import warnings

import numpy as np

from agents.auditor.ipca_differential.evaluate import recover_factor_series
from agents.auditor.ipca_differential.frozen_state import FrozenIPCAState
from agents.auditor.ipca_differential.synthetic import fit_from_truth, make_synthetic_feed
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate
from agents.quant.library.ipca import _oos_factor_realization, build_sufficient_stats

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()


def test_projection_primitive_recovers_truth_exactly():
    feed, truth = make_synthetic_feed(
        K=LAM.factor_count, L=LAM.instrument_count, T=30, n=40, seed=3, noise_sd=0.0
    )
    gamma, factors = truth["gamma"], truth["factors"]
    stats = build_sufficient_stats(feed.Z, feed.R, feed.months)
    for t in range(len(feed.months)):
        f_hat = _oos_factor_realization(stats.W[t], stats.x[t], gamma, None)
        np.testing.assert_allclose(f_hat, factors[:, t], atol=1e-9, rtol=0)


def test_evaluation_path_recovers_truth_exactly():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, truth = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=30, n=40, seed=4, noise_sd=0.0
        )
        gamma, factors = truth["gamma"], truth["factors"]
        # Freeze a state whose loadings ARE the DGP truth (restricted α=0).
        state = FrozenIPCAState.freeze(fit_from_truth(gamma, factors, feed.months), LAM)
        rec = recover_factor_series(feed, state, GATE)
        assert rec.gated.diagnostics.n_valid_months == len(feed.months)   # all months identified
        for t, period in enumerate(rec.valid_periods):
            np.testing.assert_allclose(rec.factors_by_period[period], factors[:, t], atol=1e-9, rtol=0)
