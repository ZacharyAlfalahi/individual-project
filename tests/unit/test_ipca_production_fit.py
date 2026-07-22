"""
The single production estimator rule (spec §5.2, D-A52). Single-init is deterministic; the
best-of-M branch (activated only when the §12.5 timing measurement raises n_initialisations) runs
and returns a contract-valid frozen state. One call site produces every Θ̂.
"""

from __future__ import annotations

import dataclasses
import warnings

from agents.auditor.ipca_differential.production_fit import production_fit
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import load_ipca_lambda

LAM = load_ipca_lambda()


def test_single_init_is_deterministic_and_valid():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=36, n=30, seed=0, noise_sd=0.02
        )
        a = production_fit(feed, LAM)
        b = production_fit(feed, LAM)
    assert a.content_hash == b.content_hash
    a.validate_manifest(LAM)


def test_best_of_m_runs_and_returns_valid_state():
    lam_m = dataclasses.replace(LAM, n_initialisations=3, initialisation_seeds=(1, 2, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=36, n=30, seed=1, noise_sd=0.02
        )
        state = production_fit(feed, lam_m)
    state.validate_manifest(lam_m)
    assert state.factor_count == LAM.factor_count
    # best-of-M is deterministic too (fixed seeds, deterministic tie-break).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed2, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=36, n=30, seed=1, noise_sd=0.02
        )
        again = production_fit(feed2, lam_m)
    assert again.content_hash == state.content_hash
