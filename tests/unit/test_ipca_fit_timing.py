"""
The §12.5 fit-timing measurement + the §5.3 multi-start range diagnostic.

Pins: timing is positive; the escape-hatch decision (best-of-M affordable vs collapse to single-init)
follows the budget, not preference; base_m=1 is always affordable; the multi-start range is a
non-negative diagnostic that is NEVER exposed as an interval.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.fit_timing import (
    multistart_range,
    recommend_initialisations,
    time_one_fit,
)
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()


def _feeds_and_anchor(T: int = 48):
    fn, truth = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=T, n=30, seed=11, noise_sd=0.02)
    fb, _ = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=T, n=30, seed=99, noise_sd=0.02)
    per = pd.PeriodIndex([pd.Period(ordinal=int(m), freq="M") for m in fn.months])
    w = np.linspace(1.0, -1.0, LAM.factor_count)
    anchor = pd.Series(truth["factors"].T @ w + np.random.default_rng(2).normal(0, 0.005, len(per)), index=per)
    return fn, fb, anchor


def test_time_one_fit_is_positive():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn, _, _ = _feeds_and_anchor()
        assert time_one_fit(fn, LAM, repeats=2) > 0.0


def test_escape_hatch_follows_budget_not_preference():
    # Affordable: huge budget -> keep best-of-M.
    keep = recommend_initialisations(0.01, n_replicate_fits=10, budget_seconds=1e6, base_m=5)
    assert keep.affordable and keep.recommended_n_initialisations == 5
    # Infeasible: tiny budget -> collapse to a single fixed initialisation.
    collapse = recommend_initialisations(1.0, n_replicate_fits=1000, budget_seconds=1.0, base_m=5)
    assert not collapse.affordable and collapse.recommended_n_initialisations == 1


def test_base_m_one_is_always_affordable():
    dec = recommend_initialisations(1e9, n_replicate_fits=1_000_000, budget_seconds=0.0, base_m=1)
    assert dec.affordable and dec.recommended_n_initialisations == 1


def test_multistart_range_is_a_nonneg_noninterval_diagnostic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn, fb, anchor = _feeds_and_anchor()
        ms = multistart_range("meas_err", "str", fn, fb, anchor, LAM, GATE, seeds=[1, 2, 3])
    assert ms.n_usable == 3
    assert ms.i_range >= 0.0
    assert ms.to_dict()["multistart_range"]["is_interval"] is False
