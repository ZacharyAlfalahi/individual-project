"""
Stochastic perturbation robustness (spec §6.3): the mean-zero-noise run under its honest name.

Pins: it reports the distribution of I under perturbation (a non-null centre is allowed and
interpreted, never flagged), is explicitly NOT an FPR (is_fpr=False), is deterministic, and the
secondary bootstrap zero-coverage under the matched-twin null is high (a well-calibrated interval).
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.perturbation import (
    bootstrap_zero_coverage,
    perturbation_robustness,
)
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import (
    load_ipca_bootstrap_config,
    load_ipca_fpr_config,
    load_ipca_lambda,
    load_ipca_perturbation_config,
    load_ipca_projection_gate,
)

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()
PCFG = dataclasses.replace(load_ipca_perturbation_config(), n_draws=25)


def _feeds_and_anchor(T: int = 72):
    fn, truth = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=T, n=40, seed=11, noise_sd=0.02)
    fb, _ = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=T, n=40, seed=99, noise_sd=0.02)
    per = pd.PeriodIndex([pd.Period(ordinal=int(m), freq="M") for m in fn.months])
    w = np.linspace(1.0, -1.0, LAM.factor_count)
    anchor = pd.Series(truth["factors"].T @ w + np.random.default_rng(2).normal(0, 0.005, len(per)), index=per)
    return fn, fb, anchor


def test_perturbation_reports_distribution_and_is_not_fpr():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn, fb, anchor = _feeds_and_anchor()
        pr = perturbation_robustness(fn, fb, anchor, LAM, GATE, PCFG, seed=1)
    d = pr.to_dict()["perturbation_robustness"]
    assert d["is_fpr"] is False                          # a robustness descriptive, never an FPR
    assert "expected" in d["interpretation"].lower()      # a non-null centre is expected
    assert pr.n_usable == 25
    assert pr.i_std > 0.0                                 # perturbation genuinely moves I
    assert pr.i_min <= pr.i_mean <= pr.i_max


def test_perturbation_is_deterministic():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fn, fb, anchor = _feeds_and_anchor()
        a = perturbation_robustness(fn, fb, anchor, LAM, GATE, PCFG, seed=4)
        b = perturbation_robustness(fn, fb, anchor, LAM, GATE, PCFG, seed=4)
    assert (a.i_mean, a.i_std, a.i_min, a.i_max) == (b.i_mean, b.i_std, b.i_min, b.i_max)


def test_bootstrap_zero_coverage_high_under_null():
    """Secondary §6.3 check: under the exchangeable null the §5.3 interval should cover zero most
    of the time (a well-calibrated interval)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fcfg = load_ipca_fpr_config()
        bcfg = dataclasses.replace(load_ipca_bootstrap_config(), n_replicates=150)
        zc = bootstrap_zero_coverage(fcfg, bcfg, LAM, GATE, seed=3, r=10)
    assert zc.n_usable == 10
    assert zc.coverage_fraction >= 0.6                   # near-nominal coverage under a true null
