"""
The coupling stability diagnostic (spec §5.4): R' blocked-resample refits reporting sign/magnitude
survival of the interaction bracket I. It re-fits both arms each refit (unlike the §5.3 bootstrap),
is deterministic under a fixed seed, refuses when the block scheme is incompatible with the support,
and is NEVER exposed as an interval.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from agents.auditor.checks.bootstrap import BootstrapError
from agents.auditor.ipca_differential.stability import stability_diagnostic
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import (
    load_ipca_lambda,
    load_ipca_projection_gate,
    load_ipca_stability_config,
)

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()
SCFG = dataclasses.replace(load_ipca_stability_config(), r_prime=6)   # small R' for test speed


def _setup(T: int, seed: int = 11):
    feed_n, truth = make_synthetic_feed(
        K=LAM.factor_count, L=LAM.instrument_count, T=T, n=40, seed=seed, noise_sd=0.02
    )
    feed_b, _ = make_synthetic_feed(
        K=LAM.factor_count, L=LAM.instrument_count, T=T, n=40, seed=seed + 88, noise_sd=0.02
    )
    per = pd.PeriodIndex([pd.Period(ordinal=int(m), freq="M") for m in feed_n.months])
    w = np.linspace(1.0, -1.0, LAM.factor_count)
    anchor = pd.Series(truth["factors"].T @ w + np.random.default_rng(2).normal(0, 0.005, len(per)), index=per)
    return feed_n, feed_b, anchor


def test_reports_survival_and_is_never_an_interval():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(72)
        sd = stability_diagnostic("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, seed=1)
    d = sd.to_dict()["coupling_stability_diagnostic"]
    assert d["is_interval"] is False
    assert d["block_length"] == 6 and d["effective_blocks"] == 12
    assert 0.0 <= sd.sign_survival <= 1.0
    assert 0.0 <= sd.magnitude_survival <= 1.0
    assert sd.n_usable_refits <= sd.r_prime == 6
    assert len(sd.i_draws) == 6


def test_deterministic_under_fixed_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(72)
        a = stability_diagnostic("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, seed=7)
        b = stability_diagnostic("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, seed=7)
    assert a.i_obs == b.i_obs
    np.testing.assert_array_equal(np.array(a.i_draws), np.array(b.i_draws))


def test_supplied_i_obs_is_used():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(72)
        sd = stability_diagnostic(
            "meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, i_obs=0.05, seed=1
        )
    assert sd.i_obs == 0.05


def test_refuses_small_support():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(36)                 # 36/6 = 6 < 10 blocks
        with pytest.raises(BootstrapError):
            stability_diagnostic("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, seed=1)


@pytest.mark.parametrize("t_n,t_b", [(78, 66), (66, 78)])
def test_handles_mismatched_arm_month_counts(t_n, t_b):
    """Regression (C1): P_N and P_{N\\b} routinely differ in surviving months on real panels. The
    diagnostic must resample the SHARED month support — never crash, never resample one arm over the
    other's index space."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, _ = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=t_n, n=30, seed=1, noise_sd=0.02)
        feed_b, _ = make_synthetic_feed(K=LAM.factor_count, L=LAM.instrument_count, T=t_b, n=30, seed=2, noise_sd=0.02)
        per = pd.PeriodIndex([pd.Period(ordinal=m, freq="M") for m in range(1, max(t_n, t_b) + 1)])
        anchor = pd.Series(np.random.default_rng(3).normal(0, 0.01, len(per)), index=per)
        sd = stability_diagnostic("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, SCFG, seed=1)
    assert sd.t_common == min(t_n, t_b)                     # resampled on the intersection
    assert sd.n_usable_refits >= 1
    assert 0.0 <= sd.sign_survival <= 1.0
