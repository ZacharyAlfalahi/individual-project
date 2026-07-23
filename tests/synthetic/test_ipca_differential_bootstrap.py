"""
The §5.3 conditional bootstrap wired into the 2x2: with sufficient common support every effect
carries a computed interval bracketing its point estimate; when the support is too small for the
block length the effects carry an honest 'refused' status (never a fabricated CI); with no bootstrap
requested they stay 'deferred'.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.differential import differential_from_feeds
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import (
    load_ipca_bootstrap_config,
    load_ipca_lambda,
    load_ipca_projection_gate,
)

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()
BCFG = dataclasses.replace(load_ipca_bootstrap_config(), n_replicates=200)


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


def test_computed_intervals_bracket_point_estimates():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(72)                    # 72/6 = 12 >= 10 blocks
        res = differential_from_feeds(
            "meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, bootstrap=BCFG, bootstrap_seed=1
        )
    d = res.to_dict()
    for key in ("data_margin_theta_n_corr", "data_margin_theta_b_corr", "interaction_bracket_raw",
                "doe_interaction_effect", "total_corr"):
        eff = d[key]
        assert eff["interval_status"] == "computed"
        lo, hi = eff["interval"]
        assert lo <= eff["value"] <= hi
        assert eff["conditioning"].strip()


def test_small_support_refuses_interval_not_fakes_it():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(36)                    # 36/6 = 6 < 10 blocks → refuse
        res = differential_from_feeds(
            "meas_err", "str", feed_n, feed_b, anchor, LAM, GATE, bootstrap=BCFG, bootstrap_seed=1
        )
    d = res.to_dict()
    eff = d["interaction_bracket_raw"]
    assert eff["interval_status"] == "refused"
    assert eff["interval"] is None
    assert eff["conditioning"].strip()                          # caveat still present


def test_no_bootstrap_leaves_deferred():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, feed_b, anchor = _setup(72)
        res = differential_from_feeds("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE)
    assert res.to_dict()["interaction_bracket_raw"]["interval_status"] == "deferred_inc3"
