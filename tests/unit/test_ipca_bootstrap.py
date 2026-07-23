"""
The conditional moving-block bootstrap (spec §5.3, D-A46).

Pins: percentile intervals bracket the point estimate; ONE common block sequence per replicate makes
the contrasts comonotone (the per-replicate bracket equals the difference-in-differences of the cell
draws exactly); determinism under a fixed seed; refusal (BootstrapError) when the block length is
incompatible with the common support; and the differential wiring (computed vs refused status).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from agents.auditor.checks.bootstrap import BootstrapError
from agents.auditor.ipca_differential.bootstrap import conditional_bootstrap
from agents.auditor.thresholds import load_ipca_bootstrap_config

CFG = dataclasses.replace(load_ipca_bootstrap_config(), n_replicates=300)   # smaller B for speed
K = 5


def _inputs(T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    periods = [pd.Period(ordinal=i, freq="M") for i in range(1, T + 1)]
    labels = ("Y_NN", "Y_Nb", "Y_bN", "Y_bb")
    cell_factors = {lbl: {p: rng.normal(size=K) for p in periods} for lbl in labels}
    anchor = pd.Series(rng.normal(size=T), index=pd.PeriodIndex(periods))
    return cell_factors, anchor, periods


def test_intervals_bracket_point_estimates_and_geometry():
    cell_factors, anchor, periods = _inputs(72, seed=1)
    res = conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=3)
    assert res.block_length == 6 and res.effective_blocks == 12 and res.t_common == 72
    ci = res.intervals()
    for name, draws in res.draws.items():
        lo, hi = ci[name]
        assert lo <= np.median(draws) <= hi                    # median inside its own CI


def test_common_sequence_makes_contrasts_comonotone():
    """The bracket draw equals the difference-in-differences of the margin draws, replicate by
    replicate — only possible because all four cells share one resample sequence."""
    cell_factors, anchor, periods = _inputs(90, seed=2)
    res = conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=5)
    d = res.draws
    np.testing.assert_allclose(
        d["interaction_bracket_raw_corr"],
        d["data_margin_theta_n_corr"] - d["data_margin_theta_b_corr"], atol=1e-12,
    )
    np.testing.assert_allclose(
        d["interaction_bracket_raw_corr"],
        d["est_margin_p_n_corr"] - d["est_margin_p_b_corr"], atol=1e-12,
    )
    np.testing.assert_allclose(
        d["doe_interaction_effect_corr"], d["interaction_bracket_raw_corr"] / 2.0, atol=1e-12,
    )


def test_deterministic_under_fixed_seed():
    cell_factors, anchor, periods = _inputs(72, seed=1)
    a = conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=7)
    b = conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=7)
    for name in a.draws:
        np.testing.assert_array_equal(a.draws[name], b.draws[name])


def test_refuses_when_too_few_effective_blocks():
    cell_factors, anchor, periods = _inputs(30, seed=1)     # 30/6 = 5 < min 10
    with pytest.raises(BootstrapError, match="effective blocks"):
        conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=0)


def test_refuses_when_block_exceeds_support():
    cell_factors, anchor, periods = _inputs(4, seed=1)      # T < ℓ=6
    with pytest.raises(BootstrapError, match="block length"):
        conditional_bootstrap(cell_factors, anchor, periods, CFG, seed=0)
