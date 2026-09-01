"""Stage 8 — the synchronised fixed-block vector bootstrap (§6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import agents.auditor.checks.bootstrap as boot_mod
from agents.auditor.checks.bootstrap import (
    BootstrapError,
    block_length,
    circular_block_indices,
    effective_blocks,
    first_order_covariance,
    run_bootstrap,
    stationary_block_indices,
)
from agents.auditor.data.synthetic_panel import build_scenario
from agents.auditor.schemas.lattice_types import CellReturns, MetricSet
from agents.auditor.schemas.toggle import TOGGLE_IDS
from agents.auditor.validation.layer_b_fixtures import run_scenario


# --------------------------------------------------------------------------
# Block-length rule + refusal conditions (§6.2)
# --------------------------------------------------------------------------

def test_block_length_is_the_literal_floor():
    assert block_length(holding_period=6, data_driven_months=3) == 6   # H_max binds
    assert block_length(holding_period=1, data_driven_months=6) == 6   # data-driven binds


def test_effective_blocks_floor_division():
    assert effective_blocks(47, 6) == 7


def test_circular_block_indices_length_and_wrap():
    rng = np.random.default_rng(0)
    idx = circular_block_indices(20, 6, rng)
    assert len(idx) == 20
    assert idx.min() >= 0 and idx.max() < 20  # wrapped in range


# --------------------------------------------------------------------------
# Bootstrap on a real lattice
# --------------------------------------------------------------------------

def _clean_lattice_cells():
    scenario = build_scenario(None, seed=1)
    run = run_scenario(scenario)
    return run.lattice.cells, run.common


def test_block_length_exceeding_support_refuses():
    cells, common = _clean_lattice_cells()
    with pytest.raises(BootstrapError, match="refusal condition"):
        run_bootstrap(
            cells, common, TOGGLE_IDS,
            n_replicates=50, data_driven_block_months=len(common) + 5,
            min_effective_blocks=2, holding_period=1,
        )


def test_too_few_effective_blocks_refuses():
    cells, common = _clean_lattice_cells()
    with pytest.raises(BootstrapError, match="effective blocks"):
        run_bootstrap(
            cells, common, TOGGLE_IDS,
            n_replicates=50, data_driven_block_months=max(2, len(common) // 2),
            min_effective_blocks=100, holding_period=1,
        )


def test_bootstrap_produces_draws_for_every_quantity():
    cells, common = _clean_lattice_cells()
    res = run_bootstrap(
        cells, common, TOGGLE_IDS,
        n_replicates=100, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1, seed=7,
    )
    assert res.block_length == 6
    assert len(res.gap_draws) == 100
    assert set(res.shapley_draws) == set(TOGGLE_IDS)
    # every subset has a DOE draw vector
    assert len(res.doe_draws) == 2 ** len(TOGGLE_IDS)
    lo, hi = res.gap_ci()
    assert lo <= hi


# --------------------------------------------------------------------------
# The common sequence preserves comonotonicity
# --------------------------------------------------------------------------

def _cell(on_set, index, values):
    idx = pd.DatetimeIndex(index)
    returns = pd.Series(values, index=idx, dtype=float)
    n_bonds = pd.Series([10.0] * len(idx), index=idx)
    return CellReturns(
        on_set=frozenset(on_set), run_config=None, returns=returns,
        n_bonds=n_bonds, metrics_native=MetricSet.from_summary(
            {"n_months": len(idx), "months_per_year": 12, "nw_lags_used": 0,
             "average": 0.0, "annualised_average": 0.0, "bumpiness": 0.0,
             "sharpe": 0.0, "t_stat": 0.0, "first_date": None, "last_date": None}
        ),
        run_config_hash="rc", panel_view_hash="pv", return_hash="r",
        n_bonds_hash="n", metric_hash="m",
    )


def test_common_sequence_makes_identical_cells_move_together():
    # Two cells with IDENTICAL returns => a single toggle whose effect is exactly
    # zero. The common block sequence must keep them identical in every replicate,
    # so the endpoint gap draw is exactly zero every time (no spurious interval).
    m = pd.date_range("2005-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(3)
    series = rng.normal(0.01, 0.02, size=60)
    a = _cell(set(), m, series)
    b = _cell({"meas_err"}, m, series)  # identical returns
    res = run_bootstrap(
        [a, b], m, ("meas_err",),
        n_replicates=50, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1, seed=1,
    )
    assert np.allclose(res.gap_draws, 0.0)


# --------------------------------------------------------------------------
# CHEAP PATH — the bootstrap never re-runs the engine (§14)
# --------------------------------------------------------------------------

def test_bootstrap_never_calls_the_engine(monkeypatch):
    cells, common = _clean_lattice_cells()

    def _forbidden(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("run_characteristic_sort was called during bootstrap")

    # Guard the engine symbol wherever the bootstrap could reach it.
    monkeypatch.setattr(boot_mod, "summarize_returns", boot_mod.summarize_returns)
    import agents.quant.library.characteristic_sort as cs
    monkeypatch.setattr(cs, "run_characteristic_sort", _forbidden)

    res = run_bootstrap(
        cells, common, TOGGLE_IDS,
        n_replicates=30, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1,
    )
    assert res.n_replicates == 30


# --------------------------------------------------------------------------
# A real injected effect: its bootstrap interval separates from zero
# --------------------------------------------------------------------------

def test_injected_meas_err_effect_interval_excludes_zero():
    scenario = build_scenario("meas_err", seed=2)
    run = run_scenario(scenario)
    res = run_bootstrap(
        run.lattice.cells, run.common, TOGGLE_IDS,
        n_replicates=300, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1, seed=5,
    )
    lo, hi = res.doe_ci()[frozenset({"meas_err"})]
    assert not (lo <= 0.0 <= hi), f"meas_err CI [{lo}, {hi}] should exclude 0"


# --------------------------------------------------------------------------
# Stationary (Politis–Romano) sensitivity generator (D-A29)
# --------------------------------------------------------------------------

def test_stationary_block_indices_length_and_wrap():
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(50, 6, rng)
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_stationary_block_indices_mean_block_length():
    # Empirical mean block length ≈ mean_ell: count non-contiguous breaks on a long draw.
    rng = np.random.default_rng(1)
    t, mean_ell = 3000, 6
    idx = stationary_block_indices(t, mean_ell, rng)
    breaks = 1 + int(np.sum(idx[1:] != (idx[:-1] + 1)))  # wrap counts as a rare break
    mean_len = t / breaks
    assert 4.0 < mean_len < 9.0, mean_len  # centred on 6, not 1 and not 50


def test_stationary_block_indices_deterministic():
    a = stationary_block_indices(40, 5, np.random.default_rng(7))
    b = stationary_block_indices(40, 5, np.random.default_rng(7))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------
# scheme routing: "fixed" is byte-identical to the prior default; "stationary" runs
# --------------------------------------------------------------------------

def test_scheme_fixed_reproduces_default():
    cells, common = _clean_lattice_cells()
    kw = dict(n_replicates=80, data_driven_block_months=6,
              min_effective_blocks=3, holding_period=1, seed=11)
    default = run_bootstrap(cells, common, TOGGLE_IDS, **kw)
    explicit = run_bootstrap(cells, common, TOGGLE_IDS, scheme="fixed", **kw)
    assert default.scheme == "fixed" and explicit.scheme == "fixed"
    assert np.array_equal(default.gap_draws, explicit.gap_draws)


def test_stationary_scheme_runs_and_is_recorded():
    cells, common = _clean_lattice_cells()
    res = run_bootstrap(
        cells, common, TOGGLE_IDS,
        n_replicates=100, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1, seed=5, scheme="stationary",
    )
    assert res.scheme == "stationary"
    lo, hi = res.gap_ci()
    assert lo <= hi and np.isfinite(lo) and np.isfinite(hi)


def test_unknown_scheme_refuses():
    cells, common = _clean_lattice_cells()
    with pytest.raises(BootstrapError, match="unknown bootstrap scheme"):
        run_bootstrap(
            cells, common, TOGGLE_IDS,
            n_replicates=10, data_driven_block_months=6,
            min_effective_blocks=3, holding_period=1, scheme="wild",
        )


# --------------------------------------------------------------------------
# first_order_covariance: the k×k measurement V̂_s for the hierarchy (D-A32)
# --------------------------------------------------------------------------

def test_first_order_covariance_shape_symmetry_and_diagonal():
    cells, common = _clean_lattice_cells()
    res = run_bootstrap(
        cells, common, TOGGLE_IDS,
        n_replicates=200, data_driven_block_months=6,
        min_effective_blocks=3, holding_period=1, seed=9,
    )
    labels, cov = first_order_covariance(res, TOGGLE_IDS)
    k = len(TOGGLE_IDS)
    assert labels == tuple(TOGGLE_IDS)
    assert cov.shape == (k, k)
    assert np.allclose(cov, cov.T)
    # the diagonal is exactly each first-order effect's own bootstrap variance.
    for i, tg in enumerate(TOGGLE_IDS):
        assert np.isclose(cov[i, i], np.var(res.doe_draws[frozenset({tg})], ddof=1), rtol=1e-9)
