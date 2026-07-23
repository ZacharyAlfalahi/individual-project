"""
Integration: a view()-shaped panel flows through panels.build_cell_feed → production_fit →
recover_factor_series. Proves the real per-cell path (to_merged + build_ipca_feed + feed_matrices +
the estimator) composes on realistic input — without the full maximal/view/anchor machinery, whose
end-to-end validation (run_differential on real dev data) lives in the experimentalist build.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.evaluate import recover_factor_series
from agents.auditor.ipca_differential.panels import build_cell_feed
from agents.auditor.ipca_differential.production_fit import production_fit
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate

REG = {"meta": {"default_family": "corr", "train_end": "2012-12"}, "scaler": {"floor": 0.01}}


def _view_panel(n_bonds: int = 30, n_months: int = 10, seed: int = 0) -> pd.DataFrame:
    """A synthetic view() output carrying xret + the four family-resolved IPCA signals."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rows = []
    for c in range(n_bonds):
        for d in dates:
            rows.append({
                "cusip": f"B{c:03d}", "date": d,
                "xret": float(rng.normal(0, 0.02)),
                "rating": float(rng.integers(1, 22)),
                "time_to_maturity": float(rng.uniform(1, 20)),
                "mom6": float(rng.normal()),
                "var_5pct": float(rng.normal()),
                "gamma_illiq": float(rng.normal()),
                "bond_vol": float(rng.uniform(0.005, 0.05)),
            })
    return pd.DataFrame(rows)


def test_cell_feed_flows_to_production_fit_and_recovery():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed = build_cell_feed(_view_panel(), REG, "corr")
        assert len(feed.Z) >= 1
        # every month carries the constant column last and N_m > L
        for Zt in feed.Z:
            assert Zt.shape[1] == load_ipca_lambda().instrument_count
            np.testing.assert_array_equal(Zt[:, -1], np.ones(Zt.shape[0]))

        lam = load_ipca_lambda()
        state = production_fit(feed, lam)
        state.validate_manifest(lam)                       # the produced state honours the contract

        rec = recover_factor_series(feed, state, load_ipca_projection_gate())
        assert rec.gated.diagnostics.n_valid_months >= 1
