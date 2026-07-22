"""
The behavioural invariant / sentinel (spec §4.2, D-A49).

The evaluation entry point accepts only (feed, FrozenIPCAState, λ) and has NO callable route that
fits/optimises. Proven by monkeypatching the ALS internals to raise: a full cell evaluation must
complete (it calls only ``_oos_factor_realization`` + linear algebra), while ``production_fit`` —
which DOES fit — raises, confirming the patch is effective (the trap would close on any fitting).
"""

from __future__ import annotations

import warnings

import pytest

import agents.quant.library.ipca as ipca_mod
from agents.auditor.ipca_differential.evaluate import alpha_on, recover_factor_series
from agents.auditor.ipca_differential.production_fit import production_fit
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed, synthetic_anchor
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()


def _boom(*args, **kwargs):
    raise AssertionError("a fitting/optimising routine was invoked during cell evaluation")


def test_no_fit_during_cell_evaluation(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=40, n=30, seed=0, noise_sd=0.01
        )
        state = production_fit(feed, LAM)          # fit BEFORE the sentinel is armed

        # Arm the sentinel: any ALS step now raises. fit_ipca calls these as module globals.
        monkeypatch.setattr(ipca_mod, "_als_loop", _boom)
        monkeypatch.setattr(ipca_mod, "_gamma_step", _boom)
        monkeypatch.setattr(ipca_mod, "_factor_step", _boom)

        # Evaluation must complete — it never fits.
        rec = recover_factor_series(feed, state, GATE)
        value = alpha_on(synthetic_anchor(feed), rec.factors_by_period, list(rec.valid_periods))
        assert rec.gated.diagnostics.n_valid_months > 0
        assert value == value                      # not NaN

        # Sanity: the patch is effective — a fit now raises through the ALS internals.
        with pytest.raises(AssertionError):
            production_fit(feed, LAM)
