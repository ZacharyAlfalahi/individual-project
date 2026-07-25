"""
D-A55 assert-once: gamma_illiq is INVARIANT by construction under the two panel_view toggles.

Its source data (the daily TRACE panels) lies outside both toggles' registered primitives (monthly
price_eom for stale_price; monthly terminal rows for survivorship), so gamma_illiq's per-state value is
bit-identical across P_N and P_{N∖b} on every common bond-month. This is a reclassification, NOT a
membership-only fallback: nothing is truncated because nothing the toggles reach feeds gamma. The claim
is asserted here, not assumed. Skipped when the dev panel is absent.
"""

from __future__ import annotations

import warnings

import pytest

from agents.auditor.ipca_differential.panels import panel_states
from agents.auditor.ipca_differential.runner import MAXIMAL_PANEL, load_dev_inputs


@pytest.mark.parametrize("bias", ["stale_price", "survivorship"])
def test_gamma_illiq_invariant_under_panel_view_toggles(bias):
    if not MAXIMAL_PANEL.exists():
        pytest.skip("dev maximal panel not present")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maximal, signals, _ = load_dev_inputs()
        p_n, p_b = panel_states(bias, maximal, signals)
    merged = p_n[["cusip", "date", "gamma_illiq"]].merge(
        p_b[["cusip", "date", "gamma_illiq"]], on=["cusip", "date"], suffixes=("_N", "_b")
    )
    assert len(merged) > 0
    delta = (merged["gamma_illiq_N"].fillna(-9e9) - merged["gamma_illiq_b"].fillna(-9e9)).abs()
    n_diff = int((delta > 0).sum())
    assert n_diff == 0, f"gamma_illiq differs under {bias} on {n_diff} common bond-months (expected 0)"
