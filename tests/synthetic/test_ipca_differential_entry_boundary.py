"""
Gate §6.6 — entry-boundary. Each injected bias enters through its REGISTERED PRIMITIVE (returns /
membership / characteristic timing) and produces no direct mutation of unrelated primitives —
validated at the input boundary (the attribution premise, asserted where it is true). Downstream
propagation across derived quantities is expected and unconstrained.

Reuses the audited injectors in ``agents/auditor/data/synthetic_panel.py`` on the maximal panel.
Covers the three panel_view biases handled in this build (the two construction biases are
deferred; see panels.ConstructionToggleDeferred).
"""

from __future__ import annotations

import numpy as np

from agents.auditor.data.synthetic_panel import (
    SyntheticSpec,
    inject_meas_err,
    inject_stale_price,
    inject_survivorship,
    make_clean_maximal_panel,
)

SPEC = SyntheticSpec(n_bonds=30, n_months=12, seed=0)


def test_meas_err_enters_only_through_raw_returns():
    panel, signals = make_clean_maximal_panel(SPEC)
    injected, sig2 = inject_meas_err(panel, signals, mag=0.02)
    # characteristics (signals) untouched — meas_err does not rewrite the ranking signal.
    assert sig2 is signals
    # only the RAW return family moves (the registered primitive); corr family stays clean.
    assert not injected["ret_raw"].equals(panel["ret_raw"])
    assert injected["xret_raw"].ne(panel["xret_raw"]).any()
    assert injected["ret_corr"].equals(panel["ret_corr"])
    assert injected["xret_corr"].equals(panel["xret_corr"])
    # membership + timing untouched.
    assert injected["exit_reason"].equals(panel["exit_reason"])
    assert injected["last_trade_date_raw"].equals(panel["last_trade_date_raw"])


def test_stale_price_enters_through_timing_and_moves_both_families_equally():
    panel, signals = make_clean_maximal_panel(SPEC)
    injected, sig2 = inject_stale_price(panel, signals, mag=0.05)
    assert sig2 is signals
    # staleness primitive: last_trade_date moved back for some bonds.
    assert not injected["last_trade_date_raw"].equals(panel["last_trade_date_raw"])
    # returns move in BOTH families by the SAME amount, so meas_err (a raw-vs-corr gap) stays inert.
    d_raw = (injected["ret_raw"] - panel["ret_raw"]).to_numpy()
    d_corr = (injected["ret_corr"] - panel["ret_corr"]).to_numpy()
    np.testing.assert_allclose(d_raw, d_corr, atol=1e-12)
    # membership untouched.
    assert injected["exit_reason"].equals(panel["exit_reason"])


def test_survivorship_enters_through_membership():
    panel, signals = make_clean_maximal_panel(SPEC)
    injected, sig2 = inject_survivorship(panel, signals, mag=0.30)
    assert sig2 is signals
    # membership primitive: distress exits appear where the clean panel had none.
    assert not panel["exit_reason"].notna().any()
    assert injected["exit_reason"].notna().any()
    assert (injected["exit_reason"].dropna() == "defaulted").all()
    # rows are dropped after default (a membership change), never added.
    assert len(injected) <= len(panel)
