"""
The projection identifiability gate (spec §6.4, D-A53): gated literally on B_t = Z_t Γ.

Full-rank months pass; a below-minimum cross-section is excluded; when the failed fraction exceeds
the registered maximum the whole cell is refused (its valid support is empty). Common support is
formed only from the surviving ``valid_periods``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from agents.auditor.ipca_differential.projection_gate import gate_cell
from agents.auditor.thresholds import load_ipca_projection_gate
from agents.quant.library.ipca_feed import IPCAFeed

GATE = load_ipca_projection_gate()   # min_cross_section_n=8, require_rank=5, max_failed=0.10


def _gamma(L: int = 8, K: int = 5, seed: int = 1) -> np.ndarray:
    g, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((L, K)))
    return g


def _feed_with_ns(ns, L: int = 8, seed: int = 0) -> IPCAFeed:
    rng = np.random.default_rng(seed)
    Z, R, months, asof, vol, cus = [], [], [], [], [], []
    for i, n in enumerate(ns):
        chars = rng.uniform(-0.5, 0.5, size=(n, L - 1))
        Z.append(np.column_stack([chars, np.ones(n)]))
        R.append(rng.normal(size=n))
        months.append(i + 1)
        asof.append(i)
        vol.append(np.ones(n))
        cus.append(np.arange(n))
    return IPCAFeed(Z=Z, R=R, months=np.asarray(months), asof=np.asarray(asof),
                    vol_scaler=vol, cusips=cus)


def test_full_rank_months_all_pass():
    feed = _feed_with_ns([30] * 10)
    g = gate_cell(feed, _gamma(), GATE)
    assert g.diagnostics.n_valid_months == 10
    assert not g.diagnostics.cell_refused
    assert g.diagnostics.months_removed == ()


def test_below_min_cross_section_month_excluded():
    feed = _feed_with_ns([30, 30, 5, 30, 30, 30, 30, 30, 30, 30])   # month 3 has n=5 < 8
    g = gate_cell(feed, _gamma(), GATE)
    assert 3 in g.diagnostics.months_removed
    assert g.diagnostics.n_valid_months == 9
    assert not g.diagnostics.cell_refused


def test_cell_refused_when_failed_fraction_exceeds_max():
    feed = _feed_with_ns([5, 5, 5, 30, 30, 30, 30, 30, 30, 30])     # 30% below min > 10% cap
    g = gate_cell(feed, _gamma(), GATE)
    assert g.diagnostics.cell_refused
    assert g.valid_periods == ()
    assert g.diagnostics.n_valid_months == 0


def test_cell_refusal_mode_refuses_on_any_failure():
    gate = dataclasses.replace(GATE, failed_month_handling="cell_refusal")
    feed = _feed_with_ns([30, 30, 5, 30])                            # a single failure
    g = gate_cell(feed, _gamma(), gate)
    assert g.diagnostics.cell_refused
    assert g.valid_periods == ()
