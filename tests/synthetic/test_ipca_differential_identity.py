"""
Gate §6.1 — engineering identity. With identical panels (P_0 = P_1) the production rule yields
Θ̂_0 ≡ Θ̂_1 (identical content hash), all four cells are bit-equal to numerical tolerance, and the
interaction bracket I ≡ 0. A BUILD gate (correct wiring, deterministic ALS, no provenance leakage)
— never statistical evidence.
"""

from __future__ import annotations

import warnings

from agents.auditor.ipca_differential.differential import differential_from_feeds
from agents.auditor.ipca_differential.production_fit import production_fit
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed, synthetic_anchor
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()


def test_identical_panels_give_identical_fit_hash():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=40, n=30, seed=5, noise_sd=0.01
        )
        a = production_fit(feed, LAM)
        b = production_fit(feed, LAM)         # same feed → deterministic ALS → same artefact
        assert a.content_hash == b.content_hash


def test_engineering_identity_zero_bracket():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=40, n=30, seed=6, noise_sd=0.01
        )
        anchor = synthetic_anchor(feed)
        # P_0 = P_1 = feed (identical panels): the same object drives both arms.
        res = differential_from_feeds("meas_err", "str", feed, feed, anchor, LAM, GATE)

        # Θ_0 ≡ Θ_1: exactly one unique frozen-state hash across the four cells.
        assert len({c.frozen_state_hash for c in res.cells}) == 1
        # Four cells bit-equal (to registered tolerance).
        vals = [c.value for c in res.cells]
        assert max(vals) - min(vals) < 1e-9
        # I ≡ 0 (and, hence, both data margins ≈ 0).
        assert abs(res.interaction_bracket_raw.value) < 1e-9
        assert abs(res.data_margin_theta_n.value) < 1e-9
        assert abs(res.data_margin_theta_b.value) < 1e-9
        assert res.common_support_n_months == len(feed.months)
