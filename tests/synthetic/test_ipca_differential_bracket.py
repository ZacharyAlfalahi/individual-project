"""
The exact interaction bracket (spec §2.2/§2.3, the Algebraic/Theorem row).

With genuinely distinct cells (a factor-correlated fixed anchor and two different panel states), the
difference-in-differences identities hold exactly:

    I = Y_NN − Y_bN − Y_Nb + Y_bb = Δ_data|Θ_N − Δ_data|Θ_b = Δ_est|P_N − Δ_est|P_b,
    DOE = I/2,   total = Y_NN − Y_bb = Δ_data|Θ_b + Δ_est|P_N.

These are re-asserted at result construction; here we also verify the cells are not degenerate
(a meaningful test, not 0 = 0) and that the raw bracket and the DOE effect are distinct fields.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from agents.auditor.ipca_differential.differential import differential_from_feeds
from agents.auditor.ipca_differential.synthetic import make_synthetic_feed
from agents.auditor.thresholds import load_ipca_lambda, load_ipca_projection_gate

LAM = load_ipca_lambda()
GATE = load_ipca_projection_gate()


def _factor_correlated_anchor(feed, truth, seed: int = 2) -> pd.Series:
    """An anchor that genuinely loads on the factor space, so the cells differ across fitted states."""
    periods = pd.PeriodIndex([pd.Period(ordinal=int(m), freq="M") for m in feed.months])
    w = np.linspace(1.0, -1.0, LAM.factor_count)
    y = truth["factors"].T @ w + np.random.default_rng(seed).normal(0.0, 0.005, size=len(periods))
    return pd.Series(y, index=periods)


def test_bracket_identities_hold_with_distinct_cells():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feed_n, truth = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=48, n=40, seed=11, noise_sd=0.02
        )
        # A genuinely different panel state (different DGP realisation → different fitted span).
        feed_b, _ = make_synthetic_feed(
            K=LAM.factor_count, L=LAM.instrument_count, T=48, n=40, seed=99, noise_sd=0.02
        )
        anchor = _factor_correlated_anchor(feed_n, truth)

        res = differential_from_feeds("meas_err", "str", feed_n, feed_b, anchor, LAM, GATE)
        d = res.to_dict()

        cells = {c["label"]: c["value"] for c in d["cells"]}
        y_nn, y_nb, y_bn, y_bb = cells["Y_NN"], cells["Y_Nb"], cells["Y_bN"], cells["Y_bb"]

        # Non-degenerate: the four cells genuinely differ.
        assert max(cells.values()) - min(cells.values()) > 1e-4

        bracket = y_nn - y_bn - y_nb + y_bb
        reported = d["interaction_bracket_raw"]["value"]
        assert abs(reported - bracket) < 1e-9
        assert abs(reported - (d["data_margin_theta_n_corr"]["value"] - d["data_margin_theta_b_corr"]["value"])) < 1e-9
        assert abs(reported - (d["est_margin_p_n_corr"]["value"] - d["est_margin_p_b_corr"]["value"])) < 1e-9
        assert abs(d["doe_interaction_effect"]["value"] - reported / 2.0) < 1e-9
        assert abs(d["total_corr"]["value"] - (y_nn - y_bb)) < 1e-9

        # Field-name discipline: the raw bracket and the DOE effect never share a name.
        assert d["interaction_bracket_raw"]["name"] != d["doe_interaction_effect"]["name"]
        # Every headlined effect carries a deferred-interval status + conditioning caveat.
        for key in ("data_margin_theta_n_corr", "interaction_bracket_raw"):
            assert d[key]["interval_status"] == "deferred_inc3"
            assert d[key]["conditioning"].strip()
