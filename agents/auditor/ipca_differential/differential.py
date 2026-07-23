"""
differential.py — the 2x2 factorial per (bias, anchor) (spec §2, §3.1).

Two ALS fits per bias (Θ̂_N reused across biases); everything else is linear projection and
regression. For a fixed anchor built once from P_N:

    Θ̂_N = production_fit(feed_N)          Θ̂_b = production_fit(feed_b)
    Y_{P,Θ} = alpha( anchor_fixed , recover_factor_series(feed_P, Θ) )   for P∈{N,b}, Θ∈{N,b}

Common support is formed **after every cell's projection-validity filter** (§6.4/§7): the
intersection of all four cells' valid return months with the anchor's months. Contrasts are in
correction orientation; the bracket I is the difference-in-differences. The algebraic identity is
re-asserted when the result is constructed (``schemas.IPCADifferentialResult``).

``differential_from_feeds`` is the core (fed synthetic inputs by the build gates).
``run_differential`` is the real entry point (panels via ``panel_states``, anchors via
``run_bbw_factor``); its real-data validation lives in the experimentalist build.
"""

from __future__ import annotations

import pandas as pd

from agents.auditor.thresholds import (
    IPCALambda,
    IPCAProjectionGate,
    load_ipca_lambda,
    load_ipca_projection_gate,
    load_ipca_reporting,
)
from agents.quant.library.bbw_factors import run_bbw_factor
from agents.quant.library.ipca_feed import IPCAFeed

from .evaluate import alpha_on, recover_factor_series
from .frozen_state import FrozenIPCAState
from .panels import build_cell_feed, panel_states
from .production_fit import production_fit
from .schemas import (
    IPCADifferentialCell,
    IPCADifferentialResult,
    deferred_effect,
)


def _cell(label: str, panel_state: str, fitted_state: str, value: float,
          state: FrozenIPCAState, rec) -> IPCADifferentialCell:
    return IPCADifferentialCell(
        panel_state=panel_state,
        fitted_state=fitted_state,
        label=label,
        value=value,
        frozen_state_hash=state.content_hash,
        projection_diagnostics=rec.gated.diagnostics.to_dict(),
    )


def differential_from_feeds(
    bias: str,
    anchor_name: str,
    feed_n: IPCAFeed,
    feed_b: IPCAFeed,
    anchor: pd.Series,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    *,
    is_focal: bool = False,
    anchor_b: pd.Series | None = None,
) -> IPCADifferentialResult:
    """Compute the 2x2 for one (bias, anchor) from the two panel-state feeds and a fixed anchor
    (a pd.Series indexed by pd.Period over return months). ``anchor_b`` (arm-matched) enables the
    secondary diagonal end-to-end line; when None it is omitted."""
    theta_n = production_fit(feed_n, lam)                       # §5.2 — the only fit call site
    theta_b = production_fit(feed_b, lam)

    rec_nn = recover_factor_series(feed_n, theta_n, gate)       # P_N under Θ_N
    rec_nb = recover_factor_series(feed_n, theta_b, gate)       # P_N under Θ_b
    rec_bn = recover_factor_series(feed_b, theta_n, gate)       # P_b under Θ_N
    rec_bb = recover_factor_series(feed_b, theta_b, gate)       # P_b under Θ_b

    # Common support AFTER projection validity: intersect all four valid sets with the anchor.
    common = (
        set(rec_nn.valid_periods)
        & set(rec_nb.valid_periods)
        & set(rec_bn.valid_periods)
        & set(rec_bb.valid_periods)
        & set(anchor.index)
    )
    periods = sorted(common)

    y_nn = alpha_on(anchor, rec_nn.factors_by_period, periods)
    y_nb = alpha_on(anchor, rec_nb.factors_by_period, periods)
    y_bn = alpha_on(anchor, rec_bn.factors_by_period, periods)
    y_bb = alpha_on(anchor, rec_bb.factors_by_period, periods)

    # Contrasts — correction orientation (§2.2).
    d_data_tn = y_nn - y_bn                 # Δ_data|Θ_N
    d_data_tb = y_nb - y_bb                 # Δ_data|Θ_b
    d_est_pn = y_nn - y_nb                  # Δ_est|P_N
    d_est_pb = y_bn - y_bb                  # Δ_est|P_b
    d_total = y_nn - y_bb                   # Δ_total
    bracket = y_nn - y_bn - y_nb + y_bb     # I

    cells = (
        _cell("Y_NN", "P_N", "Theta_N", y_nn, theta_n, rec_nn),
        _cell("Y_Nb", "P_N", "Theta_b", y_nb, theta_b, rec_nb),
        _cell("Y_bN", "P_b", "Theta_N", y_bn, theta_n, rec_bn),
        _cell("Y_bb", "P_b", "Theta_b", y_bb, theta_b, rec_bb),
    )

    # Secondary arm-matched diagonal end-to-end (§3.1) — one descriptive line, own-arm support.
    secondary = None
    if anchor_b is not None:
        n_periods = sorted(set(rec_nn.valid_periods) & set(anchor.index))
        b_periods = sorted(set(rec_bb.valid_periods) & set(anchor_b.index))
        secondary = {
            "N": alpha_on(anchor, rec_nn.factors_by_period, n_periods),
            "b": alpha_on(anchor_b, rec_bb.factors_by_period, b_periods),
        }

    return IPCADifferentialResult(
        bias=bias,
        anchor=anchor_name,
        is_focal=is_focal,
        cells=cells,
        data_margin_theta_n=deferred_effect("data_margin_theta_n_corr", d_data_tn),
        data_margin_theta_b=deferred_effect("data_margin_theta_b_corr", d_data_tb),
        interaction_bracket_raw=deferred_effect("interaction_bracket_raw_corr", bracket),
        doe_interaction_effect=deferred_effect("doe_interaction_effect_corr", bracket / 2.0),
        est_margin_p_n=deferred_effect("est_margin_p_n_corr", d_est_pn),
        est_margin_p_b=deferred_effect("est_margin_p_b_corr", d_est_pb),
        total=deferred_effect("total_corr", d_total),
        secondary_endtoend=secondary,
        common_support_n_months=len(periods),
    )


def _anchor_series(view_panel: pd.DataFrame, anchor_name: str) -> pd.Series:
    """Build a fixed anchor long-short return series (indexed by return-month pd.Period) from a
    view() panel via the audited characteristic-sort engine."""
    result = run_bbw_factor(view_panel, anchor_name)
    mr = result["monthly_returns"]
    periods = pd.PeriodIndex(pd.to_datetime(mr["date"]), freq="M")
    return pd.Series(mr["strategy_ret"].to_numpy(), index=periods)


def run_differential(
    bias: str,
    anchor_name: str,
    maximal: pd.DataFrame,
    signals: pd.DataFrame,
    reg: dict,
    *,
    lam: IPCALambda | None = None,
    gate: IPCAProjectionGate | None = None,
    thresholds_path=None,
) -> IPCADifferentialResult:
    """Real entry point: build (P_N, P_b) via ``panel_states``, their feeds and the fixed anchor
    from P_N, then the 2x2. Construction biases raise ``ConstructionToggleDeferred`` (see panels.py).
    Real-data validation lives in the experimentalist build."""
    lam = lam or load_ipca_lambda(thresholds_path)
    gate = gate or load_ipca_projection_gate(thresholds_path)
    reporting = load_ipca_reporting(thresholds_path)
    is_focal = reporting.focal_pairs.get(bias) == anchor_name

    p_n, p_b = panel_states(bias, maximal, signals)
    family_n = "corr"
    family_b = "raw" if bias == "meas_err" else "corr"
    feed_n = build_cell_feed(p_n, reg, family_n)
    feed_b = build_cell_feed(p_b, reg, family_b)
    anchor = _anchor_series(p_n, anchor_name)                  # fixed anchor from P_N
    return differential_from_feeds(
        bias, anchor_name, feed_n, feed_b, anchor, lam, gate, is_focal=is_focal,
    )
