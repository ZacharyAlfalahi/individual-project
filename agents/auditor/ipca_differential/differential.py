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

from agents.auditor.checks.bootstrap import BootstrapError
from agents.auditor.thresholds import (
    IPCABootstrapConfig,
    IPCALambda,
    IPCAProjectionGate,
    load_ipca_bootstrap_config,
    load_ipca_lambda,
    load_ipca_projection_gate,
    load_ipca_reporting,
)
from agents.quant.library.bbw_factors import run_bbw_factor
from agents.quant.library.characteristic_sort import run_characteristic_sort
from agents.quant.library.ipca_feed import IPCAFeed

from .bootstrap import conditional_bootstrap
from .evaluate import alpha_on, recover_factor_series
from .frozen_state import FrozenIPCAState
from .panels import build_cell_feed, panel_states
from .production_fit import production_fit
from .schemas import (
    IPCADifferentialCell,
    IPCADifferentialResult,
    computed_effect,
    deferred_effect,
    refused_effect,
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
    bootstrap: IPCABootstrapConfig | None = None,
    bootstrap_seed: int = 0,
) -> IPCADifferentialResult:
    """Compute the 2x2 for one (bias, anchor) from the two panel-state feeds and a fixed anchor
    (a pd.Series indexed by pd.Period over return months). ``anchor_b`` (arm-matched) enables the
    secondary diagonal end-to-end line; when None it is omitted. When ``bootstrap`` is supplied,
    each effect carries a conditional moving-block bootstrap interval (§5.3); if the common support
    refuses it, the effects carry an honest 'refused' status instead of a fabricated CI."""
    theta_n = production_fit(feed_n, lam)                       # §5.2 — the only fit call site
    theta_b = production_fit(feed_b, lam)
    return differential_from_states(
        bias, anchor_name, feed_n, feed_b, theta_n, theta_b, anchor, gate,
        is_focal=is_focal, anchor_b=anchor_b, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
    )


def differential_from_states(
    bias: str,
    anchor_name: str,
    feed_n: IPCAFeed,
    feed_b: IPCAFeed,
    theta_n: FrozenIPCAState,
    theta_b: FrozenIPCAState,
    anchor: pd.Series,
    gate: IPCAProjectionGate,
    *,
    is_focal: bool = False,
    anchor_b: pd.Series | None = None,
    bootstrap: IPCABootstrapConfig | None = None,
    bootstrap_seed: int = 0,
) -> IPCADifferentialResult:
    """The 2x2 from two ALREADY-frozen fitted states (no fitting here). Used by
    ``differential_from_feeds`` (after production_fit) and by the multi-start range diagnostic,
    which injects seeded states to measure ALS local-optimum sensitivity (§5.3)."""
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

    values = {
        "data_margin_theta_n_corr": d_data_tn,
        "data_margin_theta_b_corr": d_data_tb,
        "est_margin_p_n_corr": d_est_pn,
        "est_margin_p_b_corr": d_est_pb,
        "total_corr": d_total,
        "interaction_bracket_raw_corr": bracket,
        "doe_interaction_effect_corr": bracket / 2.0,
    }

    # §5.3 conditional bootstrap intervals — conditional on the two realised fitted states.
    intervals: dict[str, tuple[float, float]] | None = None
    if bootstrap is not None and periods:
        cell_factors = {
            "Y_NN": rec_nn.factors_by_period, "Y_Nb": rec_nb.factors_by_period,
            "Y_bN": rec_bn.factors_by_period, "Y_bb": rec_bb.factors_by_period,
        }
        try:
            boot = conditional_bootstrap(cell_factors, anchor, periods, bootstrap, seed=bootstrap_seed)
            intervals = boot.intervals()
        except BootstrapError:
            intervals = None                        # support refused the interval — reported honestly, not fabricated

    def _eff(name: str):
        value = values[name]
        if bootstrap is None:
            return deferred_effect(name, value)
        if intervals is None:
            return refused_effect(name, value)
        return computed_effect(name, value, intervals[name])

    return IPCADifferentialResult(
        bias=bias,
        anchor=anchor_name,
        is_focal=is_focal,
        cells=cells,
        data_margin_theta_n=_eff("data_margin_theta_n_corr"),
        data_margin_theta_b=_eff("data_margin_theta_b_corr"),
        interaction_bracket_raw=_eff("interaction_bracket_raw_corr"),
        doe_interaction_effect=_eff("doe_interaction_effect_corr"),
        est_margin_p_n=_eff("est_margin_p_n_corr"),
        est_margin_p_b=_eff("est_margin_p_b_corr"),
        total=_eff("total_corr"),
        secondary_endtoend=secondary,
        common_support_n_months=len(periods),
    )


# Single-sort anchor rulebooks (from the gold specs), routed through the audited engine. drf is a
# bivariate BBW factor (run_bbw_factor). Defaults give long top group / short bottom group (P10/P1).
#   str  (DRR-2026, Table 1 Panel A col 1): single decile sort on the reversal signal
#        prior_1m_excess_return → xret (D27 concept→column table), value-weighted.
#   mom6 (JNPS-2013): single decile sort on the 6-month momentum signal, EQUAL-weighted.
_ANCHOR_RULEBOOKS: dict[str, dict] = {
    "str":  {"score": "xret", "groups": 10, "weighting": "by_size"},
    "mom6": {"score": "mom6", "groups": 10, "weighting": "equal"},
}


def _anchor_series(view_panel: pd.DataFrame, anchor_name: str) -> pd.Series:
    """Build a fixed anchor long-short return series (indexed by return-month pd.Period) from a
    view() panel via the audited characteristic-sort engine. drf routes through run_bbw_factor
    (bivariate); str/mom6 through run_characteristic_sort (single sort) per their gold specs."""
    if anchor_name == "drf":
        result = run_bbw_factor(view_panel, "drf")
    elif anchor_name in _ANCHOR_RULEBOOKS:
        result = run_characteristic_sort(view_panel, dict(_ANCHOR_RULEBOOKS[anchor_name]))
    else:
        raise ValueError(
            f"unknown anchor {anchor_name!r}; known: {['str', 'mom6', 'drf']}"
        )
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
    bootstrap: IPCABootstrapConfig | None = None,
    bootstrap_seed: int = 0,
    thresholds_path=None,
) -> IPCADifferentialResult:
    """Real entry point: build (P_N, P_b) via ``panel_states``, their feeds and the fixed anchor
    from P_N, then the 2x2 with §5.3 conditional bootstrap intervals. Construction biases raise
    ``ConstructionToggleDeferred`` (see panels.py). Real-data validation lives in the experimentalist build."""
    lam = lam or load_ipca_lambda(thresholds_path)
    gate = gate or load_ipca_projection_gate(thresholds_path)
    bootstrap = bootstrap or load_ipca_bootstrap_config(thresholds_path)
    reporting = load_ipca_reporting(thresholds_path)
    is_focal = reporting.focal_pairs.get(bias) == anchor_name

    p_n, p_b = panel_states(bias, maximal, signals)
    family_n = "corr"
    family_b = "raw" if bias == "meas_err" else "corr"
    feed_n = build_cell_feed(p_n, reg, family_n)
    feed_b = build_cell_feed(p_b, reg, family_b)
    anchor = _anchor_series(p_n, anchor_name)                  # fixed anchor from P_N
    return differential_from_feeds(
        bias, anchor_name, feed_n, feed_b, anchor, lam, gate,
        is_focal=is_focal, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
    )
