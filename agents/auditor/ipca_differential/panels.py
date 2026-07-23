"""
panels.py — the leave-one-out-from-corrected panel states and per-cell feeds (spec §2.1).

Per bias b: P_N (all registered corrections ON) and P_{N\\b} (P_N with bias b returned to
as-published) — the same orientation as the recovery sweep's primary estimand. Built by flipping
exactly one axis of the corrected ``RunConfig`` via ``TOGGLE_AXES`` (the single toggle→config
source of truth) and materialising through the pure ``views.view``.

**Scope note (amendment / precomputed-signal nuance).** The three *panel_view* biases (meas_err,
stale_price, survivorship) propagate cleanly into the IPCA feed through ``view()`` — family
selection, the stale mask, and terminal-row membership all change the returns/cross-section the
estimator sees. The two *construction* biases (lib_gap = characteristic timing, lab_trim = a return
trim) are NOT consumed by ``view()``; propagating them into the IPCA feed needs the signal-lag
decoupling and the per-paper trim spec (plus per-panel signal recomputation) — a real-data
modelling decision left unresolved. Rather than silently return identical panels (a no-op
that would fake I = 0), ``panel_states`` raises ``ConstructionToggleDeferred`` for those two. The
contract + engine + build gates exercise the panel_view biases.
"""

from __future__ import annotations

import dataclasses

import pandas as pd

from agents.auditor.schemas.toggle import TOGGLE_AXES, ToggleId
from agents.quant.library.ipca_feed import IPCAFeed, build_ipca_feed, feed_matrices
from agents.quant.library.run_config import RunConfig, corrected
from agents.quant.library.views import view

# Biases whose correction lives in config.panel_view (consumed by view()).
PANEL_VIEW_BIASES: tuple[ToggleId, ...] = ("meas_err", "stale_price", "survivorship")
# Biases whose correction lives in config.construction (NOT consumed by view()).
CONSTRUCTION_BIASES: tuple[ToggleId, ...] = ("lib_gap", "lab_trim")

# The canonical instrument columns a view() panel must carry to build an IPCA feed.
# str_reversal is the instrument-time excess return (xret); the rest are family-resolved signals.
_MERGED_SIGNAL_COLUMNS = ("mom6", "var_5pct", "gamma_illiq", "bond_vol")


class ConstructionToggleDeferred(NotImplementedError):
    """lib_gap / lab_trim IPCA-feed propagation is an unresolved modelling decision (§3.2 focal
    pairs lib_gap→str, lab_trim→mom6 are executed then). This build handles the three
    panel_view biases, which propagate through view()."""


def config_leave_out(bias: ToggleId) -> tuple[RunConfig, RunConfig]:
    """Return (P_N config, P_{N\\b} config): the corrected config, and the corrected config with
    bias b's single axis returned to its as-published (OFF) value."""
    base = corrected()
    axis = TOGGLE_AXES[bias]
    if axis.block == "panel_view":
        flipped = dataclasses.replace(base.panel_view, **{axis.field: axis.off})
        off = dataclasses.replace(base, panel_view=flipped)
    else:
        flipped = dataclasses.replace(base.construction, **{axis.field: axis.off})
        off = dataclasses.replace(base, construction=flipped)
    return base, off


def panel_states(
    bias: ToggleId,
    maximal: pd.DataFrame,
    signals: pd.DataFrame | None = None,
    **view_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise (P_N, P_{N\\b}) for bias b via ``view()``. Raises ``ConstructionToggleDeferred``
    for the construction biases (lib_gap, lab_trim) — see the scope note."""
    if bias in CONSTRUCTION_BIASES:
        raise ConstructionToggleDeferred(
            f"bias {bias!r} is a construction toggle; its IPCA-feed propagation (characteristic "
            "timing / return trim) is not implemented here. This build handles the "
            f"panel_view biases {PANEL_VIEW_BIASES}."
        )
    base, off = config_leave_out(bias)
    p_n = view(maximal, base, signals=signals, **view_kwargs)
    p_b = view(maximal, off, signals=signals, **view_kwargs)
    return p_n, p_b


def to_merged(view_panel: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical 7-instrument merged frame from a ``view()`` panel. str_reversal is the
    instrument-time excess return (xret); the four family-resolved signals and rating/maturity pass
    through. Raises KeyError naming any missing column."""
    required = ("cusip", "date", "xret", "rating", "time_to_maturity", *_MERGED_SIGNAL_COLUMNS)
    missing = [c for c in required if c not in view_panel.columns]
    if missing:
        raise KeyError(
            f"view panel missing columns for the IPCA feed: {missing}. Pass the family-indexed "
            "IPCA signals (mom6, var_5pct, gamma_illiq, bond_vol) to view() via signals=."
        )
    return pd.DataFrame({
        "cusip": view_panel["cusip"].to_numpy(),
        "date": view_panel["date"].to_numpy(),
        "str_reversal": view_panel["xret"].to_numpy(),
        "mom6": view_panel["mom6"].to_numpy(),
        "var_5pct": view_panel["var_5pct"].to_numpy(),
        "gamma_illiq": view_panel["gamma_illiq"].to_numpy(),
        "rating": view_panel["rating"].to_numpy(),
        "time_to_maturity": view_panel["time_to_maturity"].to_numpy(),
        "bond_vol": view_panel["bond_vol"].to_numpy(),
    })


def build_cell_feed(view_panel: pd.DataFrame, reg: dict, family: str) -> IPCAFeed:
    """view() panel → canonical merged frame → build_ipca_feed → per-month matrices (IPCAFeed)."""
    merged = to_merged(view_panel)
    out, _ = build_ipca_feed(merged, reg, family, validate=True)
    return feed_matrices(out)
