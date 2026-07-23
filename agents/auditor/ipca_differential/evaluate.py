"""
evaluate.py — the frozen evaluation map (spec §5.1, D-A45).

    Y_{P,Θ} = α( r_anchor_fixed , f(P; Θ̂) )

"pricing-error transmission to a fixed corrected test asset, conditional on all other registered
corrections being ON." Per cell and month:

  1. **recompute** the factor realisation from the EVALUATION panel P:
         f̂_t = (Γ̂'W_tΓ̂)⁻¹ Γ̂'(x_t − W_tΓα)     [ipca._oos_factor_realization]
     using ONLY the frozen projection components (Γβ, Γα=None) via ``state.projection_params()`` —
     never the fitted ``state.factors`` (that is the fitting-panel path; reusing it would collapse
     the data channel). Months that fail the projection gate are dropped;
  2. run ONE OLS of the FIXED anchor return on {f̂_t} + intercept over the post-gate common support;
     the **intercept is the cell value** Y_{P,Θ}.

The alpha of a regression on a factor set is invariant to invertible rotations of a given span, so
this is an intrinsic, out-of-objective pricing metric — not the ALS minimand. Levels under the
frozen benchmark are not interpretable; only gaps are (disclosed; §3.1).

**Chimera note (§3.1 / §9.11, amendment 3):** pricing the corrected fixed anchor against a
dirty-panel factor set is an evaluation *overlay*, not a construction chimera. The view-layer
chimera guard (``views._resolve_signals_family``) fires only on a signal-family *mismatch*; on this
path every feed is built with family-consistent signals, so it never triggers. No exemption is
added to the shared construction guard — the ``benchmark_overlay`` scope stays inside this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agents.auditor.thresholds import IPCAProjectionGate
from agents.quant.library.ipca import _oos_factor_realization, build_sufficient_stats
from agents.quant.library.ipca_feed import IPCAFeed

from .frozen_state import FrozenIPCAState
from .projection_gate import GatedProjection, gate_cell


@dataclass(frozen=True)
class FactorRecovery:
    """Factors recovered by frozen projection on the evaluation panel, per valid return month."""

    factors_by_period: dict[pd.Period, np.ndarray]   # recovered f̂_t keyed by return month
    gated: GatedProjection

    @property
    def valid_periods(self) -> tuple[pd.Period, ...]:
        return self.gated.valid_periods


def recover_factor_series(
    feed: IPCAFeed, state: FrozenIPCAState, gate: IPCAProjectionGate
) -> FactorRecovery:
    """Recompute the factor path on the evaluation panel `feed` under the frozen state `state`,
    subject to the projection gate. Consumes ONLY ``state.projection_params()`` — the fitted
    ``state.factors`` is never read (the consumption boundary, §4.2)."""
    pp = state.projection_params()                       # (gamma_beta, gamma_alpha=None) — no factors
    gated = gate_cell(feed, pp.gamma_beta, gate)
    stats = build_sufficient_stats(feed.Z, feed.R, feed.months)
    factors_by_period: dict[pd.Period, np.ndarray] = {}
    for pos, period in zip(gated.valid_positions, gated.valid_periods):
        f = _oos_factor_realization(stats.W[pos], stats.x[pos], pp.gamma_beta, pp.gamma_alpha)
        factors_by_period[period] = np.asarray(f, dtype=np.float64)
    return FactorRecovery(factors_by_period=factors_by_period, gated=gated)


def alpha_on(
    anchor: pd.Series,
    factors_by_period: Mapping[pd.Period, np.ndarray],
    periods: Sequence[pd.Period],
) -> float:
    """OLS intercept of the fixed anchor return on the recovered factors over `periods` (the
    common support). `periods` must be a subset of both the anchor index and the recovered factors.
    Returns the cell value Y_{P,Θ} = the pricing-error alpha (out-of-objective)."""
    if len(periods) == 0:
        return float("nan")
    y = np.array([float(anchor.loc[p]) for p in periods], dtype=np.float64)
    X = np.array([np.asarray(factors_by_period[p], dtype=np.float64) for p in periods])  # (n, K)
    design = np.column_stack([np.ones(len(periods)), X])                                 # (n, 1+K)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0])                                                                # the intercept
