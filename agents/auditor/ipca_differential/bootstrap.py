"""
bootstrap.py — the conditional moving-block bootstrap for the IPCA differential (spec §5.3, D-A46).

The cell value Y_{P,Θ} is the OLS intercept of the FIXED anchor return on the factors recovered
under the frozen state Θ, over the common-support months. This bootstrap quantifies the SAMPLING
uncertainty in that regression, **conditional on the two realised fitted states**: the recovered
factor series are held fixed (no re-fit) — only which months enter the regression varies.

It reuses the Auditor's synchronised fixed-block scheme (``agents/auditor/checks/bootstrap.py``):

  * fixed block length ℓ = max(H_max, block_length_months) — a literal holding-horizon floor;
  * ONE common block sequence across all four cells per replicate, so the differential contrasts
    (Δ_data, Δ_est, the bracket I) are comonotone — independent resampling would inflate them;
  * a block length that exceeds the support, or leaves fewer than B_min effective blocks, is a
    REFUSAL (``BootstrapError``), never a parameter to squeeze.

It does NOT estimate the additional uncertainty from re-fitting Θ under repeated samples (§5.3) —
that is the stability diagnostic's separate, non-interval job (§5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agents.auditor.checks.bootstrap import (
    BootstrapError,
    block_length,
    circular_block_indices,
    effective_blocks,
)
from agents.auditor.thresholds import IPCABootstrapConfig

# The four cell labels, in the fixed 2x2 order.
_CELLS = ("Y_NN", "Y_Nb", "Y_bN", "Y_bb")

# Effect draws produced (correction orientation, _corr suffix). Keys mirror the schema field names.
_EFFECTS = (
    "data_margin_theta_n_corr",
    "data_margin_theta_b_corr",
    "est_margin_p_n_corr",
    "est_margin_p_b_corr",
    "total_corr",
    "interaction_bracket_raw_corr",
    "doe_interaction_effect_corr",
)


def _ols_intercept(y: np.ndarray, X: np.ndarray) -> float:
    """OLS intercept of y on [1, X]."""
    design = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0])


@dataclass(frozen=True, eq=False)
class ConditionalBootstrapResult:
    """Per-effect bootstrap draws + the realised block geometry. Intervals are percentile CIs."""

    n_replicates: int
    block_length: int
    effective_blocks: int
    t_common: int
    alpha: float
    conditioning_label: str
    draws: Mapping[str, np.ndarray]         # effect name -> (B,) draws

    def interval(self, name: str, alpha: float | None = None) -> tuple[float, float]:
        a = self.alpha if alpha is None else alpha
        lo, hi = np.percentile(self.draws[name], [100 * a / 2, 100 * (1 - a / 2)])
        return float(lo), float(hi)

    def intervals(self, alpha: float | None = None) -> dict[str, tuple[float, float]]:
        return {name: self.interval(name, alpha) for name in self.draws}


def conditional_bootstrap(
    cell_factors: Mapping[str, Mapping[pd.Period, np.ndarray]],
    anchor: pd.Series,
    periods: Sequence[pd.Period],
    cfg: IPCABootstrapConfig,
    *,
    holding_period: int | None = None,
    seed: int = 0,
) -> ConditionalBootstrapResult:
    """Bootstrap the four cell alphas over the regression sample (the common-support months),
    conditional on the fitted states. ``cell_factors[label][period]`` is the recovered factor
    vector for that cell/month; ``anchor`` is the fixed test-asset return indexed by period.

    Raises ``BootstrapError`` if the block length is incompatible with the common support."""
    t = len(periods)
    ell = block_length(
        cfg.holding_period_default if holding_period is None else holding_period,
        cfg.block_length_months,
    )
    if t == 0:
        raise BootstrapError("empty common support — no regression sample to bootstrap")
    if ell >= t:
        raise BootstrapError(
            f"block length ℓ={ell} >= T_common={t}; the holding-horizon floor exceeds the common "
            "support (a refusal condition, §6.2)"
        )
    eff = effective_blocks(t, ell)
    if eff < cfg.min_effective_blocks:
        raise BootstrapError(
            f"only {eff} effective blocks (ℓ={ell}, T_common={t}); need "
            f">= {cfg.min_effective_blocks} (§4.2/§6.2)"
        )
    if not anchor.index.is_unique:
        raise BootstrapError("anchor index has duplicate periods — cannot align the regression sample")

    # Aligned arrays over the common support (built once; conditioning on the fixed factors).
    y = np.array([float(anchor.loc[p]) for p in periods], dtype=np.float64)
    X = {
        label: np.array([np.asarray(cell_factors[label][p], dtype=np.float64) for p in periods])
        for label in _CELLS
    }

    rng = np.random.default_rng(seed)
    y_draws = {label: np.empty(cfg.n_replicates) for label in _CELLS}
    for b in range(cfg.n_replicates):
        pos = circular_block_indices(t, ell, rng)          # ONE common sequence for all four cells
        y_b = y[pos]
        for label in _CELLS:
            y_draws[label][b] = _ols_intercept(y_b, X[label][pos])

    ynn, ynb, ybn, ybb = (y_draws[k] for k in _CELLS)
    bracket = ynn - ybn - ynb + ybb
    draws = {
        "data_margin_theta_n_corr": ynn - ybn,
        "data_margin_theta_b_corr": ynb - ybb,
        "est_margin_p_n_corr": ynn - ynb,
        "est_margin_p_b_corr": ybn - ybb,
        "total_corr": ynn - ybb,
        "interaction_bracket_raw_corr": bracket,
        "doe_interaction_effect_corr": bracket / 2.0,
    }
    assert set(draws) == set(_EFFECTS), "bootstrap draw keys drifted from _EFFECTS"
    # Tripwire: a "computed" interval must be finite. Recovered factors on gate-valid periods and
    # finite anchor values make this unreachable today; if it ever fires, the differential catches
    # BootstrapError and REFUSES the interval rather than minting a fabricated (nan, nan) CI (§6.2).
    if not all(np.all(np.isfinite(d)) for d in draws.values()):
        raise BootstrapError("non-finite bootstrap draw — interval refused, not fabricated (§6.2)")
    return ConditionalBootstrapResult(
        n_replicates=cfg.n_replicates,
        block_length=ell,
        effective_blocks=eff,
        t_common=t,
        alpha=cfg.alpha,
        conditioning_label=cfg.conditioning_label,
        draws=draws,
    )
