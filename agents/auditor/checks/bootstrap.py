"""
bootstrap.py — the synchronised fixed-block vector bootstrap (§6).

NORMATIVE METHOD (§6.1, D-A29): the moving (circular) block bootstrap with a
FIXED block length applied to the 2^k-cell return vector process {r_t}, using ONE
common sequence of sampled blocks across all cells, on the common support.

  * Fixed block length (not stationary): ℓ = max(H_max, ℓ_data-driven). Every
    realised block is exactly ℓ long, so the holding-horizon floor is LITERAL —
    a stationary bootstrap's geometric blocks would let short blocks sever the
    overlapping-holding dependence the floor exists to preserve (§6.2). H_max is
    the strategy's maximum overlapping holding horizon (its holding period);
    ℓ_data-driven is pre-registered.
  * ONE common block sequence across all cells per replicate — preserves the
    cross-cell comonotonicity that makes the differential design powerful.
    Independent resampling would inflate every interval.
  * Metrics ONLY, never the engine (§14): each replicate resamples the already
    computed return matrix and recomputes metrics; the §5 transforms (linear) are
    applied per replicate. The engine is deterministic given a panel — the
    uncertainty is sampling uncertainty in the returns.

A block length that exceeds the common support, or leaves fewer than B_min
effective blocks, is a REFUSAL condition, not a parameter to squeeze (§4.2, §6.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns

from ..schemas.lattice_types import CellReturns
from ..schemas.toggle import ToggleId
from .algebra import doe_effects, harsanyi_dividends, walsh_coefficients
from .shapley import shapley_values
from .support import return_matrix


class BootstrapError(RuntimeError):
    """The block length is incompatible with the common support (a refusal
    condition, §6.2), or the return matrix is empty."""


def block_length(holding_period: int, data_driven_months: int) -> int:
    """ℓ = max(H_max, ℓ_data-driven), a LITERAL floor (§6.2). H_max is the maximum
    overlapping holding horizon = the strategy's holding period."""
    return max(int(holding_period), int(data_driven_months))


def effective_blocks(t_common: int, ell: int) -> int:
    """⌊T_common / ℓ⌋ — the number of non-overlapping blocks the support admits."""
    return t_common // ell


def circular_block_indices(t: int, ell: int, rng: np.random.Generator) -> np.ndarray:
    """A length-T resample index sequence from the circular (moving) block
    bootstrap with fixed block length ℓ: draw ⌈T/ℓ⌉ start points uniformly in
    [0,T), take a length-ℓ block from each (wrapping around), concatenate, truncate
    to T."""
    n_blocks = int(np.ceil(t / ell))
    starts = rng.integers(0, t, size=n_blocks)
    idx = np.concatenate([(np.arange(s, s + ell) % t) for s in starts])
    return idx[:t]


def stationary_block_indices(t: int, mean_ell: int, rng: np.random.Generator) -> np.ndarray:
    """A length-T resample index sequence from the stationary (Politis–Romano 1994)
    bootstrap: block lengths are geometric with success probability p = 1/ℓ (so the
    EXPECTED length is ℓ = mean_ell), each block starts uniformly in [0,T) and wraps
    around, concatenated until the sequence reaches T, then truncated.

    Pre-registered SENSITIVITY ONLY (D-A29): the fixed-block scheme is normative,
    because the geometric block length constrains only the MEAN — short blocks still
    occur and are precisely the ones that sever the overlapping-holding dependence the
    H_max floor exists to preserve (§6.2). `mean_ell` is set to ℓ = max(H_max, ℓ_data-
    driven) so the expected block length still clears the holding horizon."""
    p = 1.0 / float(mean_ell)
    parts: list[np.ndarray] = []
    filled = 0
    while filled < t:
        start = int(rng.integers(0, t))
        length = int(rng.geometric(p))  # >= 1, mean 1/p = mean_ell
        parts.append(np.arange(start, start + length) % t)
        filled += length
    return np.concatenate(parts)[:t]


def _metric_of(values: np.ndarray, metric: str, months_per_year: int) -> float:
    """The primary metric of one resampled return vector (a functional of the
    values). Uses the shared summarize_returns; the resampled series carries a
    plain RangeIndex (dates are notional after resampling)."""
    summary = summarize_returns(pd.Series(values), None, months_per_year)
    return float(summary[metric])


@dataclass(frozen=True, eq=False)
class BootstrapResult:
    n_replicates: int
    block_length: int
    effective_blocks: int
    t_common: int
    toggles: tuple[ToggleId, ...]
    gap_draws: np.ndarray                       # (B,)
    doe_draws: Mapping[frozenset, np.ndarray]   # subset -> (B,)
    shapley_draws: Mapping[ToggleId, np.ndarray]  # toggle -> (B,)
    scheme: str = "fixed"                       # "fixed" (normative) | "stationary" (D-A29 sensitivity)

    def interval(self, draws: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
        lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    def gap_ci(self, alpha: float = 0.05) -> tuple[float, float]:
        return self.interval(self.gap_draws, alpha)

    def doe_ci(self, alpha: float = 0.05) -> dict[frozenset, tuple[float, float]]:
        return {T: self.interval(d, alpha) for T, d in self.doe_draws.items()}

    def shapley_ci(self, alpha: float = 0.05) -> dict[ToggleId, tuple[float, float]]:
        return {t: self.interval(d, alpha) for t, d in self.shapley_draws.items()}


def first_order_covariance(
    result: "BootstrapResult", toggles: Sequence[ToggleId]
) -> tuple[tuple[ToggleId, ...], np.ndarray]:
    """The k×k sample covariance of the first-order DOE effects {E_i} across the
    bootstrap replicates — the FULL measurement covariance V̂_s the cross-strategy
    hierarchy consumes (D-A32: keep V̂ full, constrain the population to diagonal).

    Built entirely from the existing singleton `doe_draws[frozenset({i})]`, so it costs
    no extra resampling. Rows/columns follow `toggles` — the CALLER must build the matching
    `StrategyVector.estimate`/`coords` in the SAME order, or the joint measurement model is
    scrambled. A single-toggle strategy yields a 1×1 matrix (np.cov collapses to a scalar;
    re-shaped here)."""
    labels = tuple(toggles)
    cols = [np.asarray(result.doe_draws[frozenset({tg})], dtype=float) for tg in labels]
    stacked = np.column_stack(cols)          # (B, k)
    cov = np.atleast_2d(np.cov(stacked, rowvar=False))
    return labels, cov


def run_bootstrap(
    cells: Sequence[CellReturns],
    months: pd.DatetimeIndex,
    toggles: Sequence[ToggleId],
    *,
    n_replicates: int,
    data_driven_block_months: int,
    min_effective_blocks: int,
    holding_period: int,
    metric: str = "average",
    months_per_year: int = 12,
    seed: int = 0,
    scheme: str = "fixed",
) -> BootstrapResult:
    """Run the synchronised block bootstrap and return per-quantity draws.

    `scheme="fixed"` (default, normative §6.1) is the circular moving block bootstrap
    with a LITERAL block length ℓ; `scheme="stationary"` is the Politis–Romano geometric-
    block sensitivity (D-A29), same ℓ as the EXPECTED block length. Only the per-replicate
    index generator differs — the synchronisation, refusal gates, replicate count and §5
    maps are identical — so `scheme="fixed"` reproduces the prior behaviour exactly.

    Raises BootstrapError if the block length is incompatible with the common
    support (ℓ >= T_common, or fewer than `min_effective_blocks` effective blocks)."""
    if scheme not in ("fixed", "stationary"):
        raise BootstrapError(f"unknown bootstrap scheme {scheme!r} (fixed | stationary)")
    keys, R = return_matrix(cells, months)
    t = R.shape[0]
    if t == 0 or R.shape[1] == 0:
        raise BootstrapError("empty return matrix on the common support")
    if np.isnan(R).any():
        raise BootstrapError(
            "the common-support return matrix contains NaN — the support is not "
            "actually common to all cells"
        )

    ell = block_length(holding_period, data_driven_block_months)
    if ell >= t:
        raise BootstrapError(
            f"block length ℓ={ell} >= T_common={t}; the holding-horizon floor "
            "exceeds the common support (a refusal condition, §6.2)"
        )
    eff = effective_blocks(t, ell)
    if eff < min_effective_blocks:
        raise BootstrapError(
            f"only {eff} effective blocks (ℓ={ell}, T_common={t}); need "
            f">= {min_effective_blocks} (§4.2/§6.2)"
        )

    key_index = {k: j for j, k in enumerate(keys)}
    full = frozenset(toggles)
    empty = frozenset()

    rng = np.random.default_rng(seed)
    gap_draws = np.empty(n_replicates)
    doe_draws: dict[frozenset, list] = {}
    shapley_draws: dict[ToggleId, list] = {tg: [] for tg in toggles}

    for b in range(n_replicates):
        # ONE common sequence for all cells (synchronisation); scheme picks the generator.
        if scheme == "stationary":
            pos = stationary_block_indices(t, ell, rng)
        else:
            pos = circular_block_indices(t, ell, rng)
        # Y^(b): the metric of every cell on the SAME resampled positions.
        Yb: dict[frozenset, float] = {}
        for k, j in key_index.items():
            Yb[k] = _metric_of(R[pos, j], metric, months_per_year)
        # Apply the §5 linear maps per replicate.
        h = harsanyi_dividends(Yb, toggles)
        gamma = walsh_coefficients(Yb, toggles)
        E = doe_effects(gamma)
        phi = shapley_values(h, toggles)
        gap_draws[b] = Yb[full] - Yb[empty]
        for T, e in E.items():
            doe_draws.setdefault(T, []).append(e)
        for tg in toggles:
            shapley_draws[tg].append(phi[tg])

    return BootstrapResult(
        n_replicates=n_replicates,
        block_length=ell,
        effective_blocks=eff,
        t_common=t,
        toggles=tuple(toggles),
        gap_draws=gap_draws,
        doe_draws={T: np.asarray(v) for T, v in doe_draws.items()},
        shapley_draws={tg: np.asarray(v) for tg, v in shapley_draws.items()},
        scheme=scheme,
    )
