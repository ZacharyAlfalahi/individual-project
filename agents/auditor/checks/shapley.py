"""
shapley.py — additive attribution of the endpoint gap (§5.3).

Named for the objective (an exactly-additive allocation that sums to the endpoint
gap), not the algorithm. From the Möbius/Harsanyi dividends:

    φ_i = Σ_{T ∋ i} h(T) / |T|

Every Y(S) is an observed cell, so this is exact enumeration — no Monte-Carlo.

Efficiency (Σ_i φ_i = Y(N) - Y(∅)) is a theorem, so a violation is a BUG, not
data: `shapley_result` asserts it to numerical tolerance and raises on failure.

The percentage-denominator guard (§5.3, D-A20): shares φ_i / Δ_correction are
unstable when |Δ| is near zero and can be negative or exceed 100%. Metric-unit
contributions are always emitted; percentages only when |Δ| exceeds the
pre-registered minimum — otherwise every share is None (the report emits null,
not a number).
"""

from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

from ..schemas.decomposition import ShapleyResult
from .algebra import harsanyi_dividends

# Efficiency is exact in theory; allow only floating-point slack.
EFFICIENCY_RTOL = 1e-9
EFFICIENCY_ATOL = 1e-12


class EfficiencyViolation(AssertionError):
    """Σφ_i did not equal Y(N) - Y(∅) to tolerance — a bug in the transform, not
    a property of the data."""


def shapley_values(
    harsanyi: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> dict:
    """φ_i = Σ_{T ∋ i} h(T)/|T| for each factor i."""
    phi: dict = {i: 0.0 for i in toggles}
    for T, h in harsanyi.items():
        if not T:
            continue
        share = h / len(T)
        for i in T:
            phi[i] += share
    return phi


def shapley_result(
    Y: Mapping[frozenset, float],
    toggles: Sequence[Hashable],
    *,
    percentage_denominator_min: float,
) -> ShapleyResult:
    """Compute φ_i, assert efficiency, and apply the percentage guard.

    `percentage_denominator_min` is the pre-registered |Δ_correction| floor below
    which shares are withheld (emitted as None). It is passed in explicitly so the
    caller owns the (fail-loud) threshold read — tests supply it directly.
    """
    h = harsanyi_dividends(Y, toggles)
    phi = shapley_values(h, toggles)

    empty = frozenset()
    full = frozenset(toggles)
    gap = Y[full] - Y[empty]

    total = math.fsum(phi.values())
    residual = total - gap
    if not math.isclose(total, gap, rel_tol=EFFICIENCY_RTOL, abs_tol=EFFICIENCY_ATOL):
        raise EfficiencyViolation(
            f"Σφ_i = {total!r} but Y(N)-Y(∅) = {gap!r} (residual {residual!r}); "
            "the Shapley decomposition is not efficient — a transform bug"
        )

    denominator_ok = abs(gap) > percentage_denominator_min
    if denominator_ok:
        shares = {i: phi[i] / gap for i in toggles}
    else:
        # Guard tripped: shares would be a division artefact. Emit null, not a number.
        shares = {i: None for i in toggles}

    return ShapleyResult(
        values=phi,
        shares=shares,
        endpoint_gap=gap,
        efficiency_residual=residual,
        denominator_ok=denominator_ok,
    )
