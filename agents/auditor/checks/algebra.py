"""
algebra.py — the three lattice readings as exact linear transforms (§5).

Given the metric value Y(S) at every cell — S ⊆ N is the set of toggles held ON —
this module computes:

  * corner marginals    Δ_add[i], Δ_loo[i], and the interaction bracket (§5.1a)
  * Möbius / Harsanyi dividends h(T)                                    (§5.1b)
  * Walsh coefficients γ_T (orthogonal ±1 basis)                        (§5.1c)
  * DOE effects E_T = 2·γ_T (the average ON-minus-OFF difference)       (§5.1c)

Every quantity is an exact linear functional of the 2^k cell values — no model is
fitted, nothing is selected (§5.2). The functions are generic over an ordered
tuple of factor labels so the Layer A validator can drive them with synthetic
factors; the schema-producing wrappers assume the real ToggleId labels.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Hashable, Mapping, Sequence

from ..schemas.decomposition import (
    BiasClassPartition,
    CornerMarginals,
    SaturatedBasis,
)
from ..schemas.toggle import construction_toggles, data_quality_toggles


def all_subsets(toggles: Sequence[Hashable]) -> list[frozenset]:
    """Every subset of the factor set, as frozensets (2^k of them)."""
    out: list[frozenset] = []
    for r in range(len(toggles) + 1):
        out.extend(frozenset(c) for c in combinations(toggles, r))
    return out


def _require_complete(Y: Mapping[frozenset, float], toggles: Sequence[Hashable]) -> None:
    subsets = all_subsets(toggles)
    missing = [s for s in subsets if s not in Y]
    if missing:
        raise KeyError(
            f"Y is missing {len(missing)} of 2^{len(toggles)} cells; a saturated "
            "transform needs every cell value"
        )


# ---------------------------------------------------------------------------
# Corner marginals (§5.1a)
# ---------------------------------------------------------------------------

def corner_marginals(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> tuple[dict, dict, dict]:
    """Return (add_in, leave_out, bracket) dicts keyed by factor.

      add_in[i]    = Y({i}) - Y(∅)
      leave_out[i] = Y(N)   - Y(N \\ {i})
      bracket[i]   = add_in[i] - leave_out[i]
    """
    _require_complete(Y, toggles)
    empty = frozenset()
    full = frozenset(toggles)
    y_empty = Y[empty]
    y_full = Y[full]
    add_in: dict = {}
    leave_out: dict = {}
    bracket: dict = {}
    for i in toggles:
        add_in[i] = Y[frozenset({i})] - y_empty
        leave_out[i] = y_full - Y[full - {i}]
        bracket[i] = add_in[i] - leave_out[i]
    return add_in, leave_out, bracket


# ---------------------------------------------------------------------------
# Möbius / Harsanyi dividends (§5.1b)
# ---------------------------------------------------------------------------

def harsanyi_dividends(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> dict[frozenset, float]:
    """h(T) = Σ_{U ⊆ T} (-1)^{|T|-|U|} Y(U), for every T ⊆ N. Exact and
    invertible: Y(S) = Σ_{T ⊆ S} h(T). Note h({i}) = Δ_add[i]."""
    _require_complete(Y, toggles)
    h: dict[frozenset, float] = {}
    for T in all_subsets(toggles):
        t = len(T)
        acc = 0.0
        for r in range(t + 1):
            for U in combinations(sorted(T, key=lambda x: str(x)), r):
                sign = -1.0 if (t - r) % 2 else 1.0
                acc += sign * Y[frozenset(U)]
        h[T] = acc
    return h


# ---------------------------------------------------------------------------
# Walsh coefficients and DOE effects (§5.1c)
# ---------------------------------------------------------------------------

def walsh_coefficients(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> dict[frozenset, float]:
    """γ_T = 2^{-k} Σ_S Y(S) · Π_{i∈T} z_i(S), with z_i = +1 if i∈S else -1.

    Since Π_{i∈T} z_i(S) = (-1)^{|T \\ S|}, this is the orthogonal ±1-basis
    expansion coefficient. It IS orthogonal (unlike the Möbius basis)."""
    _require_complete(Y, toggles)
    k = len(toggles)
    scale = 1.0 / (2**k)
    gamma: dict[frozenset, float] = {}
    subsets = all_subsets(toggles)
    for T in subsets:
        acc = 0.0
        for S in subsets:
            parity = len(T - S)  # |T \ S|
            sign = -1.0 if parity % 2 else 1.0
            acc += sign * Y[S]
        gamma[T] = scale * acc
    return gamma


def doe_effects(walsh: Mapping[frozenset, float]) -> dict[frozenset, float]:
    """E_T = 2·γ_T — the conventional difference-in-means factorial effect (the
    average ON-minus-OFF difference). This is what appears in every table, the
    Bayesian model and the FDR family (§5.1c). The empty-set coefficient γ_∅ is
    the grand mean; E_∅ = 2·γ_∅ is retained only for completeness."""
    return {T: 2.0 * g for T, g in walsh.items()}


# ---------------------------------------------------------------------------
# Schema-producing wrappers (assume ToggleId labels)
# ---------------------------------------------------------------------------

def corner_marginals_result(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> CornerMarginals:
    add_in, leave_out, bracket = corner_marginals(Y, toggles)
    return CornerMarginals(add_in=add_in, leave_out=leave_out, bracket=bracket)


def saturated_basis(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> SaturatedBasis:
    walsh = walsh_coefficients(Y, toggles)
    return SaturatedBasis(
        harsanyi=harsanyi_dividends(Y, toggles),
        walsh=walsh,
        doe=doe_effects(walsh),
    )


def bias_class_partition(
    harsanyi: Mapping[frozenset, float],
    toggles: Sequence[Hashable],
) -> BiasClassPartition:
    """Partition the endpoint gap Y(N)-Y(∅) by `bias_class` (ADR §5.2), from the
    Harsanyi dividends restricted to coalitions PRESENT in `toggles` (the runnable
    lattice). Every non-empty coalition falls in exactly one bucket:

      method       — S touches only methodological-construction toggles
      data_quality — S touches only data-quality-correction toggles
      cross_class  — S mixes both classes (a modulation term, |S|>1)

    Each component is **None when its bucket of present coalitions is empty** — never
    a summed-from-nothing 0.0 (CF-2 / ADR §5.4). The three buckets sum to
    `endpoint_gap` (= Σ present non-empty h(S) = Y(N)-Y(∅) by Möbius);
    `reconciliation_residual` is the float error of that identity.

    NOTE: the classes are read from the registry (`construction_toggles()` /
    `data_quality_toggles()`), so a future third class extends here with no design
    change. As configured, `data_quality_toggles()` is `("meas_err",)`."""
    present = set(toggles)
    m_set = set(construction_toggles()) & present
    d_set = set(data_quality_toggles()) & present

    method_bucket: list[float] = []
    dq_bucket: list[float] = []
    cross_bucket: list[float] = []
    all_present: list[float] = []
    for S, h in harsanyi.items():
        if not S or not (S <= present):
            # h(∅) is the grand-mean offset (not part of the gap); coalitions
            # outside the lattice are ignored defensively (a caller may pass the
            # full 2^k harsanyi with a reduced toggle set).
            continue
        all_present.append(h)
        touches_m = bool(S & m_set)
        touches_d = bool(S & d_set)
        if touches_m and touches_d:
            cross_bucket.append(h)
        elif touches_d:
            dq_bucket.append(h)
        else:                       # touches only construction toggles
            method_bucket.append(h)

    def _sum_or_none(bucket: list[float]) -> float | None:
        return math.fsum(bucket) if bucket else None

    method = _sum_or_none(method_bucket)
    data_quality = _sum_or_none(dq_bucket)
    cross_class = _sum_or_none(cross_bucket)
    endpoint_gap = math.fsum(all_present)
    # Explicit None->0 (not `x or 0.0`) so a genuine 0.0 component is unambiguous and
    # the None-not-zero discipline reads clearly next to the None-returning buckets.
    def _z(v: float | None) -> float:
        return v if v is not None else 0.0

    residual = _z(method) + _z(data_quality) + _z(cross_class) - endpoint_gap
    return BiasClassPartition(
        methodological_construction_component=method,
        data_quality_component=data_quality,
        cross_class_modulation=cross_class,
        endpoint_gap=endpoint_gap,
        reconciliation_residual=residual,
        data_quality_present=bool(d_set),
    )
