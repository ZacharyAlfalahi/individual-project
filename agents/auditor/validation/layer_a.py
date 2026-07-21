"""
layer_a.py — Layer A algebraic transform validation (§10.2).

The first and cheapest gate: no backtest, no synthetic bond panel, no engine run.
Construct synthetic cell-value functions with ANALYTICALLY KNOWN coefficients,
evaluate all 2^k cells, and assert exact recovery (to numerical tolerance) of:

  * Walsh coefficients γ_T and the DOE scaling E_T = 2·γ_T
  * Möbius/Harsanyi dividends h(T) via the inversion Y(S) = Σ_{T⊆S} h(T)
  * Shapley values φ_i — cross-checked against an INDEPENDENT permutation oracle
  * efficiency Σφ_i = Δ_correction
  * interaction signs and magnitudes — every T with |T| ≥ 2, all higher orders

"If the transforms are wrong, nothing downstream is worth running" (§17, step 8),
and finding that out with algebra costs milliseconds. This module is imported only
by tests / the orchestrator's gate — never by the per-audit analytical path.

Ground truth is a multilinear polynomial in the ±1 (Walsh) basis:

    Y(S) = Σ_T a_T · Π_{i∈T} z_i(S),   z_i = +1 if i∈S else -1

whose Walsh coefficients are exactly the planted a_T. Everything else (Möbius,
Shapley) is checked by an algorithm independent of the one under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Hashable, Mapping, Sequence

import numpy as np

from ..checks.algebra import (
    all_subsets,
    corner_marginals,
    doe_effects,
    harsanyi_dividends,
    walsh_coefficients,
)
from ..checks.shapley import shapley_values

DEFAULT_ATOL = 1e-9
DEFAULT_RTOL = 1e-9


class AlgebraRecoveryError(AssertionError):
    """A transform failed to recover its analytically-known target to tolerance."""


@dataclass(frozen=True)
class LayerAReport:
    k_values: tuple[int, ...]
    n_trials: int
    max_abs_error: float
    checks_run: int
    passed: bool


def _evaluate_polynomial(
    coeffs: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> dict[frozenset, float]:
    """Y(S) = Σ_T a_T · (-1)^{|T \\ S|} for every S ⊆ N."""
    Y: dict[frozenset, float] = {}
    for S in all_subsets(toggles):
        acc = 0.0
        for T, a in coeffs.items():
            parity = len(T - S)
            acc += a * (-1.0 if parity % 2 else 1.0)
        Y[S] = acc
    return Y


def _shapley_by_permutation(
    Y: Mapping[frozenset, float], toggles: Sequence[Hashable]
) -> dict:
    """Independent Shapley oracle: φ_i = (1/k!) Σ_orderings [Y(pre∪{i}) - Y(pre)].
    A different algorithm from the Harsanyi-dividend formula under test — so
    agreement is a genuine cross-check, not a tautology. Exact for k ≤ 5."""
    n = len(toggles)
    phi = {i: 0.0 for i in toggles}
    n_perms = 0
    for order in permutations(toggles):
        n_perms += 1
        prefix: set = set()
        for i in order:
            before = Y[frozenset(prefix)]
            prefix.add(i)
            after = Y[frozenset(prefix)]
            phi[i] += after - before
    return {i: phi[i] / n_perms for i in toggles} if n else phi


def validate_algebra(
    k_values: Sequence[int] = (1, 2, 3, 4, 5),
    n_trials: int = 200,
    seed: int = 20260721,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> LayerAReport:
    """Run the algebraic recovery battery. Raises AlgebraRecoveryError on the
    first mismatch (with a locating message); returns a LayerAReport otherwise."""
    rng = np.random.default_rng(seed)
    max_err = 0.0
    checks = 0

    def _close(got: float, want: float, what: str) -> None:
        nonlocal max_err, checks
        checks += 1
        err = abs(got - want)
        max_err = max(max_err, err)
        if not np.isclose(got, want, atol=atol, rtol=rtol):
            raise AlgebraRecoveryError(f"{what}: got {got!r}, want {want!r} (|err|={err})")

    for k in k_values:
        toggles = tuple(f"t{j}" for j in range(k))
        subsets = all_subsets(toggles)
        for _ in range(n_trials):
            planted = {T: float(rng.normal(0.0, 1.0)) for T in subsets}
            Y = _evaluate_polynomial(planted, toggles)

            # 1. Walsh recovers the planted ±1-basis coefficients exactly.
            gamma = walsh_coefficients(Y, toggles)
            for T in subsets:
                _close(gamma[T], planted[T], f"walsh γ_{sorted(T)} (k={k})")

            # 2. DOE scaling E_T = 2·γ_T.
            E = doe_effects(gamma)
            for T in subsets:
                _close(E[T], 2.0 * planted[T], f"doe E_{sorted(T)} (k={k})")

            # 3. Möbius inversion reconstructs every cell: Y(S) = Σ_{T⊆S} h(T).
            h = harsanyi_dividends(Y, toggles)
            for S in subsets:
                recon = sum(h[T] for T in subsets if T <= S)
                _close(recon, Y[S], f"möbius reconstruction Y({sorted(S)}) (k={k})")

            # 4. h({i}) == Δ_add[i]; leave-one-out / bracket sanity.
            add_in, leave_out, bracket = corner_marginals(Y, toggles)
            for i in toggles:
                _close(h[frozenset({i})], add_in[i], f"h({{{i}}})==Δ_add (k={k})")
                _close(
                    bracket[i], add_in[i] - leave_out[i], f"bracket {i} (k={k})"
                )

            # 5. Shapley: module vs independent permutation oracle, and efficiency.
            phi = shapley_values(h, toggles)
            phi_oracle = _shapley_by_permutation(Y, toggles)
            for i in toggles:
                _close(phi[i], phi_oracle[i], f"shapley φ_{i} vs oracle (k={k})")
            gap = Y[frozenset(toggles)] - Y[frozenset()]
            _close(sum(phi.values()), gap, f"efficiency Σφ==gap (k={k})")

    return LayerAReport(
        k_values=tuple(k_values),
        n_trials=n_trials,
        max_abs_error=max_err,
        checks_run=checks,
        passed=True,
    )
