"""Stage 2 — algebra, Shapley, and the Layer A gate (§5, §10.2).

Layer A is the first validation gate: exact algebraic recovery of every basis and
Shapley, cross-checked against an independent permutation oracle. Plus targeted
tests of the percentage-denominator guard (§5.3) and the efficiency assertion, and
a small hand-computed decomposition.
"""

from __future__ import annotations

import math
from itertools import combinations, permutations

import pytest

from agents.auditor.checks.algebra import (
    corner_marginals,
    doe_effects,
    harsanyi_dividends,
    walsh_coefficients,
)
from agents.auditor.checks.shapley import (
    EfficiencyViolation,
    shapley_result,
    shapley_values,
)
from agents.auditor.validation.layer_a import validate_algebra


# --------------------------------------------------------------------------
# Layer A — the gate
# --------------------------------------------------------------------------

def test_layer_a_recovers_all_bases_and_shapley_for_k_1_to_5():
    report = validate_algebra(k_values=(1, 2, 3, 4, 5), n_trials=60)
    assert report.passed
    assert report.max_abs_error < 1e-9
    # Every trial exercises Walsh, DOE, Möbius, marginals, and two Shapley algos.
    assert report.checks_run > 0


def test_layer_a_detects_a_broken_transform(monkeypatch):
    # Corrupt DOE scaling (E_T = 3·γ_T instead of 2·γ_T) and confirm Layer A fails.
    import agents.auditor.validation.layer_a as la

    def bad_doe(walsh):
        return {T: 3.0 * g for T, g in walsh.items()}

    monkeypatch.setattr(la, "doe_effects", bad_doe)
    with pytest.raises(la.AlgebraRecoveryError, match="doe"):
        validate_algebra(k_values=(2,), n_trials=1)


# --------------------------------------------------------------------------
# A concrete 2-factor decomposition, checked by hand
# --------------------------------------------------------------------------

def _Y2(a, b, c, d):
    # Y(∅)=a, Y({x})=b, Y({y})=c, Y({x,y})=d
    return {
        frozenset(): a,
        frozenset({"x"}): b,
        frozenset({"y"}): c,
        frozenset({"x", "y"}): d,
    }


def test_harsanyi_and_shapley_hand_values():
    Y = _Y2(1.0, 4.0, 2.0, 6.0)
    tog = ("x", "y")
    h = harsanyi_dividends(Y, tog)
    # h(∅)=1; h({x})=b-a=3; h({y})=c-a=1; h({x,y})=d-b-c+a = 6-4-2+1 = 1
    assert h[frozenset()] == 1.0
    assert h[frozenset({"x"})] == 3.0
    assert h[frozenset({"y"})] == 1.0
    assert h[frozenset({"x", "y"})] == 1.0
    # φ_x = h({x}) + h({x,y})/2 = 3 + 0.5 = 3.5 ; φ_y = 1 + 0.5 = 1.5
    phi = shapley_values(h, tog)
    assert phi["x"] == 3.5
    assert phi["y"] == 1.5
    # efficiency: 3.5 + 1.5 = 5 = Y(N)-Y(∅) = 6-1
    assert math.isclose(phi["x"] + phi["y"], 5.0)


def test_walsh_doe_hand_values():
    Y = _Y2(1.0, 4.0, 2.0, 6.0)
    tog = ("x", "y")
    g = walsh_coefficients(Y, tog)
    # γ_x = ( -Y∅ + Y{x} - Y{y} + Y{xy} ) / 4 = (-1+4-2+6)/4 = 7/4
    assert math.isclose(g[frozenset({"x"})], 7.0 / 4.0)
    E = doe_effects(g)
    # first-order DOE main effect = mean(Y|x ON) - mean(Y|x OFF)
    #   = ((Y{x}+Y{xy})/2) - ((Y∅+Y{y})/2) = (10/2) - (3/2) = 3.5 = 2·γ_x
    assert math.isclose(E[frozenset({"x"})], 3.5)
    assert math.isclose(E[frozenset({"x"})], 2.0 * g[frozenset({"x"})])


def test_corner_marginals_and_bracket():
    Y = _Y2(1.0, 4.0, 2.0, 6.0)
    add_in, leave_out, bracket = corner_marginals(Y, ("x", "y"))
    assert add_in["x"] == 3.0            # Y{x}-Y∅
    assert leave_out["x"] == 4.0          # Y{xy}-Y{y} = 6-2
    assert bracket["x"] == -1.0           # 3 - 4


# --------------------------------------------------------------------------
# ADR bias_class §9.1 — the saturated transforms are invariant to which factor
# is "pulled out" (toggle ordering). This is the isomorphism the ADR §3 relies on:
# restructuring the 2^5 into 2^4×2 is a pure relabelling of coefficients already
# computed, so the k=4 "drop meas_err" restructure is unnecessary.
# --------------------------------------------------------------------------

def _all_subsets(toggles):
    out = []
    for r in range(len(toggles) + 1):
        out.extend(frozenset(c) for c in combinations(toggles, r))
    return out


def test_walsh_doe_harsanyi_invariant_to_toggle_ordering():
    from agents.auditor.schemas.toggle import TOGGLE_IDS

    toggles = list(TOGGLE_IDS)
    # A fixed, arbitrary metric over all 32 cells (deterministic, no RNG).
    Y = {s: (len(s) * 1.7 - 0.3 * sum(hash(t) % 7 for t in s)) for s in _all_subsets(toggles)}

    base_w = walsh_coefficients(Y, toggles)
    base_d = doe_effects(base_w)
    base_h = harsanyi_dividends(Y, toggles)

    # Every permutation "pulls out" a different factor first; coefficients are
    # keyed by frozenset coalition, so they must be identical per-coalition.
    orderings = [list(reversed(toggles)), ["meas_err"] + toggles[1:][::-1]]
    orderings += [list(p) for p in list(permutations(toggles))[::37][:6]]  # a spread of perms
    for order in orderings:
        w = walsh_coefficients(Y, order)
        d = doe_effects(w)
        h = harsanyi_dividends(Y, order)
        for T in _all_subsets(toggles):
            assert math.isclose(w[T], base_w[T], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(d[T], base_d[T], rel_tol=0, abs_tol=1e-12)
            assert math.isclose(h[T], base_h[T], rel_tol=0, abs_tol=1e-12)


def test_meas_err_doe_equals_the_two_block_contrast():
    # The "E main effect" IS the average contrast between the E=ON and E=OFF halves
    # of the lattice — the number the 2^5 basis already reports (ADR §3).
    from agents.auditor.schemas.toggle import TOGGLE_IDS

    toggles = list(TOGGLE_IDS)
    Y = {s: (2.0 * len(s) + 0.11 * sum(hash(t) % 5 for t in s)) for s in _all_subsets(toggles)}
    doe = doe_effects(walsh_coefficients(Y, toggles))

    on = [s for s in _all_subsets(toggles) if "meas_err" in s]
    off = [s for s in _all_subsets(toggles) if "meas_err" not in s]
    assert len(on) == 16 and len(off) == 16
    block_contrast = sum(Y[s] for s in on) / 16.0 - sum(Y[s] for s in off) / 16.0
    assert math.isclose(doe[frozenset({"meas_err"})], block_contrast, abs_tol=1e-12)


# --------------------------------------------------------------------------
# Percentage-denominator guard (§5.3)
# --------------------------------------------------------------------------

def test_shares_reported_when_denominator_large():
    Y = _Y2(0.0, 3.0, 1.0, 5.0)  # gap = 5
    res = shapley_result(Y, ("x", "y"), percentage_denominator_min=0.5)
    assert res.denominator_ok
    assert res.shares["x"] is not None
    # shares sum to 1 exactly (efficiency / gap)
    assert math.isclose(res.shares["x"] + res.shares["y"], 1.0)


def test_shares_withheld_when_denominator_below_threshold():
    # Construct a near-zero endpoint gap: Y(N) ≈ Y(∅).
    Y = _Y2(1.0, 4.0, -2.0, 1.0)  # gap = Y{xy}-Y∅ = 0
    res = shapley_result(Y, ("x", "y"), percentage_denominator_min=0.05)
    assert not res.denominator_ok
    assert res.shares["x"] is None and res.shares["y"] is None
    # metric-unit contributions remain well defined regardless
    assert res.values["x"] is not None


def test_negative_and_over_100_percent_shares_allowed_when_denominator_ok():
    # Offsetting contributions: φ can exceed the gap in magnitude with opposite signs.
    Y = _Y2(0.0, 10.0, -9.0, 1.0)  # gap = 1; large offsetting main effects
    res = shapley_result(Y, ("x", "y"), percentage_denominator_min=0.5)
    assert res.denominator_ok
    assert math.isclose(res.shares["x"] + res.shares["y"], 1.0)
    # at least one share is outside [0,1] (a genuine offsetting contribution)
    assert res.shares["x"] > 1.0 or res.shares["y"] < 0.0


def test_efficiency_violation_raises(monkeypatch):
    import agents.auditor.checks.shapley as sh

    # Corrupt the dividend map so Σφ no longer equals the gap.
    real = sh.harsanyi_dividends

    def broken(Y, toggles):
        h = dict(real(Y, toggles))
        for T in h:
            if len(T) == 1:
                h[T] += 1.0  # break additivity
        return h

    monkeypatch.setattr(sh, "harsanyi_dividends", broken)
    with pytest.raises(EfficiencyViolation):
        shapley_result(_Y2(0.0, 3.0, 1.0, 5.0), ("x", "y"),
                       percentage_denominator_min=0.5)
