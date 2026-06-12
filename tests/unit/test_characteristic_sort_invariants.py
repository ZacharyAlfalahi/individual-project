"""
Invariant / property tests for the characteristic-sort engine.

Each test asserts an algebraic property the engine MUST satisfy regardless
of the specific numbers in the panel. A specific-example test catches one
bug; an invariant test catches an infinite family of bugs.

Plus two differential tests that compare the production engine against an
independent, deliberately-naive plain-Python reference implementation in
`_naive_reference.py`. Agreement to 1e-12 means two implementations with
maximally different code styles arrived at the same answer -- much stronger
evidence than either implementation alone.

The synthetic panels here are rich enough that a buggy engine has many
chances to fail per test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.quant.library.characteristic_sort import (
    run_characteristic_sort,
    summarize_returns,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _naive_reference import naive_run_characteristic_sort  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic panel builders
# ---------------------------------------------------------------------------

def _build_panel(
    n_bonds: int,
    n_months: int,
    *,
    seed: int,
    start: str = "2010-01-31",
) -> pd.DataFrame:
    """Reproducible panel with persistent per-bond score quality + monthly
    noise. The 'good' bonds (high score) earn higher expected returns so
    the engine has something to find."""
    rng = np.random.default_rng(seed)
    bonds = [f"B{i:03d}" for i in range(1, n_bonds + 1)]
    dates = pd.date_range(start, periods=n_months, freq="ME")
    quality = rng.uniform(-1.0, 1.0, size=n_bonds)
    sizes = rng.uniform(50.0, 500.0, size=n_bonds)
    rows = []
    for j, d in enumerate(dates):
        for i, bid in enumerate(bonds):
            score_noise = rng.normal(scale=0.2)
            ret_noise = rng.normal(scale=0.02)
            rows.append(
                {
                    "cusip": bid,
                    "date": d,
                    "ret": 0.005 * quality[i] + ret_noise,
                    "size": sizes[i] * (1.0 + 0.05 * rng.normal()),
                    "score": quality[i] + score_noise,
                    "control_value": rng.uniform(0.0, 1.0),
                }
            )
    return pd.DataFrame(rows)


def _run(panel, **overrides):
    base = {"score": "score", "groups": 5, "weighting": "by_size", "min_bonds": 5}
    base.update(overrides)
    return run_characteristic_sort(panel, base)


# ---------------------------------------------------------------------------
# Invariant 1 — Long-short sign flip
# ---------------------------------------------------------------------------

def test_invariant_long_short_swap_flips_sign() -> None:
    """Swapping long_group <-> short_group multiplies strategy_ret by -1
    in every month. Long and short legs themselves also swap."""
    panel = _build_panel(40, 18, seed=11)
    normal = _run(panel, long_group=4, short_group=0)
    swapped = _run(panel, long_group=0, short_group=4)
    n = normal["monthly_returns"]
    s = swapped["monthly_returns"]
    assert len(n) == len(s) > 0
    np.testing.assert_allclose(
        n["strategy_ret"].values, -s["strategy_ret"].values, atol=1e-12
    )
    np.testing.assert_allclose(
        n["long_ret"].values, s["short_ret"].values, atol=1e-12
    )
    np.testing.assert_allclose(
        n["short_ret"].values, s["long_ret"].values, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Invariant 2 — Return scale equivariance
# ---------------------------------------------------------------------------

def test_invariant_return_scale_equivariance() -> None:
    """Multiplying every `ret` by k > 0 multiplies strategy_ret by k."""
    panel = _build_panel(40, 18, seed=22)
    base = _run(panel)
    k = 3.7
    scaled_panel = panel.copy()
    scaled_panel["ret"] = scaled_panel["ret"] * k
    scaled = _run(scaled_panel)
    np.testing.assert_allclose(
        base["monthly_returns"]["strategy_ret"].values * k,
        scaled["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 3 — Return translation invariance (long-short cancellation)
# ---------------------------------------------------------------------------

def test_invariant_return_translation_invariance() -> None:
    """Adding a constant c to every `ret` leaves strategy_ret UNCHANGED
    (long-short cancellation)."""
    panel = _build_panel(40, 18, seed=33)
    base = _run(panel)
    c = 0.015
    shifted_panel = panel.copy()
    shifted_panel["ret"] = shifted_panel["ret"] + c
    shifted = _run(shifted_panel)
    np.testing.assert_allclose(
        base["monthly_returns"]["strategy_ret"].values,
        shifted["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )
    # Long and short legs DO shift by c.
    np.testing.assert_allclose(
        base["monthly_returns"]["long_ret"].values + c,
        shifted["monthly_returns"]["long_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 4 — Size scale invariance under by_size weighting
# ---------------------------------------------------------------------------

def test_invariant_size_scale_invariance_by_size() -> None:
    """Multiplying every `size` by k > 0 leaves strategy_ret UNCHANGED
    under weighting='by_size' (weights normalise inside each leg)."""
    panel = _build_panel(40, 18, seed=44)
    base = _run(panel, weighting="by_size")
    k = 1000.0
    scaled_panel = panel.copy()
    scaled_panel["size"] = scaled_panel["size"] * k
    scaled = _run(scaled_panel, weighting="by_size")
    np.testing.assert_allclose(
        base["monthly_returns"]["strategy_ret"].values,
        scaled["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 5 — Leg decomposition identity
# ---------------------------------------------------------------------------

def test_invariant_leg_decomposition_identity_single_sort() -> None:
    """In a single sort, strategy_ret == long_ret - short_ret exactly for
    every row. (In a double sort, the relationship is mean-of-stripe-
    spreads, which CAN differ from mean(long) - mean(short) when stripes
    have different counts contributing.)"""
    panel = _build_panel(40, 18, seed=55)
    res = _run(panel)
    mr = res["monthly_returns"]
    np.testing.assert_allclose(
        mr["strategy_ret"].values,
        mr["long_ret"].values - mr["short_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 6 — Idempotence
# ---------------------------------------------------------------------------

def test_invariant_idempotence() -> None:
    """Two runs on the same panel produce identical monthly_returns rows."""
    panel = _build_panel(40, 18, seed=66)
    r1 = _run(panel)
    r2 = _run(panel)
    pd.testing.assert_frame_equal(r1["monthly_returns"], r2["monthly_returns"])


# ---------------------------------------------------------------------------
# Invariant 7 — equal == by_size when sizes are constant
# ---------------------------------------------------------------------------

def test_invariant_equal_eq_by_size_when_sizes_constant() -> None:
    """When all sizes are equal, weighting='equal' and weighting='by_size'
    produce identical strategy_ret to 1e-12."""
    panel = _build_panel(40, 18, seed=77)
    panel["size"] = 100.0  # all equal
    r_eq = _run(panel, weighting="equal")
    r_bs = _run(panel, weighting="by_size")
    np.testing.assert_allclose(
        r_eq["monthly_returns"]["strategy_ret"].values,
        r_bs["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 8 — Score is rank-only (strict-monotone transform invariant)
# ---------------------------------------------------------------------------

def test_invariant_score_rank_only() -> None:
    """Applying a strict-monotone transform to the score (e.g. score**3 for
    a positive-shifted score) preserves rank order and therefore must
    preserve all group assignments and strategy returns."""
    panel = _build_panel(40, 18, seed=88)
    # Shift score to be strictly positive so cubing is strictly monotone.
    panel["score"] = panel["score"] + 10.0
    base = _run(panel)
    transformed = panel.copy()
    transformed["score"] = transformed["score"] ** 3
    out = _run(transformed)
    np.testing.assert_allclose(
        base["monthly_returns"]["strategy_ret"].values,
        out["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 9 — Adding a NaN-next_ret bond is eligibility-invariant
# ---------------------------------------------------------------------------

def test_invariant_extra_bond_with_no_next_ret_is_invisible() -> None:
    """Adding an extra bond that has rows at every month but whose `next_ret`
    is uniformly NaN -- by virtue of having NO row at t+1 -- must leave
    every strategy return unchanged."""
    panel = _build_panel(30, 12, seed=99)
    base = _run(panel)
    # Add a bond whose rows exist only at month 6 of 12 -- so no t+1 row at
    # month 7 (and no t-1 row at month 5 for signal_lag). This bond's
    # next_ret will always be NaN at every formation month, so eligibility
    # drops it.
    extra = pd.DataFrame(
        [
            {
                "cusip": "EXTRA",
                "date": pd.Timestamp("2010-06-30"),
                "ret": 999.0,
                "size": 999.0,
                "score": 999.0,
                "control_value": 0.5,
            }
        ]
    )
    augmented = pd.concat([panel, extra], ignore_index=True)
    out = _run(augmented)
    pd.testing.assert_frame_equal(base["monthly_returns"], out["monthly_returns"])


# ---------------------------------------------------------------------------
# Invariant 10 — Permutation invariance under cusip rename (no ties)
# ---------------------------------------------------------------------------

def test_invariant_permutation_invariance_under_cusip_rename() -> None:
    """A bijective rename of cusips must not change strategy_ret -- as
    long as no inter-bond ties in score exist for which the cusip order
    matters. Numerically generated scores from a continuous distribution
    have measure-zero probability of ties, so this holds in practice."""
    panel = _build_panel(40, 18, seed=101)
    base = _run(panel)
    # Reverse every cusip string: "B007" -> "700B".
    renamed = panel.copy()
    renamed["cusip"] = renamed["cusip"].str[::-1]
    out = _run(renamed)
    np.testing.assert_allclose(
        base["monthly_returns"]["strategy_ret"].values,
        out["monthly_returns"]["strategy_ret"].values,
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Invariant 11 — Sharpe / mean equivariance under return scaling
# ---------------------------------------------------------------------------

def test_invariant_sharpe_and_mean_scale_equivariance() -> None:
    """For a return series scaled by k > 0:
       mean         scales by k
       bumpiness    scales by k
       Sharpe       unchanged
       t-stat (NW0) unchanged."""
    rng = np.random.default_rng(202)
    s = pd.Series(rng.normal(scale=0.02, size=60))
    base = summarize_returns(s, nw_lags=0, months_per_year=12)
    k = 5.0
    scaled = summarize_returns(s * k, nw_lags=0, months_per_year=12)
    assert scaled["average"] == pytest.approx(base["average"] * k, abs=1e-12)
    assert scaled["bumpiness"] == pytest.approx(base["bumpiness"] * k, abs=1e-12)
    assert scaled["sharpe"] == pytest.approx(base["sharpe"], abs=1e-12)
    assert scaled["t_stat"] == pytest.approx(base["t_stat"], abs=1e-12)


# ---------------------------------------------------------------------------
# Invariant 12 — Single eligible bond per month -> all months skipped
# ---------------------------------------------------------------------------

def test_invariant_single_eligible_bond_skips_month() -> None:
    """With min_bonds=1 and exactly one eligible bond per month, every
    month is still skipped because long and short groups cannot both be
    populated from a single bond (either same group or one empty)."""
    rows = []
    for d_str in ["2010-01", "2010-02", "2010-03"]:
        rows.append(
            {
                "cusip": "A",
                "date": pd.Timestamp(d_str) + pd.offsets.MonthEnd(0),
                "ret": 0.01,
                "size": 100.0,
                "score": 1.0,
            }
        )
    panel = pd.DataFrame(rows)
    result = run_characteristic_sort(
        panel,
        {"score": "score", "groups": 5, "weighting": "equal", "min_bonds": 1},
    )
    assert len(result["monthly_returns"]) == 0


# ---------------------------------------------------------------------------
# Differential test 1 -- single sort, production engine == naive reference
# ---------------------------------------------------------------------------

def test_differential_single_sort_against_naive_reference() -> None:
    """Production engine and plain-Python reference agree on a non-trivial
    synthetic single-sort panel to 1e-12. Two independent implementations
    that match give much stronger evidence than either alone."""
    panel = _build_panel(30, 14, seed=303)
    rulebook = {
        "score": "score",
        "groups": 5,
        "weighting": "by_size",
        "min_bonds": 5,
    }
    prod = run_characteristic_sort(panel, rulebook)
    naive = naive_run_characteristic_sort(panel, rulebook)

    p_mr = prod["monthly_returns"]
    n_rows = naive["monthly_returns"]
    assert len(p_mr) == len(n_rows) > 0
    for i, n_row in enumerate(n_rows):
        assert p_mr["date"].iloc[i] == n_row["date"], f"date mismatch at row {i}"
        assert p_mr["strategy_ret"].iloc[i] == pytest.approx(
            n_row["strategy_ret"], abs=1e-12
        ), f"strategy_ret mismatch at row {i}"
        assert p_mr["long_ret"].iloc[i] == pytest.approx(
            n_row["long_ret"], abs=1e-12
        ), f"long_ret mismatch at row {i}"
        assert p_mr["short_ret"].iloc[i] == pytest.approx(
            n_row["short_ret"], abs=1e-12
        ), f"short_ret mismatch at row {i}"
        assert p_mr["n_bonds"].iloc[i] == n_row["n_bonds"], (
            f"n_bonds mismatch at row {i}"
        )


# ---------------------------------------------------------------------------
# Differential test 2 -- double sort, production engine == naive reference
# ---------------------------------------------------------------------------

def test_differential_double_sort_against_naive_reference() -> None:
    """Same as differential test 1 but with an independent control variable
    creating a 5x5 double sort. Catches mismatches in the per-stripe
    averaging logic."""
    panel = _build_panel(40, 14, seed=404)
    rulebook = {
        "score": "score",
        "control": "control_value",
        "groups": 5,
        "control_groups": 5,
        "weighting": "by_size",
        "min_bonds": 5,
    }
    prod = run_characteristic_sort(panel, rulebook)
    naive = naive_run_characteristic_sort(panel, rulebook)

    p_mr = prod["monthly_returns"]
    n_rows = naive["monthly_returns"]
    assert len(p_mr) == len(n_rows) > 0
    for i, n_row in enumerate(n_rows):
        assert p_mr["date"].iloc[i] == n_row["date"]
        assert p_mr["strategy_ret"].iloc[i] == pytest.approx(
            n_row["strategy_ret"], abs=1e-12
        )
        assert p_mr["long_ret"].iloc[i] == pytest.approx(
            n_row["long_ret"], abs=1e-12
        )
        assert p_mr["short_ret"].iloc[i] == pytest.approx(
            n_row["short_ret"], abs=1e-12
        )
        assert p_mr["n_bonds"].iloc[i] == n_row["n_bonds"]
