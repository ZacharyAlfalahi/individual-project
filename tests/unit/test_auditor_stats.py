"""The hand-rolled distribution functions in agents/auditor/checks/stats.py.

These exist because the project carries no scipy/statsmodels, so nothing else in the repository
cross-checks them — the reference values here ARE the check. The t assertions are keyed to
published critical values (a t-table, not a library call), so the test stays independently
verifiable by anyone without the package the module deliberately avoids.
"""
from __future__ import annotations

import math

import pytest

from agents.auditor.checks.stats import (
    betainc_regularised,
    ks_uniform,
    normal_cdf,
    student_t_cdf,
    student_t_sf,
)

# (df, critical value, upper-tail mass) — standard two-sided 5% / one-sided 5% t-table entries.
T_TABLE = [
    (1, 12.706204736432095, 0.025),
    (2, 4.302652729911275, 0.025),
    (5, 2.0150483726691575, 0.05),
    (29, 2.045229642132703, 0.025),
    (55, 1.673033964797687, 0.05),
    (59, 2.000995378274888, 0.025),   # the TOST development window (60 months, BBW-4)
]


@pytest.mark.parametrize("df, crit, tail", T_TABLE)
def test_student_t_sf_reproduces_the_published_critical_values(df, crit, tail):
    assert student_t_sf(crit, df) == pytest.approx(tail, abs=1e-10)


@pytest.mark.parametrize("df, crit, tail", T_TABLE)
def test_student_t_is_symmetric_and_cdf_is_the_complement(df, crit, tail):
    assert student_t_sf(-crit, df) == pytest.approx(1.0 - tail, abs=1e-10)
    assert student_t_cdf(crit, df) == pytest.approx(1.0 - tail, abs=1e-10)
    assert student_t_cdf(crit, df) + student_t_sf(crit, df) == pytest.approx(1.0, abs=1e-14)


def test_student_t_sf_is_exact_at_the_centre_and_the_infinities():
    for df in (1, 5, 59, 500):
        assert student_t_sf(0.0, df) == pytest.approx(0.5, abs=1e-14)
        assert student_t_sf(math.inf, df) == 0.0
        assert student_t_sf(-math.inf, df) == 1.0
    assert math.isnan(student_t_sf(float("nan"), 10))


def test_student_t_holds_precision_in_the_deep_tail():
    # Where a TOST p-value actually lands once equivalence is comfortable. Relative, not
    # absolute, accuracy is what matters at this magnitude.
    assert student_t_sf(8.0, 59) == pytest.approx(2.7358991088913963e-11, rel=1e-9)
    assert student_t_sf(3.0, 120) == pytest.approx(0.0016419508601170805, rel=1e-9)


def test_student_t_approaches_the_normal_as_df_grows():
    # The reason two_sided_p's large-sample normal reference is sound for long series — and the
    # reason it is NOT sound at df ~ 59, which is why this module carries the exact t at all.
    assert student_t_sf(1.96, 1_000_000) == pytest.approx(1.0 - normal_cdf(1.96), abs=1e-6)
    assert abs(student_t_sf(1.96, 59) - (1.0 - normal_cdf(1.96))) > 1e-3


@pytest.mark.parametrize("df", [0, -1])
def test_student_t_refuses_non_positive_df(df):
    with pytest.raises(ValueError, match="df > 0"):
        student_t_sf(1.0, df)


def test_betainc_regularised_matches_its_closed_forms():
    # I_x(1, 1) = x, and I_x(a, b) = 1 - I_{1-x}(b, a) across the branch boundary.
    for x in (0.0, 0.1, 0.5, 0.9, 1.0):
        assert betainc_regularised(1.0, 1.0, x) == pytest.approx(x, abs=1e-14)
    for a, b, x in ((0.5, 3.0, 0.2), (3.0, 0.5, 0.8), (29.5, 0.5, 0.97), (0.5, 29.5, 0.03)):
        assert betainc_regularised(a, b, x) == pytest.approx(
            1.0 - betainc_regularised(b, a, 1.0 - x), abs=1e-13)


@pytest.mark.parametrize("a, b, x", [(0.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, 1.5),
                                     (1.0, 1.0, -0.1)])
def test_betainc_regularised_refuses_out_of_domain_arguments(a, b, x):
    with pytest.raises(ValueError):
        betainc_regularised(a, b, x)


def test_ks_uniform_statistic_is_the_exact_supremum_gap():
    # Five points at the decile midpoints of a perfect uniform: D = 1/n - (largest step) = 0.1.
    assert ks_uniform([0.1, 0.3, 0.5, 0.7, 0.9])[0] == pytest.approx(0.1, abs=1e-15)
    # Everything crammed into [0, 0.2]: the empirical CDF reaches 1 by 0.2, so D = 0.8.
    assert ks_uniform([0.0, 0.05, 0.1, 0.15, 0.2])[0] == pytest.approx(0.8, abs=1e-15)


def test_ks_uniform_separates_uniform_from_clustered():
    assert ks_uniform([0.1, 0.3, 0.5, 0.7, 0.9])[1] > 0.05
    assert ks_uniform([0.01, 0.01, 0.02, 0.01, 0.02])[1] < 0.05


def test_ks_uniform_p_is_a_probability_and_falls_as_the_gap_widens():
    previous = 1.1
    for shrink in (1.0, 0.8, 0.5, 0.25, 0.1):
        _, p = ks_uniform([shrink * v for v in (0.1, 0.3, 0.5, 0.7, 0.9)])
        assert 0.0 <= p <= 1.0
        assert p < previous
        previous = p


@pytest.mark.parametrize("values", [[], [0.5, 1.5], [-0.01, 0.5]])
def test_ks_uniform_refuses_an_empty_sample_or_values_off_the_unit_interval(values):
    with pytest.raises(ValueError):
        ks_uniform(values)
