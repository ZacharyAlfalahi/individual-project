"""
Signal-recovery validation for the characteristic-sort engine.

Unlike the deterministic correctness tests in `test_characteristic_sort.py`
and the algebraic invariants in `test_characteristic_sort_invariants.py`,
these tests are STATISTICAL: they generate synthetic panels from a known
data-generating process (DGP) and verify the engine recovers the planted
truth within sampling tolerance.

This is the project-standard validation pattern (also used
for the Auditor's bias-injection tests). It catches statistical-bias bugs
the algebraic invariants would miss -- e.g., a subtle off-by-(T-1)/T in
the variance estimator does not violate scale equivariance but would
bias the recovered Sharpe.

All tests use fixed RNG seeds so they are reproducible across runs.
Tolerances are at +-3 standard errors so the per-test false-failure rate
is approximately 0.27% under correct behaviour.
"""

import math

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import (
    regress_on_benchmark,
    run_characteristic_sort,
)


# ---------------------------------------------------------------------------
# DGP helper: bond quality drives next-month return
# ---------------------------------------------------------------------------

def _build_quality_dgp(
    n_bonds: int,
    n_months: int,
    alpha: float,
    sigma: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build a panel where:
      quality_i ~ U(-1, 1)            (drawn once per bond)
      score_i,t  = quality_i           (constant across time per bond)
      ret_i,t    = alpha * quality_i + N(0, sigma^2)  (for t >= 1)

    The engine sees ret_i,(t+1) as `next_ret` at formation month t, so the
    realised long-short spread at formation t equals:
       alpha * (mean(top-quintile quality) - mean(bot-quintile quality))
                 + mean(20 noise) - mean(20 noise)
    """
    rng = np.random.default_rng(seed)
    bonds = [f"B{i:03d}" for i in range(n_bonds)]
    qualities = rng.uniform(-1.0, 1.0, size=n_bonds)
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    rows = []
    for j, d in enumerate(dates):
        # At month j=0 the engine never reads `ret` (it would be the
        # backward-looking return ending at month 0; no preceding formation
        # month exists). Set to 0 for cleanliness.
        if j == 0:
            month_rets = np.zeros(n_bonds)
        else:
            month_rets = alpha * qualities + rng.normal(scale=sigma, size=n_bonds)
        for i, bid in enumerate(bonds):
            rows.append(
                {
                    "cusip": bid,
                    "date": d,
                    "ret": float(month_rets[i]),
                    "size": 100.0,
                    "score": float(qualities[i]),
                }
            )
    return pd.DataFrame(rows), qualities


# ---------------------------------------------------------------------------
# Test 1 — Recovery of a planted positive alpha
# ---------------------------------------------------------------------------

def test_recovery_of_planted_long_short_alpha() -> None:
    """With a strong planted alpha (alpha=0.01, sigma=0.04 over 120 months
    and 100 bonds), the engine's recovered mean strategy return must lie
    within +-3 SE of the realised theoretical spread, AND the t-stat must
    indicate a clearly significant signal.

    Realised theoretical spread:
       alpha * (mean(top-20 quality) - mean(bot-20 quality))
    where the qualities are the actual draws (not the population means),
    so the assertion is exact-in-expectation given the random panel.
    """
    n_bonds = 100
    n_months = 121  # 120 formation months
    alpha = 0.01
    sigma = 0.04

    panel, qualities = _build_quality_dgp(
        n_bonds, n_months, alpha, sigma, seed=2026
    )

    # Realised theoretical spread under the engine's quintile split.
    sorted_q = np.sort(qualities)
    bot_quintile_mean_q = sorted_q[:20].mean()
    top_quintile_mean_q = sorted_q[-20:].mean()
    theoretical_spread = alpha * (top_quintile_mean_q - bot_quintile_mean_q)

    result = run_characteristic_sort(
        panel,
        {
            "score": "score",
            "groups": 5,
            "weighting": "equal",
            "min_bonds": 100,
            "nw_lags": 0,
        },
    )

    mean_recovered = result["summary"]["average"]
    t_stat = result["summary"]["t_stat"]
    n_months_run = result["summary"]["n_months"]

    assert n_months_run == 120

    # Sampling SE: per-month spread variance is approximately 2 * sigma^2 / 20
    # = sigma^2 / 10. SE of the time-mean is sqrt(sigma^2 / 10 / T).
    expected_se = math.sqrt(sigma**2 / 10.0 / n_months_run)

    assert abs(mean_recovered - theoretical_spread) < 3.0 * expected_se, (
        f"Recovered mean {mean_recovered:.6f} is more than 3*SE from "
        f"theoretical {theoretical_spread:.6f} "
        f"(3*SE = {3 * expected_se:.6f})"
    )
    # Strong signal -> t-stat clearly above 3.
    assert t_stat > 3.0, (
        f"Engine recovered t = {t_stat:.3f}; expected >> 3 for the planted "
        f"alpha = {alpha} over {n_months_run} months."
    )


# ---------------------------------------------------------------------------
# Test 2 — Null recovery (no relationship between score and next-ret)
# ---------------------------------------------------------------------------

def test_recovery_under_null_no_signal() -> None:
    """When score and next-ret are independent (no signal), the engine must
    NOT spuriously detect one. Recovered mean lies within +-3 SE of zero
    and |t-stat| < 3."""
    n_bonds = 100
    n_months = 121
    sigma = 0.04
    rng = np.random.default_rng(seed=2027)

    bonds = [f"B{i:03d}" for i in range(n_bonds)]
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")

    rows = []
    for j, d in enumerate(dates):
        # Independent score every month, independent return -- no relationship.
        scores = rng.normal(size=n_bonds)
        if j == 0:
            month_rets = np.zeros(n_bonds)
        else:
            month_rets = rng.normal(scale=sigma, size=n_bonds)
        for i, bid in enumerate(bonds):
            rows.append(
                {
                    "cusip": bid,
                    "date": d,
                    "ret": float(month_rets[i]),
                    "size": 100.0,
                    "score": float(scores[i]),
                }
            )
    panel = pd.DataFrame(rows)

    result = run_characteristic_sort(
        panel,
        {
            "score": "score",
            "groups": 5,
            "weighting": "equal",
            "min_bonds": 100,
            "nw_lags": 0,
        },
    )

    mean_recovered = result["summary"]["average"]
    t_stat = result["summary"]["t_stat"]
    n_months_run = result["summary"]["n_months"]
    bumpiness = result["summary"]["bumpiness"]

    # Under H0, |t| < 3 with probability > 99.7%. With the fixed seed this
    # is deterministic; if a future engine bug biases the mean, |t| jumps.
    assert abs(t_stat) < 3.0, (
        f"Null DGP produced |t| = {abs(t_stat):.3f} (>= 3); engine may "
        "be spuriously detecting a signal."
    )

    # Recovered mean within 3 SE of zero.
    se_realised = bumpiness / math.sqrt(n_months_run)
    assert abs(mean_recovered) < 3.0 * se_realised, (
        f"Recovered mean {mean_recovered:.6f} is more than 3*SE from 0 "
        f"under H0 (3*SE = {3 * se_realised:.6f})."
    )


# ---------------------------------------------------------------------------
# Test 3 — Monotonic quintile structure
# ---------------------------------------------------------------------------

def test_recovery_monotonic_quintile_means() -> None:
    """Under a DGP where score predicts next-ret monotonically, the per-
    quintile time-averaged next-returns must be monotonically increasing
    across the 5 quintiles. Tests the grouping logic at every quintile,
    not just the extremes."""
    n_bonds = 100
    n_months = 61  # 60 formation months
    alpha = 0.02
    sigma = 0.04

    panel, _ = _build_quality_dgp(
        n_bonds, n_months, alpha, sigma, seed=2028
    )

    quintile_means: list[float] = []
    for short_k in range(5):
        # Set long_group to any value != short_k so the engine accepts the
        # rulebook; we only use the `short_ret` column (= mean of quintile
        # short_k's next_ret) from the output.
        long_k = 4 if short_k != 4 else 0
        result = run_characteristic_sort(
            panel,
            {
                "score": "score",
                "groups": 5,
                "weighting": "equal",
                "min_bonds": 100,
                "nw_lags": 0,
                "long_group": long_k,
                "short_group": short_k,
            },
        )
        quintile_means.append(float(result["monthly_returns"]["short_ret"].mean()))

    for i in range(4):
        assert quintile_means[i] < quintile_means[i + 1], (
            f"Quintile means not monotone increasing: "
            f"q{i} = {quintile_means[i]:.6f} not < "
            f"q{i + 1} = {quintile_means[i + 1]:.6f}; "
            f"full sequence = {quintile_means}"
        )


# ---------------------------------------------------------------------------
# Test 4 — Benchmark regression recovers planted alpha and beta
# ---------------------------------------------------------------------------

def test_recovery_benchmark_regression() -> None:
    """regress_on_benchmark recovers planted alpha and beta within +-3 SE
    on a synthetic single-factor model. Validates the regression layer
    the way Tests 1-3 validate the sorting layer."""
    T = 240
    sigma_eps = 0.02
    true_alpha = 0.002
    true_beta = 0.5
    rng = np.random.default_rng(seed=2029)

    dates = pd.date_range("2000-01-31", periods=T, freq="ME")
    factor_market = rng.normal(scale=0.04, size=T)
    eps = rng.normal(scale=sigma_eps, size=T)
    y_vals = true_alpha + true_beta * factor_market + eps

    y = pd.Series(y_vals, index=dates)
    factors = pd.DataFrame({"date": dates, "MKT": factor_market})

    out = regress_on_benchmark(y, factors, nw_lags=0)

    assert out["n_obs"] == T

    var_factor = float(np.var(factor_market, ddof=1))
    mean_factor = float(np.mean(factor_market))
    se_beta = sigma_eps / math.sqrt(T * var_factor)
    se_alpha = (sigma_eps / math.sqrt(T)) * math.sqrt(
        1.0 + mean_factor**2 / var_factor
    )

    assert abs(out["alpha"] - true_alpha) < 3.0 * se_alpha, (
        f"Recovered alpha {out['alpha']:.6f} is more than 3*SE from "
        f"{true_alpha} (3*SE = {3 * se_alpha:.6f})."
    )
    assert abs(out["betas"]["MKT"] - true_beta) < 3.0 * se_beta, (
        f"Recovered beta {out['betas']['MKT']:.6f} is more than 3*SE from "
        f"{true_beta} (3*SE = {3 * se_beta:.6f})."
    )
