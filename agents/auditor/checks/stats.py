"""
stats.py — the few statistical primitives the inference layer needs, in pure
numpy/math (the project carries no scipy/statsmodels; §9.1 keeps the stats
contract small and shared).

  * normal_cdf / normal_ppf — standard-normal CDF (via erf) and its inverse
    (Acklam's rational approximation, ~1e-9 absolute error).
  * two_sided_p — the large-sample two-sided p-value for a z/t statistic.
  * betainc_regularised / student_t_sf / student_t_cdf — the exact t tails, for the callers
    whose df is too small for the normal reference (the TOST equivalence family).
  * ks_uniform — the one-sample Kolmogorov-Smirnov statistic against U(0,1) and its
    asymptotic p-value, for descriptive uniformity reporting.

Large-sample normal approximation is used for HAC p-values (the estimand is a
mean; the monthly contrast series is long enough that the t reference is ~normal).
Credible/confidence intervals elsewhere come from the bootstrap percentiles, which
need no closed-form quantile.
"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    """Standard-normal CDF Φ(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam (2003) inverse-normal-CDF rational approximation coefficients.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)


def normal_ppf(p: float) -> float:
    """Inverse standard-normal CDF Φ⁻¹(p) for 0 < p < 1 (Acklam)."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"normal_ppf requires 0 < p < 1; got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def two_sided_p(z: float) -> float:
    """Large-sample two-sided p-value for a z/t statistic: 2·(1 - Φ(|z|))."""
    if math.isnan(z):
        return float("nan")
    return 2.0 * (1.0 - normal_cdf(abs(z)))


# --- Student-t -------------------------------------------------------------------------------
#
# The t distribution is needed where the large-sample normal reference is NOT good enough: the
# TOST equivalence family (shared/evaluation/equivalence.py) tests a development window whose df
# is ~55-60, and at that df the normal tail understates a one-sided p by enough to flip a
# marginal equivalence decision. Hand-rolled here, beside normal_cdf, for the reason given at the
# top of this module: the project carries no scipy/statsmodels.
#
# Both functions route through the regularised incomplete beta I_x(a, b), evaluated by the
# Lentz continued fraction (Numerical Recipes §6.4). Agreement with scipy.stats.t is ~1e-15
# absolute over df in [1, 500] and |t| in [0, 40]; the tripwire lives in
# tests/unit/test_auditor_inference.py.

_BETACF_MAX_ITER = 300
_BETACF_EPS = 3.0e-16
_BETACF_TINY = 1.0e-300


def _betacf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion for the incomplete beta (Lentz's modified algorithm)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _BETACF_TINY:
        d = _BETACF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _BETACF_MAX_ITER + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _BETACF_TINY:
            d = _BETACF_TINY
        c = 1.0 + aa / c
        if abs(c) < _BETACF_TINY:
            c = _BETACF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _BETACF_EPS:
            return h
    raise ArithmeticError(
        f"incomplete beta continued fraction did not converge in {_BETACF_MAX_ITER} iterations "
        f"(a={a}, b={b}, x={x}) — refusing to return a half-converged p-value")


def betainc_regularised(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if not (a > 0.0 and b > 0.0):
        raise ValueError(f"betainc_regularised requires a > 0 and b > 0; got a={a}, b={b}")
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"betainc_regularised requires 0 <= x <= 1; got {x}")
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    # The fraction converges fast only for x < (a+1)/(a+b+2); otherwise use the symmetry
    # I_x(a, b) = 1 - I_{1-x}(b, a), which puts the argument back in the fast region.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t: float, df: int) -> float:
    """Upper-tail probability P(T > t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"student_t_sf requires df > 0; got {df}")
    if math.isnan(t):
        return float("nan")
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    # I_{df/(df+t^2)}(df/2, 1/2) is the two-tailed mass; halve it and orient by the sign of t.
    half = 0.5 * betainc_regularised(0.5 * df, 0.5, df / (df + t * t))
    return half if t > 0 else 1.0 - half


def student_t_cdf(t: float, df: int) -> float:
    """Lower-tail probability P(T < t) for Student's t with ``df`` degrees of freedom."""
    return student_t_sf(-t, df)


# --- Kolmogorov-Smirnov ----------------------------------------------------------------------


def ks_uniform(values: "list[float] | tuple[float, ...]") -> tuple[float, float]:
    """One-sample KS against U(0,1): ``(D, p)``.

    ``D`` is exact (it agrees with the exact statistic to the last bit). ``p`` is the ASYMPTOTIC
    Kolmogorov limit Q(lambda) = 2 * sum_{k>=1} (-1)^(k-1) exp(-2 k^2 lambda^2) evaluated at
    lambda = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D, the Stephens small-sample correction — not the
    exact finite-n distribution. Measured against the exact distribution over n in [5, 1000], the
    error lies in [-0.0023, +0.0227] and is almost entirely one-sided UPWARD, i.e. this p is
    conservative for a uniformity check: it errs towards failing to flag non-uniformity, never
    towards manufacturing it. The large end of that band sits in the bulk; near the conventional
    0.05 region the error is under 0.004.

    That is ample here, and deliberately so. The only caller
    (``scripts/run_ipca_randomisation_fpr.py``) reports this descriptively and never gates on it,
    and its inputs are randomisation p-values that are DISCRETE on multiples of 1/(Q+1) — so the
    continuous-uniform reference is itself approximate by construction, by more than the
    asymptotic correction costs. An exact finite-n KS distribution here would be false precision.
    """
    n = len(values)
    if n == 0:
        raise ValueError("ks_uniform requires at least one value")
    ordered = sorted(float(v) for v in values)
    if not all(0.0 <= v <= 1.0 for v in ordered):
        raise ValueError("ks_uniform compares against U(0,1); every value must lie in [0, 1]")
    d = max(max((i + 1) / n - v, v - i / n) for i, v in enumerate(ordered))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    if lam <= 0.0:
        return d, 1.0
    total, sign = 0.0, 1.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * lam * lam)
        total += sign * term
        sign = -sign
        if term < 1.0e-16:
            break
    return d, min(1.0, max(0.0, 2.0 * total))
