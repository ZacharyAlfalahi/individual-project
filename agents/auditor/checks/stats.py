"""
stats.py — the few statistical primitives the inference layer needs, in pure
numpy/math (the project carries no scipy/statsmodels; §9.1 keeps the stats
contract small and shared).

  * normal_cdf / normal_ppf — standard-normal CDF (via erf) and its inverse
    (Acklam's rational approximation, ~1e-9 absolute error).
  * two_sided_p — the large-sample two-sided p-value for a z/t statistic.

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
