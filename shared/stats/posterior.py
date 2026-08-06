"""Scalar normal–normal posterior for holdout extension alphas (P3 / WS-A).

The presentation layer pre-registered as amendment SC-SCI-11: for each
extension evaluated on the holdout, report P(alpha > 0 | data) under 2–3
pre-stated zero-centred normal priors, BESIDE the frequentist interval and
paired test. BH-FDR remains the sole decision rule — the posterior is a
re-expression of the same data beside the frequentist interval, never
independent corroboration (the Auditor's shared-covariance honesty rule,
carried over). Posteriors are reported numerically; no qualitative adjectives
are attached.

This is the d=1 special case of the Auditor's pre-registered Bayesian normal
approximation (`agents/auditor/checks/bayes.py`):

  Likelihood:  alpha_hat | alpha ~ N(alpha, se^2)
  Prior:       alpha ~ N(0, sigma^2)
  Posterior:   alpha | alpha_hat ~ N(mu_post, sd_post^2)   (conjugate)

with mu_post = alpha_hat * (se^-2) / (se^-2 + sigma^-2) and
sd_post^2 = 1 / (se^-2 + sigma^-2). A unit test pins agreement with the
Auditor's `_posterior` at d=1, so the reuse claim is tested, not asserted.

The SE plugs in from the existing G3 output: `regress_on_benchmark` returns
`alpha` and the NW-HAC `alpha_t`, so se = alpha / alpha_t (see
`alpha_se_from_t`); when the Auditor's `InferenceResult` routing supplies a CI
instead, the caller derives se from it under the same normal approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from agents.auditor.checks.stats import normal_cdf, normal_ppf


@dataclass(frozen=True)
class PriorPosterior:
    """One prior's posterior re-expression of (point, se). Numeric only."""

    label: str
    prior_sigma: float
    post_mean: float
    post_sd: float
    ci_low: float
    ci_high: float
    p_positive: float          # P(alpha > 0 | data) under this prior

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "prior_sigma": self.prior_sigma,
            "post_mean": self.post_mean,
            "post_sd": self.post_sd,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_positive": self.p_positive,
        }


@dataclass(frozen=True)
class PosteriorSummary:
    """The full prior-sensitivity row set for one extension's holdout alpha."""

    point: float
    se: float
    posteriors: tuple[PriorPosterior, ...]

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "se": self.se,
            "posteriors": [p.to_dict() for p in self.posteriors],
        }


def alpha_se_from_t(alpha: float, alpha_t: float) -> float:
    """The standard error implied by a point estimate and its t-statistic
    (se = alpha / t) — the seam between `regress_on_benchmark`'s NW-HAC output
    and `posterior_summary`. Fail-loud on a zero/non-finite t: an SE cannot be
    recovered, and silently substituting one would launder a made-up number."""
    if not (math.isfinite(alpha) and math.isfinite(alpha_t)) or alpha_t == 0.0:
        raise ValueError(
            f"alpha_se_from_t requires finite alpha and non-zero finite t; "
            f"got alpha={alpha!r}, alpha_t={alpha_t!r}"
        )
    return abs(alpha / alpha_t)


def posterior_summary(
    point: float,
    se: float,
    priors: Mapping[str, float],
    *,
    interval_mass: float = 0.95,
) -> PosteriorSummary:
    """Closed-form normal–normal posterior of a scalar alpha under each prior.

    ``priors`` maps label -> prior sigma (zero-centred normal on MONTHLY
    holdout alpha, in return units). Pure function; fail-loud on non-finite or
    non-positive inputs (a silent default here would be a researcher degree of
    freedom, the exact thing the amendment pre-commits away)."""
    if not math.isfinite(point):
        raise ValueError(f"point must be finite; got {point!r}")
    if not (math.isfinite(se) and se > 0.0):
        raise ValueError(f"se must be finite and > 0; got {se!r}")
    if not priors:
        raise ValueError("priors must be non-empty")
    if not (0.0 < interval_mass < 1.0):
        raise ValueError(f"interval_mass must be in (0, 1); got {interval_mass!r}")

    z = normal_ppf(0.5 + interval_mass / 2.0)
    rows: list[PriorPosterior] = []
    for label, sigma in priors.items():
        if not (isinstance(sigma, (int, float)) and math.isfinite(sigma) and sigma > 0.0):
            raise ValueError(f"prior {label!r}: sigma must be finite and > 0; got {sigma!r}")
        precision = se**-2 + float(sigma) ** -2
        post_var = 1.0 / precision
        post_mean = (point / se**2) * post_var
        post_sd = math.sqrt(post_var)
        rows.append(PriorPosterior(
            label=label,
            prior_sigma=float(sigma),
            post_mean=post_mean,
            post_sd=post_sd,
            ci_low=post_mean - z * post_sd,
            ci_high=post_mean + z * post_sd,
            p_positive=1.0 - normal_cdf(-post_mean / post_sd),
        ))
    return PosteriorSummary(point=point, se=se, posteriors=tuple(rows))
