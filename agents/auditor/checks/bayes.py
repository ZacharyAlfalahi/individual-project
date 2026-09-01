"""
bayes.py — the single-strategy Bayesian normal approximation (§7.3).

A Bayesian normal approximation using the bootstrap-estimated sampling covariance
— described as exactly that (§7.3.4), NOT a generative model at the monthly-return
level.

  Likelihood:  θ̂ | θ ~ N(θ, V̂_boot)
  Prior:       θ ~ N(0, prior_scale² I)   (generic, weakly informative — §7.4;
                                            DRR-informed prior is sensitivity only)
  Posterior:   θ | θ̂ ~ N(μ_post, Σ_post)  (conjugate, closed form)

θ is the NON-REDUNDANT coordinate vector (§7.3.1): the five main DOE effects plus
the three confirmatory interactions — NEVER a stack of Harsanyi + Walsh + DOE +
Shapley (that is singular). On a reduced lattice θ has variable dimension via the
selection A_s (`confirmatory.confirmatory_coordinates`). Marginals and Shapley are
derived from posterior draws, not entered as separate likelihood coordinates.

V̂_boot handling is pre-registered, not improvised (§7.3.3): symmetrise → clip
eigenvalues below ε → RECORD the regularisation → report the diagonal-V̂
sensitivity. Recording it turns a silent researcher degree of freedom into a
disclosed modelling choice — the standard the project holds others to.

Outputs per coordinate: posterior mean, SD, 95% credible interval, P(θ<0),
P(θ>0), and P(|θ| > ϑ) — the material-effect probability that feeds prevalence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .bootstrap import BootstrapResult
from .stats import normal_cdf, normal_ppf


@dataclass(frozen=True)
class CovarianceRegularisation:
    """The applied V̂_boot regularisation, recorded for the report (§7.3.3)."""

    n_eigenvalues_clipped: int
    epsilon: float
    min_eigenvalue_before: float

    def to_dict(self) -> dict:
        return {
            "n_eigenvalues_clipped": self.n_eigenvalues_clipped,
            "epsilon": self.epsilon,
            "min_eigenvalue_before": self.min_eigenvalue_before,
        }


@dataclass(frozen=True)
class CoordinatePosterior:
    coordinate: frozenset
    mean: float
    sd: float
    ci_low: float
    ci_high: float
    p_negative: float
    p_positive: float
    p_material: float          # P(|θ| > ϑ) — feeds prevalence (§8.2.3)
    # ((ϑ, P(|θ|>ϑ)), ...) neighbouring-threshold sensitivity (§8.2.3/§9); empty when
    # no grid was requested. Always contains the headline ϑ when populated.
    p_material_sweep: tuple = ()

    def to_dict(self) -> dict:
        d = {
            "mean": self.mean, "sd": self.sd,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_negative": self.p_negative, "p_positive": self.p_positive,
            "p_material": self.p_material,
        }
        if self.p_material_sweep:
            d["p_material_sweep"] = [[v, p] for v, p in self.p_material_sweep]
        return d


@dataclass(frozen=True, eq=False)
class BayesianResult:
    coordinates: tuple[frozenset, ...]
    posteriors: dict            # frozenset -> CoordinatePosterior
    regularisation: CovarianceRegularisation
    prior_scale: float
    vartheta: float
    diagonal_sensitivity_max_mean_shift: float
    conditioning_signature: tuple = ()

    def to_dict(self) -> dict:
        from ..schemas.decomposition import subset_label

        return {
            "prior_scale": self.prior_scale,
            "vartheta": self.vartheta,
            "regularisation": self.regularisation.to_dict(),
            "diagonal_sensitivity_max_mean_shift": self.diagonal_sensitivity_max_mean_shift,
            "conditioning_signature": [list(x) for x in self.conditioning_signature],
            "posteriors": {
                subset_label(T): p.to_dict() for T, p in self.posteriors.items()
            },
        }


def assemble_theta(
    bootstrap: BootstrapResult,
    coordinates: Sequence[frozenset],
    doe_point: Mapping[frozenset, float],
) -> tuple[np.ndarray, np.ndarray]:
    """(θ̂, draws): the point DOE effects and the joint bootstrap draws for the
    confirmatory coordinates, aligned column-for-column to `coordinates`."""
    theta_hat = np.array([float(doe_point[T]) for T in coordinates])
    draws = np.column_stack([bootstrap.doe_draws[T] for T in coordinates])
    return theta_hat, draws


def regularise_covariance(
    V: np.ndarray, epsilon: float
) -> tuple[np.ndarray, np.ndarray, CovarianceRegularisation]:
    """Symmetrise and clip eigenvalues at ε (§7.3.3), returning BOTH the regularised
    covariance and its inverse. Clipping at `<= epsilon` (with ε > 0 enforced by the
    loader) guarantees the returned matrices are positive definite, so the posterior
    never touches an unguarded `np.linalg.inv`. Records how many eigenvalues were
    clipped and the smallest one before clipping."""
    Vs = 0.5 * (V + V.T)
    eigvals, eigvecs = np.linalg.eigh(Vs)
    min_before = float(eigvals.min()) if eigvals.size else float("nan")
    clipped = eigvals <= epsilon
    n_clipped = int(np.sum(clipped))
    eigvals_reg = np.where(clipped, epsilon, eigvals)
    V_reg = (eigvecs * eigvals_reg) @ eigvecs.T
    V_reg = 0.5 * (V_reg + V_reg.T)
    # V_inv from the same eigendecomposition — no second inversion, and finite
    # because every eigenvalue is now >= epsilon > 0.
    V_inv = (eigvecs / eigvals_reg) @ eigvecs.T
    V_inv = 0.5 * (V_inv + V_inv.T)
    return V_reg, V_inv, CovarianceRegularisation(n_clipped, epsilon, min_before)


def _material_probability(m: float, s: float, vartheta: float) -> float:
    """P(|θ| > ϑ) under the posterior N(m, s²) — the material-effect probability that feeds
    §9 classification and §8.2.3 prevalence. Used for both the headline ϑ and the sweep."""
    if s <= 0:
        p = float(abs(m) > vartheta)
    else:
        p = (1.0 - normal_cdf((vartheta - m) / s)) + normal_cdf((-vartheta - m) / s)
    return min(1.0, max(0.0, p))


def _posterior(theta_hat: np.ndarray, V_inv: np.ndarray, prior_scale: float):
    """Conjugate normal-normal posterior with prior N(0, prior_scale² I). Takes the
    (regularised) precision `V_inv` directly; `V_inv + prior_prec` is positive
    definite (ridge prior), so its inverse is the only one needed and never raises."""
    d = theta_hat.shape[0]
    prior_prec = np.eye(d) / (prior_scale ** 2)
    Sigma_post = np.linalg.inv(V_inv + prior_prec)
    mu_post = Sigma_post @ (V_inv @ theta_hat)
    return mu_post, Sigma_post


def run_bayes(
    theta_hat: np.ndarray,
    draws: np.ndarray,
    coordinates: Sequence[frozenset],
    *,
    prior_scale: float,
    vartheta: float,
    epsilon: float,
    conditioning_signature: tuple = (),
    vartheta_grid: Sequence[float] = (),
) -> BayesianResult:
    """Fit the Bayesian normal approximation over the confirmatory coordinates. When
    `vartheta_grid` is non-empty, each coordinate also carries P(|θ|>ϑ) recomputed at every
    grid ϑ (the §8.2.3/§9 neighbouring-threshold sensitivity), never replacing the headline."""
    coords = list(coordinates)
    d = len(coords)
    if d == 0:
        raise ValueError("no confirmatory coordinates — no Bayesian model (REFUSED)")

    V_boot = np.cov(draws, rowvar=False)
    V_boot = np.atleast_2d(V_boot)
    V_reg, V_inv, reg = regularise_covariance(V_boot, epsilon)

    mu_post, Sigma_post = _posterior(theta_hat, V_inv, prior_scale)
    sd = np.sqrt(np.clip(np.diag(Sigma_post), 0.0, None))

    # Diagonal-V̂ sensitivity (§7.3.3 step 4): refit with off-diagonals dropped. Its
    # precision is the reciprocal of the (positive) regularised variances — no inversion.
    V_diag_inv = np.diag(1.0 / np.diag(V_reg))
    mu_diag, _ = _posterior(theta_hat, V_diag_inv, prior_scale)
    max_shift = float(np.max(np.abs(mu_post - mu_diag))) if d else 0.0

    z = normal_ppf(0.975)
    posteriors: dict[frozenset, CoordinatePosterior] = {}
    for i, T in enumerate(coords):
        m, s = float(mu_post[i]), float(sd[i])
        p_neg = float(m < 0) if s <= 0 else normal_cdf((0.0 - m) / s)
        sweep = tuple((float(v), _material_probability(m, s, v)) for v in vartheta_grid)
        posteriors[T] = CoordinatePosterior(
            coordinate=T, mean=m, sd=s,
            ci_low=m - z * s, ci_high=m + z * s,
            p_negative=p_neg, p_positive=1.0 - p_neg,
            p_material=_material_probability(m, s, vartheta),
            p_material_sweep=sweep,
        )

    return BayesianResult(
        coordinates=tuple(coords),
        posteriors=posteriors,
        regularisation=reg,
        prior_scale=prior_scale,
        vartheta=vartheta,
        diagonal_sensitivity_max_mean_shift=max_shift,
        conditioning_signature=conditioning_signature,
    )
