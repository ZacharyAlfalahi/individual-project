"""
randomisation_fpr.py — the matched-twin exchangeability FPR (spec §6.2, D-A51).

The mean-zero-noise "placebo" is WITHDRAWN as a null: E[ε]=0 does NOT give E[I]=0 through a
nonlinear fitted model (the mean-zero fallacy the project's own errors-in-variables mechanism
refutes — a rejection would be a *true* variance-channel effect misread as a false positive).

The valid instrument is EXCHANGEABILITY, not expectation:

  1. **Matched-twin DGP:** each bond carries two i.i.d. idiosyncratic return draws (r^A, r^B) sharing
     one characteristic path and one set of true loadings — so which draw an arm sees has no causal
     role.
  2. **Placebo swap operator:** a label σ ∈ {0,1}^n decides, per bond, whether the two arms see
     (r^A, r^B) or the swapped (r^B, r^A). The panels genuinely differ (P_0 ≠ P_1); the labels are
     exchangeable by construction.
  3. Apply the full production estimator (§5.2) and compute I_obs (at σ = 0).
  4. Re-randomise σ Q times (sample fixed), recompute I under the same production rule.
  5. p = (1 + #{|I^perm| ≥ |I_obs|}) / (Q + 1).
  6. Over R independently generated datasets: FPR_hat = mean(p_r < α_nom), judged against a
     **binomial acceptance band**.

Under exact exchangeability the permutation p is uniform BY MATHEMATICS; FPR_hat therefore verifies
that the REALISED pipeline (convergence gates, tie-breaks, serialization, the production rule)
preserves that guarantee in practice. Synthetic-scale (~R(1+Q)M fits). If even the reduced-(Q,R)
fallback is infeasible, the result is RENAMED "stochastic null-invariance calibration" and NEVER
called an empirical FPR.

The anchor is a fixed synthetic test asset (a σ-independent functional of the shared true factors),
held constant across permutations — a legitimate fixed test asset that keeps I(σ) exchangeable while
avoiding a per-permutation sort rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from agents.auditor.thresholds import IPCAFprConfig, IPCALambda, IPCAProjectionGate, IPCATwinDGP
from agents.quant.library.ipca_feed import IPCAFeed

from .differential import differential_from_feeds

EMPIRICAL_FPR = "empirical_fpr"


@dataclass(frozen=True, eq=False)
class TwinDataset:
    """A matched-twin synthetic dataset: shared characteristics Z, two i.i.d. idiosyncratic return
    sets (RA, RB), and a fixed σ-independent anchor."""

    Z: list[np.ndarray]              # per-month (n, L), shared by both twins
    RA: list[np.ndarray]             # per-month (n,) — twin A idiosyncratic returns
    RB: list[np.ndarray]             # per-month (n,) — twin B idiosyncratic returns
    months: np.ndarray               # (T,)
    anchor: pd.Series                # fixed test asset, indexed by return-month period
    n_bonds: int


def make_matched_twin_dataset(twin: IPCATwinDGP, lam: IPCALambda, *, seed: int) -> TwinDataset:
    """Generate one matched-twin dataset (§6.2 step 1). RA and RB share Z Γ f; only the i.i.d.
    idiosyncratic draw differs, so the two twins are exchangeable by construction."""
    n, T = twin.n_bonds, twin.n_months
    L, K = lam.instrument_count, lam.factor_count
    rng = np.random.default_rng(seed)
    gamma, _ = np.linalg.qr(rng.standard_normal((L, K)))
    scales = np.linspace(1.5, 0.5, K)                        # distinct factor moments (clean id.)
    factors = rng.standard_normal((K, T)) * scales[:, None]

    Z: list[np.ndarray] = []
    base: list[np.ndarray] = []
    for t in range(T):
        chars = rng.uniform(-0.5, 0.5, size=(n, L - 1))
        Zt = np.column_stack([chars, np.ones(n)])            # constant LAST
        Z.append(Zt)
        base.append(Zt @ gamma @ factors[:, t])
    RA = [base[t] + rng.normal(0.0, twin.noise_sd, n) for t in range(T)]
    RB = [base[t] + rng.normal(0.0, twin.noise_sd, n) for t in range(T)]

    # Fixed anchor: a σ-independent linear functional of the shared true factors + fixed noise.
    w = np.linspace(1.0, -1.0, K)
    per = pd.PeriodIndex([pd.Period(ordinal=t + 1, freq="M") for t in range(T)])
    anchor = pd.Series(factors.T @ w + rng.normal(0.0, 0.005, T), index=per)

    return TwinDataset(
        Z=Z, RA=RA, RB=RB, months=np.arange(1, T + 1, dtype=np.int64), anchor=anchor, n_bonds=n,
    )


def _panels_from_label(twin: TwinDataset, sigma: np.ndarray) -> tuple[IPCAFeed, IPCAFeed]:
    """Build the two arm feeds under label σ (§6.2 step 2): for labelled bonds (σ=1) the two arms
    see the swapped twin. Z is shared; only R differs by the swap."""
    T, n = len(twin.months), twin.n_bonds
    R0 = [np.where(sigma == 0, twin.RA[t], twin.RB[t]) for t in range(T)]
    R1 = [np.where(sigma == 0, twin.RB[t], twin.RA[t]) for t in range(T)]
    asof = twin.months - 1
    vol = [np.ones(n) for _ in range(T)]
    cus = [np.arange(n) for _ in range(T)]
    feed0 = IPCAFeed(Z=twin.Z, R=R0, months=twin.months, asof=asof, vol_scaler=vol, cusips=cus)
    feed1 = IPCAFeed(Z=twin.Z, R=R1, months=twin.months, asof=asof, vol_scaler=vol, cusips=cus)
    return feed0, feed1


def _bracket_under_label(twin: TwinDataset, sigma: np.ndarray, lam: IPCALambda, gate: IPCAProjectionGate) -> float:
    """The interaction bracket I for one label σ, under the full production estimator."""
    feed0, feed1 = _panels_from_label(twin, sigma)
    res = differential_from_feeds("placebo", "placebo", feed0, feed1, twin.anchor, lam, gate)
    return res.interaction_bracket_raw.value


def randomisation_pvalue(
    twin: TwinDataset, lam: IPCALambda, gate: IPCAProjectionGate, *, q: int, seed: int
) -> tuple[float, float, tuple[float, ...]]:
    """One dataset's randomisation p-value (§6.2 steps 3-5): I_obs at σ=0, Q permuted labels, then
    p = (1 + #{|I^perm| ≥ |I_obs|})/(Q+1). Returns (p, i_obs, i_perms)."""
    n = twin.n_bonds
    i_obs = _bracket_under_label(twin, np.zeros(n, dtype=int), lam, gate)
    rng = np.random.default_rng(seed)
    perms: list[float] = []
    count = 0
    for _ in range(q):
        sigma = rng.integers(0, 2, size=n)
        i_perm = _bracket_under_label(twin, sigma, lam, gate)
        perms.append(i_perm)
        if abs(i_perm) >= abs(i_obs):
            count += 1
    p = (1 + count) / (q + 1)
    return p, i_obs, tuple(perms)


def binomial_acceptance_band(r: int, p: float, level: float) -> tuple[int, int]:
    """Equal-tailed exact binomial acceptance band on the reject COUNT under X ~ Binomial(r, p).
    Returns [lo, hi] such that the lower/upper tails each carry ≤ (1-level)/2 probability."""
    pmf = [comb(r, k) * (p ** k) * ((1 - p) ** (r - k)) for k in range(r + 1)]
    cdf = np.cumsum(pmf)
    sf = np.array([float(np.sum(pmf[k:])) for k in range(r + 1)])   # sf[k] = P(X >= k)
    tail = (1.0 - level) / 2.0
    lo = 0
    for k in range(r + 1):                       # lo = 1 + max{k : cdf(k) <= tail}
        if cdf[k] <= tail:
            lo = k + 1
        else:
            break
    hi = r
    for k in range(r + 1):                       # hi = -1 + min{k : sf(k) <= tail}
        if sf[k] <= tail:
            hi = k - 1
            break
    return lo, hi


@dataclass(frozen=True, eq=False)
class FprResult:
    """The randomisation-FPR outcome (§6.2 step 6). ``label`` is EMPIRICAL_FPR unless the compute
    fell back to the rename ("stochastic null-invariance calibration")."""

    r_datasets: int
    q_permutations: int
    alpha_nominal: float
    n_reject: int
    fpr_hat: float
    acceptance_band: tuple[int, int]     # count band [lo, hi]
    within_band: bool
    p_values: tuple[float, ...]
    label: str

    def to_dict(self) -> dict:
        return {
            "randomisation_fpr": {
                "r_datasets": self.r_datasets,
                "q_permutations": self.q_permutations,
                "alpha_nominal": self.alpha_nominal,
                "n_reject": self.n_reject,
                "fpr_hat": self.fpr_hat,
                "acceptance_band_counts": list(self.acceptance_band),
                "within_band": self.within_band,
                "label": self.label,
            }
        }


def randomisation_fpr(
    cfg: IPCAFprConfig,
    lam: IPCALambda,
    gate: IPCAProjectionGate,
    *,
    seed: int,
    r: int | None = None,
    q: int | None = None,
    label: str = EMPIRICAL_FPR,
) -> FprResult:
    """Run the matched-twin randomisation FPR over R datasets (§6.2). ``r``/``q`` override the
    registered defaults (e.g. the reduced-(Q,R) fallback). ``label`` becomes the rename fallback
    when even reduced compute is infeasible — the result is then NEVER called an empirical FPR."""
    n_r = cfg.r_datasets if r is None else r
    n_q = cfg.q_permutations if q is None else q
    pvals: list[float] = []
    for rr in range(n_r):
        twin = make_matched_twin_dataset(cfg.twin_dgp, lam, seed=seed + rr)
        p, _, _ = randomisation_pvalue(twin, lam, gate, q=n_q, seed=seed + 100_000 + rr)
        pvals.append(p)
    n_reject = int(sum(1 for p in pvals if p < cfg.alpha_nominal))
    lo, hi = binomial_acceptance_band(n_r, cfg.alpha_nominal, cfg.acceptance_band_level)
    return FprResult(
        r_datasets=n_r,
        q_permutations=n_q,
        alpha_nominal=cfg.alpha_nominal,
        n_reject=n_reject,
        fpr_hat=n_reject / n_r,
        acceptance_band=(lo, hi),
        within_band=lo <= n_reject <= hi,
        p_values=tuple(pvals),
        label=label,
    )
