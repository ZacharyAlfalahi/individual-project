"""
synthetic.py — feed-level synthetic fixtures for the build gates (spec §6.1, §6.5, bracket algebra).

Deliberately FEED-level: an ``IPCAFeed`` with a KNOWN Γβ and factor path, R_t = Z_t Γβ f_t (+ noise),
plus a synthetic fixed anchor. This bypasses the rank-transform receipt check (which is
``validate_panel``'s job on real feeds) so the gates can exercise the projection + fit + 2x2 algebra
on controlled inputs. The entry-boundary gate (§6.6) instead uses the real injectors in
``agents/auditor/data/synthetic_panel.py`` at the panel input boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.quant.library.ipca import IPCAFit
from agents.quant.library.ipca_feed import IPCAFeed


def make_synthetic_feed(
    *,
    L: int = 8,
    K: int = 5,
    T: int = 48,
    n: int = 40,
    seed: int = 0,
    noise_sd: float = 0.0,
    gamma: np.ndarray | None = None,
    factors: np.ndarray | None = None,
) -> tuple[IPCAFeed, dict]:
    """A feed-level synthetic IPCA panel with known truth. Z_t has L-1 characteristic columns in
    [-0.5, 0.5] plus the constant LAST; R_t = Z_t Γβ f_t (+ optional idiosyncratic noise). Returns
    (feed, truth) where truth = {gamma (L,K), factors (K,T)}. Noiseless (noise_sd=0) gives exact
    factor recovery under the frozen projection (§6.5)."""
    rng = np.random.default_rng(seed)
    if gamma is None:
        gamma, _ = np.linalg.qr(rng.standard_normal((L, K)))     # (L, K) orthonormal columns
    if factors is None:
        # Distinct factor second moments (descending) so identification is well-separated.
        scales = np.linspace(1.5, 0.5, K)
        factors = rng.standard_normal((K, T)) * scales[:, None]
    Z: list[np.ndarray] = []
    R: list[np.ndarray] = []
    months: list[int] = []
    asof: list[int] = []
    vol: list[np.ndarray] = []
    cusips: list[np.ndarray] = []
    for t in range(T):
        chars = rng.uniform(-0.5, 0.5, size=(n, L - 1))
        Zt = np.column_stack([chars, np.ones(n)])                # constant appended LAST
        rt = Zt @ gamma @ factors[:, t]
        if noise_sd > 0:
            rt = rt + rng.normal(0.0, noise_sd, size=n)
        Z.append(Zt)
        R.append(rt.astype(np.float64))
        months.append(t + 1)
        asof.append(t)
        vol.append(np.ones(n))
        cusips.append(np.arange(n))
    feed = IPCAFeed(
        Z=Z, R=R, months=np.asarray(months, dtype=np.int64),
        asof=np.asarray(asof, dtype=np.int64), vol_scaler=vol, cusips=cusips,
    )
    return feed, {"gamma": gamma, "factors": factors}


def perturb_feed(feed: IPCAFeed, *, seed: int = 1, scale: float = 0.05) -> IPCAFeed:
    """Return a copy of ``feed`` with returns perturbed (a synthetic 'bias' making P_b ≠ P_N). The
    characteristics Z_t are unchanged; only R_t moves, so the two panel states are distinct."""
    rng = np.random.default_rng(seed)
    R = [r + rng.normal(0.0, scale, size=r.shape[0]) for r in feed.R]
    return feed._replace(R=R)


def synthetic_anchor(feed: IPCAFeed, *, seed: int = 7) -> pd.Series:
    """A synthetic fixed anchor return series indexed by return-month pd.Period over feed.months."""
    rng = np.random.default_rng(seed)
    periods = pd.PeriodIndex([pd.Period(ordinal=int(m), freq="M") for m in feed.months])
    return pd.Series(rng.normal(0.0, 0.02, size=len(periods)), index=periods)


def fit_from_truth(gamma: np.ndarray, factors: np.ndarray, months: np.ndarray) -> IPCAFit:
    """Wrap a known (Γβ, factors) as an ``IPCAFit`` (restricted α=0) — used to freeze a state whose
    loadings are exactly the DGP truth, for the known-truth gate."""
    return IPCAFit(
        gamma_beta=np.asarray(gamma, dtype=np.float64),
        gamma_alpha=None,
        factors=np.asarray(factors, dtype=np.float64),
        months=np.asarray(months, dtype=np.int64),
        n_iter=0, converged=True, tol_final=0.0, init_kind="cold",
        weighting="per_month_normalized",
    )
