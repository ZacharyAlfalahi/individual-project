"""
IPCA estimator — faithful replication of Kelly–Palhares–Pruitt (KPP) Instrumented
PCA for corporate bonds. Spec: ``docs/quant/specs/ipca_spec.md`` (register-closed v1.0).

This is a PURE, DETERMINISTIC library: the Quant agent *configures* it via YAML; nothing
here is authored at run time. It consumes per-month sufficient statistics or raw matrices
(§1), never reads files beyond what the harness hands it, never imputes, and raises loudly
on contract violations. numpy + pandas-free numerical core (the average-tie rank is
reimplemented in numpy so the module has no stub-typed dependency and stays `mypy --strict`
clean); yaml appears only in the config loader.

Indexing convention (single source of truth, §1.1): panel month index ``m = 1..T`` denotes
the RETURN month. ``Z[m]`` carries characteristics dated month ``m-1`` (KPP "next-return"
convention), already rank-transformed, with the constant as the LAST column. ``R[m]`` is the
vector of scaled excess returns for month ``m``. The module asserts ``characteristic_asof ==
m-1`` so a second lag cannot be applied accidentally.

Adjudicated ambiguities (see ``docs/quant/registers/ipca_adjudications.md``):
- R1: the rank-transform receipt check is lane-aware (strict ``max==+0.5`` only on the
  faithful lane; on the demeaned lane verify ``mean≈0`` + bounded, non-zero spread).
- R2: identification is made a deterministic function of the subspace — orthonormalise via
  sign-stabilised QR, order by descending unconditional second moment, pin eigenvector signs
  by largest-abs-component-positive, then apply KPP's ``mean ≥ 0`` rule with a near-zero-mean
  fallback to the eigenvector sign. Guarantees the idempotency the battery (test 8) requires.
- R3 (bootstrap, added with the rest of the module): hold ``W[m]`` fixed, resample only the
  managed-portfolio return ``x`` — conditioning on the realised characteristic cross-section.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ContractViolation(ValueError):
    """A data-contract or degeneracy violation (§1).

    Subclasses ``ValueError`` so existing ``except ValueError`` paths still catch it, while
    giving the spec-named type that contract tests assert against precisely.
    """


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


class SuffStats(NamedTuple):
    """Per-month sufficient statistics (§1.2). Months are ascending.

    W[m] = Z[m]'Z[m] / N_m   (T, L, L)
    x[m] = Z[m]'R[m] / N_m   (T, L)        — the managed-portfolio return vector (eq. 4)
    N[m] = N_m               (T,) int
    months                   (T,) int period labels
    """

    W: np.ndarray
    x: np.ndarray
    N: np.ndarray
    months: np.ndarray


class IPCAFit(NamedTuple):
    """In-sample IPCA estimate (identified)."""

    gamma_beta: np.ndarray            # (L, K), orthonormal columns
    gamma_alpha: np.ndarray | None    # (L, 1) when alpha is on, else None
    factors: np.ndarray               # (K, T) identified factor path
    months: np.ndarray                # (T,)
    n_iter: int
    converged: bool
    tol_final: float
    init_kind: str                    # "cold" | "warm"
    weighting: str                    # K22 arm actually used


Weighting = Literal["per_month_normalized", "observation_weighted"]

# Numerical tolerances for the §1.3 receipt check (the rank map produces exact halves).
_RANK_EPS = 1e-9
_MEAN_EPS = 1e-8


# ---------------------------------------------------------------------------
# Canonical rank transform (§1.3)
# ---------------------------------------------------------------------------


def _average_tie_ranks(v: np.ndarray) -> np.ndarray:
    """1-based average-tie ranks over non-missing entries (``tiedrank`` semantics).

    NaNs are preserved (never ranked). Vectorised reimplementation of
    ``scipy.stats.rankdata(method="average")`` so the module carries no scipy dependency.
    """
    out = np.full(v.shape[0], np.nan, dtype=np.float64)
    mask = ~np.isnan(v)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    a = v[idx]
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = np.cumsum(obs)[inv]                              # dense rank 1..g
    boundaries = np.r_[np.flatnonzero(obs), sorter.size]     # group start positions + total
    avg = 0.5 * (boundaries[dense - 1] + boundaries[dense] + 1.0)
    out[idx] = avg
    return out


def rank_transform(values: np.ndarray, *, demean_after_map: bool = False) -> np.ndarray:
    """Map raw characteristic values to the KPP rank space (§1.3).

    Steps: average-tie ranks over non-missing; ``(rank-1)/(rmax-1) - 0.5`` where ``rmax`` is
    the max tied rank (NOT ``(rank-0.5)/N``); NaNs preserved; constant column is NOT appended
    here (the caller/panel layer appends it, §1.3.4). With top-ties the cross-sectional mean
    is strictly positive and KPP leave it so — there is NO re-demeaning in the faithful lane.

    ``demean_after_map=True`` subtracts the cross-sectional mean (corrected-lane experiments
    only); the caller must log this as a deviation.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    ranks = _average_tie_ranks(v)
    finite = ~np.isnan(ranks)
    out = np.full(v.shape, np.nan, dtype=np.float64)
    if not finite.any():
        return out
    rmax = float(np.nanmax(ranks))
    if rmax > 1.0:
        out[finite] = (ranks[finite] - 1.0) / (rmax - 1.0) - 0.5
    else:
        out[finite] = 0.0   # single non-missing obs maps to the centre
    if demean_after_map:
        out[finite] = out[finite] - float(np.mean(out[finite]))
    return out


# ---------------------------------------------------------------------------
# Data-contract validation (§1)
# ---------------------------------------------------------------------------


def validate_panel(
    Z: list[np.ndarray],
    R: list[np.ndarray],
    months: np.ndarray,
    *,
    L: int,
    family: str,
    characteristic_asof: np.ndarray,
    demean_after_map: bool = False,
) -> None:
    """Enforce the §1 data contract on the raw per-month matrices; raise ``ContractViolation``.

    Checks: shapes; ``N_m > L`` (§1.4, hard raise — no silent skip); no NaN anywhere (§1.4);
    the rank-transform receipt check (§1.3, lane-aware per R1); the constant column ≡ 1; that
    no non-constant column is degenerate (constant within a month → singular W); the
    double-lag guard ``characteristic_asof[m] == months[m] - 1`` (§1.5); a single family tag
    (§1.5). Complete-case selection itself is the panel layer's job (§1.4).
    """
    if family not in ("raw", "corr"):
        raise ContractViolation(f"family must be 'raw' or 'corr'; got {family!r}")
    T = len(Z)
    if not (len(R) == T == len(months) == len(characteristic_asof)):
        raise ContractViolation(
            f"Z, R, months, characteristic_asof must align; got lengths "
            f"{len(Z)}, {len(R)}, {len(months)}, {len(characteristic_asof)}"
        )
    months = np.asarray(months)
    asof = np.asarray(characteristic_asof)
    for m in range(T):
        Zm = np.asarray(Z[m], dtype=np.float64)
        Rm = np.asarray(R[m], dtype=np.float64)
        n = Zm.shape[0]
        if Zm.ndim != 2 or Zm.shape[1] != L:
            raise ContractViolation(f"Z[{m}] must be (N_m, {L}); got {Zm.shape}")
        if Rm.shape != (n,):
            raise ContractViolation(f"R[{m}] must be ({n},); got {Rm.shape}")
        if n <= L:
            raise ContractViolation(f"month {m}: N_m={n} <= L={L} (degenerate; §1.4)")
        if np.isnan(Zm).any() or np.isnan(Rm).any():
            raise ContractViolation(f"month {m}: NaN in Z or R (§1.4 — panel must complete-case)")
        const = Zm[:, -1]
        if not np.all(np.abs(const - 1.0) <= _RANK_EPS):
            raise ContractViolation(f"month {m}: last column must be the constant ≡ 1")
        cols = Zm[:, :-1]
        cmin = cols.min(axis=0)
        cmax = cols.max(axis=0)
        if np.any(cmax - cmin <= 0.0):
            raise ContractViolation(
                f"month {m}: a non-constant instrument is degenerate (no cross-sectional spread)"
            )
        if not demean_after_map:
            if np.any(cmin < -0.5 - _RANK_EPS) or np.any(cmax > 0.5 + _RANK_EPS):
                raise ContractViolation(f"month {m}: ranked column out of [-0.5, 0.5] (§1.3)")
            if np.any(np.abs(cmax - 0.5) > _RANK_EPS):
                raise ContractViolation(f"month {m}: ranked column max != +0.5 (§1.3, faithful lane)")
        else:
            if np.any(np.abs(cols.mean(axis=0)) > _MEAN_EPS):
                raise ContractViolation(f"month {m}: demeaned column mean != 0 (§1.3 corrected lane)")
            if np.any(cmax - cmin > 1.0 + _RANK_EPS):
                raise ContractViolation(f"month {m}: demeaned column spread > 1 (§1.3 corrected lane)")
        if int(asof[m]) != int(months[m]) - 1:
            raise ContractViolation(
                f"month {m}: characteristic_asof={asof[m]} != months-1={int(months[m]) - 1} "
                f"(double-lag guard, §1.5)"
            )


# ---------------------------------------------------------------------------
# Sufficient statistics (§1.2)
# ---------------------------------------------------------------------------


def build_sufficient_stats(
    Z: list[np.ndarray], R: list[np.ndarray], months: np.ndarray
) -> SuffStats:
    """Compute per-month W[m]=Z'Z/N, x[m]=Z'R/N, N[m] (§1.2). One streaming pass.

    Raises ``ContractViolation`` on ``N_m <= L`` (no silent skip; §1.4). This is the only
    place raw bond rows are touched for estimation; the ALS/OOS/bootstrap operate on (W, x).
    """
    T = len(Z)
    if T == 0:
        raise ContractViolation("empty panel")
    L = int(np.asarray(Z[0]).shape[1])
    W = np.empty((T, L, L), dtype=np.float64)
    x = np.empty((T, L), dtype=np.float64)
    N = np.empty(T, dtype=np.int64)
    for m in range(T):
        Zm = np.asarray(Z[m], dtype=np.float64)
        Rm = np.asarray(R[m], dtype=np.float64)
        n = Zm.shape[0]
        if n <= L:
            raise ContractViolation(f"month {m}: N_m={n} <= L={L} (§1.4)")
        W[m] = Zm.T @ Zm / n
        x[m] = Zm.T @ Rm / n
        N[m] = n
    return SuffStats(W=W, x=x, N=N, months=np.asarray(months))


# ---------------------------------------------------------------------------
# ALS core (§2)
# ---------------------------------------------------------------------------


def _month_weights(weighting: str, N: np.ndarray) -> np.ndarray:
    """Per-month weight c_m (§2.4): 1 (per_month_normalized) or N_m (observation_weighted)."""
    if weighting == "per_month_normalized":
        return np.ones(N.shape[0], dtype=np.float64)
    if weighting == "observation_weighted":
        return N.astype(np.float64)
    raise ValueError(f"weighting must be per_month_normalized | observation_weighted; got {weighting!r}")


def _build_g(F: np.ndarray, n_psf: int) -> np.ndarray:
    """g_m = [f_m; 1] when alpha is on (n_psf=1), else g_m = f_m. F is (K, T) -> (K+n_psf, T)."""
    if n_psf == 0:
        return F
    return np.vstack([F, np.ones((1, F.shape[1]), dtype=np.float64)])


def _factor_step(
    W: np.ndarray, x: np.ndarray, gamma_beta: np.ndarray, gamma_alpha: np.ndarray | None
) -> np.ndarray:
    """Factor step (§2.2.1): f_m = (Γβ' W[m] Γβ)^{-1} Γβ' (x[m] - W[m] Γα). Returns F (K, T).

    Solved via batched ``np.linalg.solve`` (Cholesky-equivalent LU; never an explicit inverse).
    """
    A = np.einsum("ak,mab,bj->mkj", gamma_beta, W, gamma_beta)   # (T, K, K)
    rhs = np.einsum("ak,ma->mk", gamma_beta, x)                  # (T, K)
    if gamma_alpha is not None:
        rhs = rhs - np.einsum("ak,mab,b->mk", gamma_beta, W, gamma_alpha[:, 0])
    F = np.linalg.solve(A, rhs[..., None])[..., 0]              # (T, K); explicit RHS column
    return F.T


def _gamma_step(
    W: np.ndarray, x: np.ndarray, g: np.ndarray, c: np.ndarray, L: int, K: int, n_psf: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Γ step (§2.2.2): solve [Σ c_m W[m]⊗g_m g_m'] vec(Γ') = Σ c_m x[m]⊗g_m (einsum form).

    C-order ``p = l*Kg + k`` coincides with the spec's ``vec(Γ')`` so the solution reshapes
    directly to Γ[l, k]. See ``_gamma_step_reference`` for the explicit-kron cross-check.
    """
    Kg = K + n_psf
    A = np.einsum("m,mab,km,jm->akbj", c, W, g, g).reshape(L * Kg, L * Kg)
    b = np.einsum("m,ml,km->lk", c, x, g).reshape(L * Kg)
    Gamma = np.linalg.solve(A, b).reshape(L, Kg)
    gamma_beta = Gamma[:, :K]
    gamma_alpha = Gamma[:, K:] if n_psf else None
    return gamma_beta, gamma_alpha


def _gamma_step_reference(
    W: np.ndarray, x: np.ndarray, g: np.ndarray, c: np.ndarray, L: int, K: int, n_psf: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Explicit-kron reference form of the Γ step (battery cross-check to 1e-12; §2.2)."""
    Kg = K + n_psf
    A = np.zeros((L * Kg, L * Kg), dtype=np.float64)
    b = np.zeros(L * Kg, dtype=np.float64)
    for m in range(W.shape[0]):
        gm = g[:, m]
        A += c[m] * np.kron(W[m], np.outer(gm, gm))
        b += c[m] * np.kron(x[m], gm)
    Gamma = np.linalg.solve(A, b).reshape(L, Kg)
    return Gamma[:, :K], (Gamma[:, K:] if n_psf else None)


def _identify(
    gamma_beta: np.ndarray, F: np.ndarray, *, sign_fallback_mean_eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical identification applied INSIDE every iteration (§2.2.3); R2 determinism rules.

    (1) orthonormalise Γβ via sign-stabilised QR (R-diagonal forced positive) so Γβ'Γβ=I and
        the fit Γβ F is preserved; (2) order factors by descending unconditional second
        moment, pinning each eigenvector's sign by largest-abs-component-positive; (3) sign
        each factor by KPP's mean ≥ 0 rule, falling back to the (already-deterministic)
        eigenvector sign when |mean| < sign_fallback_mean_eps. The whole map is a deterministic
        function of span(Γβ) and is idempotent (battery test 8).
    """
    K = gamma_beta.shape[1]
    T = F.shape[1]
    # (1) sign-stabilised QR
    Q, Rqr = np.linalg.qr(gamma_beta)
    d = np.sign(np.diag(Rqr))
    d[d == 0.0] = 1.0
    gb = Q * d
    Fr = (d[:, None] * Rqr) @ F
    # (2) order by descending second moment; deterministic eigenvector signs
    M = (Fr @ Fr.T) / T
    evals, U = np.linalg.eigh(M)
    order = np.argsort(evals)[::-1]
    U = U[:, order]
    for k in range(K):
        col = U[:, k]
        if col[int(np.argmax(np.abs(col)))] < 0.0:
            U[:, k] = -col
    gb = gb @ U
    Fr = U.T @ Fr
    # (3) mean >= 0 final arbiter, near-zero-mean fallback to the eigenvector sign
    for k in range(K):
        mean_k = float(Fr[k].mean())
        if abs(mean_k) >= sign_fallback_mean_eps and mean_k < 0.0:
            Fr[k] = -Fr[k]
            gb[:, k] = -gb[:, k]
    return gb, Fr


def _svd_cold_start(x: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    """Cold start (§2.3): Γ0 = top-K left singular vectors of the L×T managed-portfolio
    matrix, F0 = s V'."""
    X = x.T                                                  # (L, T)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    gamma0 = U[:, :K].copy()
    F0 = s[:K, None] * Vt[:K, :]
    return gamma0, F0


def _als_loop(
    stats: SuffStats,
    *,
    K: int,
    n_psf: int,
    c: np.ndarray,
    gamma0: np.ndarray,
    F0: np.ndarray,
    tol: float,
    max_iter: int,
    sign_fallback_mean_eps: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, int, bool, float]:
    """Run the ALS to convergence (§2.3 loop semantics: iter from 0, ``while iter<=cap and
    tol>tolerance``; convergence measured on the IDENTIFIED iterates). Returns
    (Γβ, Γα|None, F, n_iter, converged, tol_final)."""
    W, x = stats.W, stats.x
    L = W.shape[1]
    gamma_alpha: np.ndarray | None = np.zeros((L, 1), dtype=np.float64) if n_psf == 1 else None
    gamma_beta, F = _identify(gamma0, F0, sign_fallback_mean_eps=sign_fallback_mean_eps)
    it = 0
    cur_tol = np.inf
    while it <= max_iter and cur_tol > tol:
        gb_prev = gamma_beta
        F_prev = F
        F = _factor_step(W, x, gamma_beta, gamma_alpha)
        g = _build_g(F, n_psf)
        gamma_beta, gamma_alpha = _gamma_step(W, x, g, c, L, K, n_psf)
        gamma_beta, F = _identify(gamma_beta, F, sign_fallback_mean_eps=sign_fallback_mean_eps)
        cur_tol = max(
            float(np.max(np.abs(gamma_beta - gb_prev))), float(np.max(np.abs(F - F_prev)))
        )
        it += 1
    return gamma_beta, gamma_alpha, F, it, bool(cur_tol <= tol), float(cur_tol)


def _warn_identification_degeneracy(
    F: np.ndarray, sign_fallback_mean_eps: float, near_tie_moment_eps: float
) -> None:
    """Warn once (not per-iteration) if the identified factors have near-equal second moments
    (rotation under-determined) or a near-zero mean (sign pinned by the eigenvector rule)."""
    T = F.shape[1]
    moments = np.sort(np.diag((F @ F.T) / T))[::-1]
    if moments.size > 1:
        gaps = np.abs(np.diff(moments))
        scale = max(float(moments[0]), 1e-300)
        if float(np.min(gaps)) < near_tie_moment_eps * scale:
            warnings.warn(
                "IPCA identification: near-equal factor second moments — ordering rotation is "
                "under-determined; sign/order are deterministic but may be sensitive.",
                RuntimeWarning,
                stacklevel=2,
            )
    means = np.abs(F.mean(axis=1))
    if np.any(means < sign_fallback_mean_eps):
        warnings.warn(
            "IPCA identification: near-zero factor mean — sign pinned by the eigenvector rule "
            "(mean ≥ 0 arbiter uninformative).",
            RuntimeWarning,
            stacklevel=2,
        )


def fit_ipca(
    stats: SuffStats,
    *,
    K: int,
    alpha: bool = False,
    weighting: Weighting = "per_month_normalized",
    tol: float = 1e-4,
    max_iter: int = 1000,
    gamma0: np.ndarray | None = None,
    F0: np.ndarray | None = None,
    sign_fallback_mean_eps: float = 1e-8,
    near_tie_moment_eps: float = 1e-8,
) -> IPCAFit:
    """Fit the IPCA model in sample (§2).

    ``alpha`` toggles the constant-PSF column (Appendix-B alpha model, §2.1). ``weighting`` is
    the K22 arm (§2.4). ``gamma0/F0`` supply a warm start (recursion / bootstrap, §2.3);
    otherwise the SVD cold start is used. Raises ``ContractViolation`` on any singular system
    (no regularisation anywhere; §1.4).
    """
    W, x, N = stats.W, stats.x, stats.N
    T, L, _ = W.shape
    if not 1 <= K <= min(L, T):
        raise ValueError(f"K must be in [1, min(L,T)={min(L, T)}]; got {K}")
    if np.any(N <= L):
        raise ContractViolation("a month has N_m <= L (§1.4)")
    n_psf = 1 if alpha else 0
    c = _month_weights(weighting, N)
    init_kind = "cold"
    if gamma0 is None or F0 is None:
        gamma0, F0 = _svd_cold_start(x, K)
    else:
        init_kind = "warm"
    try:
        gb, ga, F, it, conv, tol_final = _als_loop(
            stats, K=K, n_psf=n_psf, c=c, gamma0=gamma0, F0=F0, tol=tol,
            max_iter=max_iter, sign_fallback_mean_eps=sign_fallback_mean_eps,
        )
    except np.linalg.LinAlgError as exc:
        raise ContractViolation(f"singular system during ALS (§1.4): {exc}") from exc
    _warn_identification_degeneracy(F, sign_fallback_mean_eps, near_tie_moment_eps)
    return IPCAFit(
        gamma_beta=gb, gamma_alpha=ga, factors=F, months=stats.months,
        n_iter=it, converged=conv, tol_final=tol_final, init_kind=init_kind, weighting=weighting,
    )


# ---------------------------------------------------------------------------
# Fit metrics (§5) — zero benchmark (no demeaning in any denominator)
# ---------------------------------------------------------------------------


def _month_loading(fit: IPCAFit, m: int) -> np.ndarray:
    """Per-month bond loading vector λ_m = Γβ f_m (+ Γα) so fitted r̂_i = Z_i' λ_m (§5)."""
    load = fit.gamma_beta @ fit.factors[:, m]
    if fit.gamma_alpha is not None:
        load = load + fit.gamma_alpha[:, 0]
    return np.asarray(load, dtype=np.float64)


def total_r2(stats: SuffStats, fit: IPCAFit) -> float:
    """Managed-portfolio total R² = 1 − Σ‖x − Wλ‖² / Σ‖x‖² (no demeaning; §5)."""
    num = 0.0
    den = 0.0
    for m in range(stats.W.shape[0]):
        xhat = stats.W[m] @ _month_loading(fit, m)
        num += float(np.sum((stats.x[m] - xhat) ** 2))
        den += float(np.sum(stats.x[m] ** 2))
    return 1.0 - num / den


def bond_total_r2(Z: list[np.ndarray], R: list[np.ndarray], fit: IPCAFit) -> float:
    """Individual-bond total R² = 1 − Σ(r − Zλ)² / Σ r² (streaming over {Z,R}; §5)."""
    num = 0.0
    den = 0.0
    for m in range(len(Z)):
        fitted = np.asarray(Z[m], dtype=np.float64) @ _month_loading(fit, m)
        rm = np.asarray(R[m], dtype=np.float64)
        num += float(np.sum((rm - fitted) ** 2))
        den += float(np.sum(rm ** 2))
    return 1.0 - num / den


def cross_section_r2(
    Z: list[np.ndarray], R: list[np.ndarray], fit: IPCAFit, *, reestimated: bool = False
) -> float:
    """Cross-section R² pooled over periods (§5, eq. 8 + fn. 11).

    ``reestimated=False`` (the model metric): the slope is the realised factor return itself —
    fitted r̂_i = Z_i'Γβ f_t (+ Γα), never re-estimated. ``reestimated=True`` re-estimates the
    cross-sectional slope each period by OLS of r_t on the betas — the misspecification-
    revealing Table-V-Panel-E diagnostic, NOT the model metric.
    """
    num = 0.0
    den = 0.0
    for m in range(len(Z)):
        Zm = np.asarray(Z[m], dtype=np.float64)
        rm = np.asarray(R[m], dtype=np.float64)
        beta = Zm @ fit.gamma_beta                       # (n, K)
        if reestimated:
            lam, *_ = np.linalg.lstsq(beta, rm, rcond=None)
            fitted = beta @ lam
        else:
            fitted = beta @ fit.factors[:, m]
            if fit.gamma_alpha is not None:
                fitted = fitted + Zm @ fit.gamma_alpha[:, 0]
        num += float(np.sum((rm - fitted) ** 2))
        den += float(np.sum(rm ** 2))
    return 1.0 - num / den


def _bond_panel_long(
    Z: list[np.ndarray], R: list[np.ndarray], bond_ids: list[np.ndarray], fit: IPCAFit
) -> dict[int, tuple[list[float], list[float]]]:
    """Collect (return, fitted) pairs per bond id across months (for §5 TS-R² and RPE)."""
    acc: dict[int, tuple[list[float], list[float]]] = {}
    for m in range(len(Z)):
        Zm = np.asarray(Z[m], dtype=np.float64)
        rm = np.asarray(R[m], dtype=np.float64)
        fitted = Zm @ _month_loading(fit, m)
        ids = np.asarray(bond_ids[m])
        for i in range(rm.shape[0]):
            bid = int(ids[i])
            rs, fs = acc.setdefault(bid, ([], []))
            rs.append(float(rm[i]))
            fs.append(float(fitted[i]))
    return acc


def timeseries_r2(
    Z: list[np.ndarray], R: list[np.ndarray], bond_ids: list[np.ndarray], fit: IPCAFit
) -> float:
    """Time-series R²: per-asset R², then T_i-weighted average (§5, eq. 7)."""
    acc = _bond_panel_long(Z, R, bond_ids, fit)
    wsum = 0.0
    rsum = 0.0
    for rs, fs in acc.values():
        r = np.asarray(rs)
        f = np.asarray(fs)
        den = float(np.sum(r ** 2))
        if den == 0.0:
            continue
        r2_i = 1.0 - float(np.sum((r - f) ** 2)) / den
        rsum += r.shape[0] * r2_i
        wsum += r.shape[0]
    return rsum / wsum if wsum > 0 else float("nan")


def relative_pricing_error(
    Z: list[np.ndarray], R: list[np.ndarray], bond_ids: list[np.ndarray], fit: IPCAFit
) -> float:
    """Relative pricing error = Σ ᾱ_i² / Σ r̄_i² (§5, eq. 9); may exceed 100% OOS (fn. 3).

    ᾱ_i is bond i's time-average pricing error (mean residual); r̄_i its mean return.
    """
    acc = _bond_panel_long(Z, R, bond_ids, fit)
    num = 0.0
    den = 0.0
    for rs, fs in acc.values():
        r = np.asarray(rs)
        f = np.asarray(fs)
        num += float(np.mean(r - f)) ** 2
        den += float(np.mean(r)) ** 2
    return num / den if den > 0 else float("nan")


def aggregate_bond_fit(
    Z: list[np.ndarray],
    R: list[np.ndarray],
    bond_ids: list[np.ndarray],
    fit: IPCAFit,
    portfolio_map: dict[int, str],
) -> dict[str, float]:
    """Aggregate bond-level fits to harness-supplied portfolio groups (§5): per-group total R².

    ``portfolio_map`` maps bond id → group label (e.g. size/maturity bucket, industry).
    """
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for m in range(len(Z)):
        Zm = np.asarray(Z[m], dtype=np.float64)
        rm = np.asarray(R[m], dtype=np.float64)
        fitted = Zm @ _month_loading(fit, m)
        ids = np.asarray(bond_ids[m])
        for i in range(rm.shape[0]):
            grp = portfolio_map.get(int(ids[i]))
            if grp is None:
                continue
            num[grp] = num.get(grp, 0.0) + float((rm[i] - fitted[i]) ** 2)
            den[grp] = den.get(grp, 0.0) + float(rm[i] ** 2)
    return {grp: 1.0 - num[grp] / den[grp] for grp in num if den[grp] > 0}


# ---------------------------------------------------------------------------
# Windows and the wall (§7)
# ---------------------------------------------------------------------------


def assert_within_wall(months: np.ndarray, *, train_end: int) -> None:
    """Refuse any observation beyond ``train_end`` (§7). The module-side realisation of the
    train-end wall for a pure library that receives (rather than reads) data."""
    mx = np.asarray(months)
    if mx.size and int(np.max(mx)) > train_end:
        raise ContractViolation(
            f"observation month {int(np.max(mx))} > train_end={train_end} (wall, §7)"
        )


def select_window(months: np.ndarray, *, start: int, end: int, train_end: int) -> np.ndarray:
    """Boolean mask for months in [start, end]; raises if end exceeds the wall (§7).

    Window-name → (start, end) resolution lives in the config/harness; the module enforces
    the bound on whatever period labels it is given.
    """
    if end > train_end:
        raise ContractViolation(f"window end {end} exceeds train_end={train_end} (§7)")
    mx = np.asarray(months)
    return (mx >= start) & (mx <= end)


# ---------------------------------------------------------------------------
# Scaling-lane diagnostics (§6) — verify/diagnose only; the panel layer applies scaling
# ---------------------------------------------------------------------------


def diagnose_scaling_lane(
    lane: str,
    *,
    dts_values: np.ndarray | None = None,
    vol_values: np.ndarray | None = None,
    dts_floor: float = 0.25,
    vol_floor: float = 0.01,
    composition_warn: tuple[float, float] = (0.20, 0.60),
) -> dict[str, float | str]:
    """Diagnose a return-scaling lane tag (§6). The module never COMPUTES scaling — it
    verifies/diagnoses the lane the panel layer applied.

    DTS: report the below-floor share, warn if outside ``composition_warn``. VOL: assert
    decimal-monthly units (raise on percent-scale input), report the below-floor share. Raises
    ``ContractViolation`` only on a hard unit violation.
    """
    out: dict[str, float | str] = {"lane": lane}
    if lane == "Unscaled":
        return out
    if lane == "DTSScaled25":
        if dts_values is None:
            raise ValueError("DTSScaled25 requires dts_values for diagnostics")
        v = np.asarray(dts_values, dtype=np.float64)
        share = float(np.mean(v < dts_floor))
        out["below_floor_share"] = share
        if not composition_warn[0] <= share <= composition_warn[1]:
            warnings.warn(
                f"DTS below-floor share {share:.2%} outside {composition_warn} (§6 composition check)",
                RuntimeWarning, stacklevel=2,
            )
        return out
    if lane == "VOLScaled010":
        if vol_values is None:
            raise ValueError("VOLScaled010 requires vol_values for diagnostics")
        v = np.asarray(vol_values, dtype=np.float64)
        med = float(np.nanmedian(v))
        if med >= 1.0:
            raise ContractViolation(
                f"VOL scaler median {med:.3f} >= 1 — expected decimal monthly units, not percent (§6)"
            )
        # "below-floor share" = fraction actually floored, i.e. strictly below the floor
        # (a value exactly at the floor is unchanged by max(v, floor)); matches the DTS branch.
        out["below_floor_share"] = float(np.mean(v[~np.isnan(v)] < vol_floor))
        return out
    raise ValueError(f"lane must be DTSScaled25 | VOLScaled010 | Unscaled; got {lane!r}")


# ---------------------------------------------------------------------------
# Out-of-sample recursion + strategies (§4)
# ---------------------------------------------------------------------------


class RecursiveResult(NamedTuple):
    """Recursive OOS output (§4). Arrays are ordered by ascending OOS prediction month."""

    oos_months: np.ndarray            # (J,) predicted months (t+1)
    f_oos: np.ndarray                 # (K, J) eq-(2) OOS factor realisations
    xhat_oos: np.ndarray              # (L, J) W_{t+1} Γ̂ f̂_{t+1}
    x_oos: np.ndarray                 # (L, J) realised managed portfolios x[t+1]
    lambdas: np.ndarray               # (K, J) λ̂_t = mean in-sample factor path at t
    insample_mean: np.ndarray         # (K, J) μ_t
    insample_cov: np.ndarray          # (J, K, K) S_t
    gammas: list[np.ndarray]          # Γ̂_t per step (L, K)
    n_iters: list[int]
    init_kinds: list[str]


class StratResult(NamedTuple):
    returns: np.ndarray               # realised strategy returns per period
    sharpe: float
    weights: np.ndarray | None        # weight path (J, A) if applicable
    turnover: np.ndarray | None


def _oos_factor_realization(
    W_next: np.ndarray, x_next: np.ndarray, gamma_beta: np.ndarray,
    gamma_alpha: np.ndarray | None = None,
) -> np.ndarray:
    """f̂_{t+1} = (Γ̂'W[t+1]Γ̂)^{-1} Γ̂'(x[t+1] − W[t+1]Γα) with Γ̂ from data through t (eq. 2)."""
    A = gamma_beta.T @ W_next @ gamma_beta
    rhs = gamma_beta.T @ x_next
    if gamma_alpha is not None:
        rhs = rhs - gamma_beta.T @ W_next @ gamma_alpha[:, 0]
    try:
        return np.asarray(np.linalg.solve(A, rhs), dtype=np.float64)
    except np.linalg.LinAlgError as exc:
        raise ContractViolation(f"singular Γ'W[t+1]Γ at OOS factor realisation (§1.4): {exc}") from exc


def fit_ipca_recursive(
    stats: SuffStats,
    *,
    K: int,
    alpha: bool = False,
    weighting: Weighting = "per_month_normalized",
    burn_in: int = 36,
    tol: float = 1e-4,
    max_iter: int = 2000,
    warm_start: bool = True,
    sign_fallback_mean_eps: float = 1e-8,
) -> RecursiveResult:
    """Recursive OOS estimation (§4): re-estimate on data through t for t∈{burn_in..T-1},
    predict t+1. Warm-starts at the previous Γ̂ with F0 = [F̂, mean(F̂)] (§2.3). The in-sample
    factor moments at each t use the FULL in-sample path under the current Γ̂ (§4, K5)."""
    T, L, _ = stats.W.shape
    if burn_in >= T:
        raise ValueError(f"burn_in={burn_in} must be < T={T}")
    oos_months: list[int] = []
    f_oos: list[np.ndarray] = []
    xhat: list[np.ndarray] = []
    xact: list[np.ndarray] = []
    lambdas: list[np.ndarray] = []
    mus: list[np.ndarray] = []
    covs: list[np.ndarray] = []
    gammas: list[np.ndarray] = []
    n_iters: list[int] = []
    kinds: list[str] = []
    prev_gamma: np.ndarray | None = None
    prev_F: np.ndarray | None = None
    for end in range(burn_in, T):
        sub = SuffStats(W=stats.W[:end], x=stats.x[:end], N=stats.N[:end], months=stats.months[:end])
        g0: np.ndarray | None = None
        f0: np.ndarray | None = None
        if warm_start and prev_gamma is not None and prev_F is not None:
            g0 = prev_gamma
            f0 = np.column_stack([prev_F, prev_F.mean(axis=1)])
        fit = fit_ipca(
            sub, K=K, alpha=alpha, weighting=weighting, tol=tol, max_iter=max_iter,
            gamma0=g0, F0=f0, sign_fallback_mean_eps=sign_fallback_mean_eps,
        )
        gamma = fit.gamma_beta
        Fis = fit.factors
        prev_gamma, prev_F = gamma, Fis
        f_next = _oos_factor_realization(stats.W[end], stats.x[end], gamma, fit.gamma_alpha)
        load_next = gamma @ f_next
        if fit.gamma_alpha is not None:
            load_next = load_next + fit.gamma_alpha[:, 0]   # alpha model: x̂ = W(Γβ f̂ + Γα)
        oos_months.append(int(stats.months[end]))
        f_oos.append(f_next)
        xhat.append(stats.W[end] @ load_next)
        xact.append(stats.x[end])
        lambdas.append(Fis.mean(axis=1))
        mus.append(Fis.mean(axis=1))
        covs.append(np.atleast_2d(np.cov(Fis, ddof=1)))
        gammas.append(gamma)
        n_iters.append(fit.n_iter)
        kinds.append(fit.init_kind)
    return RecursiveResult(
        oos_months=np.asarray(oos_months), f_oos=np.asarray(f_oos).T,
        xhat_oos=np.asarray(xhat).T, x_oos=np.asarray(xact).T, lambdas=np.asarray(lambdas).T,
        insample_mean=np.asarray(mus).T, insample_cov=np.asarray(covs), gammas=gammas,
        n_iters=n_iters, init_kinds=kinds,
    )


def oos_total_r2(rec: RecursiveResult) -> float:
    """OOS managed total R² = 1 − Σ‖x[t+1] − x̂[t+1]‖² / Σ‖x[t+1]‖² (no demeaning; §5/§4)."""
    num = float(np.sum((rec.x_oos - rec.xhat_oos) ** 2))
    den = float(np.sum(rec.x_oos ** 2))
    return 1.0 - num / den


# ---- strategy / cost primitives (§4.1–4.3) --------------------------------


def sharpe(returns: np.ndarray, *, periods_per_year: int = 12) -> float:
    """Annualised Sharpe with POPULATION std (ddof=0, matching KPP ``nanstd(·,1)``; §4.1)."""
    r = np.asarray(returns, dtype=np.float64)
    sd = float(np.std(r, ddof=0))
    if sd == 0.0:
        return float("nan")
    return float(float(np.mean(r)) / sd * float(np.sqrt(periods_per_year)))


def turnover(weights_path: np.ndarray) -> np.ndarray:
    """Per-period turnover Σ_i |w_{i,t} − w_{i,t-1}| (§4.3, eq. 13); first period vs zero."""
    w = np.asarray(weights_path, dtype=np.float64)
    prev = np.vstack([np.zeros((1, w.shape[1])), w[:-1]])
    return np.asarray(np.sum(np.abs(w - prev), axis=1), dtype=np.float64)


def apply_costs(returns: np.ndarray, weights_path: np.ndarray, *, cost_bp: float = 19.0) -> np.ndarray:
    """Net return = gross − (cost_bp/1e4)·turnover (eq. 12; 19 bp one-way on all trades; §4.3)."""
    return np.asarray(returns, dtype=np.float64) - (cost_bp / 1e4) * turnover(weights_path)


def smooth_weights(weights_path: np.ndarray, gamma: float) -> np.ndarray:
    """Exponential weight smoothing w̃_t = (1−γ)w_t + γ w̃_{t-1}, w̃_0 = w_0 (eq. 14; §4.3)."""
    if not 0.0 <= gamma < 1.0:
        raise ValueError(f"gamma must be in [0, 1); got {gamma}")
    w = np.asarray(weights_path, dtype=np.float64)
    out = np.empty_like(w)
    out[0] = w[0]
    for t in range(1, w.shape[0]):
        out[t] = (1.0 - gamma) * w[t] + gamma * out[t - 1]
    return out


def tangency_weights(mu: np.ndarray, S: np.ndarray, *, vol_target: float | None = 0.01) -> np.ndarray:
    """Tangency weights (§4.1): w = S⁻¹μ / (ι'S⁻¹μ), sign-flipped if w'μ<0. With ``vol_target``
    a float, scale to that per-period volatility (the OOS `tanptfnext` form); with
    ``vol_target=None`` return the plain regression form (the in-sample `tanptf` form, no vol
    target). The vol target moves reported means, not the Sharpe ratio."""
    try:
        sinv_mu = np.linalg.solve(S, mu)
    except np.linalg.LinAlgError as exc:
        raise ContractViolation(f"singular factor covariance in tangency_weights (§1.4): {exc}") from exc
    w = sinv_mu / float(np.sum(sinv_mu))
    if float(w @ mu) < 0.0:
        w = -w
    if vol_target is None:
        return np.asarray(w, dtype=np.float64)
    scale = vol_target / np.sqrt(float(w @ S @ w))
    return np.asarray(w * scale, dtype=np.float64)


def tangency_strategy(rec: RecursiveResult, *, vol_target: float = 0.01) -> StratResult:
    """Recursive vol-targeted tangency on the factor portfolios (§4.1): at each t use the
    in-sample factor moments (μ_t, S_t) to form weights, realise on f̂_{t+1}."""
    J = rec.f_oos.shape[1]
    rets = np.empty(J)
    wpath = np.empty((J, rec.f_oos.shape[0]))
    for j in range(J):
        w = tangency_weights(rec.insample_mean[:, j], rec.insample_cov[j], vol_target=vol_target)
        wpath[j] = w
        rets[j] = float(w @ rec.f_oos[:, j])
    return StratResult(returns=rets, sharpe=sharpe(rets), weights=wpath, turnover=turnover(wpath))


def spread_strategy(
    Z: list[np.ndarray], R: list[np.ndarray], rec: RecursiveResult, *, quantiles: int = 5
) -> StratResult:
    """Recursive long-short spread strategy (§4.2): Êr_i = Z_i'Γ̂λ̂, sort into quantiles,
    equal-weighted long top / short bottom (each leg unit gross), monthly reconstitution.

    Z/R are the FULL panel; the OOS months in ``rec`` index into them by position
    (T − J .. T − 1)."""
    T = len(Z)
    J = rec.f_oos.shape[1]
    offset = T - J
    rets = np.empty(J)
    for j in range(J):
        m = offset + j
        Zm = np.asarray(Z[m], dtype=np.float64)
        rm = np.asarray(R[m], dtype=np.float64)
        exp_ret = Zm @ rec.gammas[j] @ rec.lambdas[:, j]
        order = np.argsort(exp_ret)
        n = order.size
        q = max(1, n // quantiles)
        short_leg = order[:q]
        long_leg = order[-q:]
        rets[j] = float(np.mean(rm[long_leg]) - np.mean(rm[short_leg]))
    return StratResult(returns=rets, sharpe=sharpe(rets), weights=None, turnover=None)


def tangency_insample(fit: IPCAFit) -> StratResult:
    """In-sample tangency on the factor portfolios (§4.1, plain `tanptf` form — NO vol target).

    Static weights from the full in-sample factor moments (μ, S with T−1 normalisation), realised
    on the in-sample factor path. Reported separately from the recursive OOS `tangency_strategy`.
    """
    F = fit.factors
    mu = F.mean(axis=1)
    S = np.atleast_2d(np.cov(F, ddof=1))
    w = tangency_weights(mu, S, vol_target=None)
    rets = np.asarray(w @ F, dtype=np.float64)
    return StratResult(returns=rets, sharpe=sharpe(rets), weights=w, turnover=None)


def smoothing_cost_curve(
    weights_path: np.ndarray, factor_path: np.ndarray, grid: list[float], *, cost_bp: float = 19.0
) -> dict[float, StratResult]:
    """Net-of-cost performance across the exponential-smoothing γ-grid (§4.3, eq.14 / Table VIII)
    for a weight-emitting strategy (e.g. the tangency leg).

    For each γ: smooth the weight path, realise ``w̃_t·f_t``, subtract the per-period 19 bp turnover
    drag, report the net Sharpe. ``weights_path`` is (J, K); ``factor_path`` is (K, J). At γ=0 the
    net return is gross minus the un-smoothed turnover cost. (The spread leg emits no weight path —
    that path is the deferred item; see docs/quant/registers/ipca_adjudications.md.)
    """
    wp = np.asarray(weights_path, dtype=np.float64)
    fp = np.asarray(factor_path, dtype=np.float64)
    out: dict[float, StratResult] = {}
    for g in grid:
        sm = smooth_weights(wp, float(g))
        gross = np.einsum("jk,kj->j", sm, fp)
        net = apply_costs(gross, sm, cost_bp=cost_bp)
        out[float(g)] = StratResult(returns=net, sharpe=sharpe(net), weights=sm, turnover=turnover(sm))
    return out


# ---------------------------------------------------------------------------
# Bootstrap inference (§3) — resample-then-multiply hybrid; R3: hold W[m] fixed
# ---------------------------------------------------------------------------


class BootResult(NamedTuple):
    statistic: float          # observed W (‖Γα‖² or Σ_k Γβ[ℓ,k]²)
    p_value: float            # one-sided p = mean(W_obs < W_boot), strict
    n_sims: int
    boot_stats: np.ndarray    # (n_sims,) bootstrap statistics
    magnitude: float          # √W (alpha) or RMS √(mean_k Γβ[ℓ,k]²) (characteristic)


MultiplierVariance = Literal["kpp_faithful", "unit"]


def _block_index(rng: np.random.Generator, T: int, block_len: int) -> np.ndarray:
    """Block-bootstrap residual time indices (§3): random-start consecutive runs of length
    ``block_len``, concatenated and truncated at T. block_len=1 is iid month resampling."""
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T - block_len + 1, size=n_blocks)
    blocks = [np.arange(s, s + block_len) for s in starts]
    return np.concatenate(blocks)[:T]


def _multipliers(
    rng: np.random.Generator, T: int, dof: int, variance: MultiplierVariance
) -> np.ndarray:
    """Per-period t(dof) multipliers (§3). ``kpp_faithful`` = raw t(7) (variance dof/(dof-2),
    inflated, conservative toward the null); ``unit`` = variance-normalised (÷√(dof/(dof-2)))."""
    eta = rng.standard_t(dof, size=T)
    if variance == "unit":
        eta = eta / np.sqrt(dof / (dof - 2))
    elif variance != "kpp_faithful":
        raise ValueError(f"multiplier_variance must be kpp_faithful | unit; got {variance!r}")
    return eta


def _pseudo_stats(stats: SuffStats, center: np.ndarray, resid: np.ndarray,
                  btix: np.ndarray, eta: np.ndarray) -> SuffStats:
    """Build a managed-portfolio pseudo-panel (§3, R3): x*[:,m] = center[:,m] + η[m]·resid[:,btix[m]].

    W[m] is held FIXED at period m (conditioning on the realised characteristic cross-section —
    the null constrains the pricing structure x|W, not the distribution that generates W). The
    residual is deliberately time-shifted by btix; W is NEVER indexed by btix.
    """
    x_star = center + resid[:, btix] * eta[None, :]
    pseudo = SuffStats(W=stats.W, x=x_star.T, N=stats.N, months=stats.months)
    assert pseudo.W is stats.W, "R3: W must stay indexed by m, never by btix"
    return pseudo


def bootstrap_alpha_test(
    stats: SuffStats,
    *,
    K: int,
    weighting: Weighting = "per_month_normalized",
    n_sims: int = 1000,
    dof: int = 7,
    block_len: int = 7,
    multiplier_variance: MultiplierVariance = "kpp_faithful",
    seed: int = 20260612,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> BootResult:
    """Bootstrap test of W_α = ‖Γ̂_α‖² (§3).

    Pseudo-data centre = RESTRICTED (Γα=0) fitted values; residuals = UNRESTRICTED-model
    residuals (the deliberate KPP deviation, logged); block length 7; raw t(7) multipliers
    under ``kpp_faithful``. Each pseudo-panel is re-estimated UNRESTRICTED, warm-started at the
    point estimate. One-sided strict p = mean(W_obs < W_boot).
    """
    T, L, _ = stats.W.shape
    fit_u = fit_ipca(stats, K=K, alpha=True, weighting=weighting, tol=tol, max_iter=max_iter)
    fit_r = fit_ipca(stats, K=K, alpha=False, weighting=weighting, tol=tol, max_iter=max_iter)
    ga_u = fit_u.gamma_alpha
    assert ga_u is not None
    w_obs = float(np.sum(ga_u ** 2))
    center = np.empty((L, T))
    resid = np.empty((L, T))
    for m in range(T):
        center[:, m] = stats.W[m] @ (fit_r.gamma_beta @ fit_r.factors[:, m])
        load_u = fit_u.gamma_beta @ fit_u.factors[:, m] + ga_u[:, 0]
        resid[:, m] = stats.x[m] - stats.W[m] @ load_u
    rng = np.random.default_rng(seed)
    boot = np.empty(n_sims)
    for b in range(n_sims):
        btix = _block_index(rng, T, block_len)
        eta = _multipliers(rng, T, dof, multiplier_variance)
        pseudo = _pseudo_stats(stats, center, resid, btix, eta)
        fb = fit_ipca(
            pseudo, K=K, alpha=True, weighting=weighting, tol=tol, max_iter=max_iter,
            gamma0=fit_u.gamma_beta, F0=fit_u.factors,
        )
        gb_a = fb.gamma_alpha
        assert gb_a is not None
        boot[b] = float(np.sum(gb_a ** 2))
    return BootResult(
        statistic=w_obs, p_value=float(np.mean(w_obs < boot)), n_sims=n_sims,
        boot_stats=boot, magnitude=float(np.sqrt(w_obs)),
    )


def bootstrap_characteristic_test(
    stats: SuffStats,
    *,
    K: int,
    rows: list[int] | None = None,
    weighting: Weighting = "per_month_normalized",
    n_sims: int = 1000,
    dof: int = 7,
    block_len: int = 1,
    multiplier_variance: MultiplierVariance = "unit",
    seed: int = 20260612,
    tol: float = 1e-4,
    max_iter: int = 1000,
) -> dict[int, BootResult]:
    """Bootstrap test of W_{β,ℓ} = Σ_k Γ̂_β[ℓ,k]² for each instrument row ℓ (§3).

    Pseudo-data centre = full-model fit with row ℓ of Γ̂_β zeroed (full-model factors);
    residuals = full-model residuals; block length 1 (iid); unit-variance-normalised
    multipliers. Re-estimate the full restricted model per draw. Reported magnitude is the RMS
    √(mean_k Γ̂_β[ℓ,k]²) (monotone in W; p unaffected)."""
    T, L, _ = stats.W.shape
    fit = fit_ipca(stats, K=K, alpha=False, weighting=weighting, tol=tol, max_iter=max_iter)
    row_list = list(range(L)) if rows is None else rows
    resid = np.empty((L, T))
    for m in range(T):
        resid[:, m] = stats.x[m] - stats.W[m] @ (fit.gamma_beta @ fit.factors[:, m])
    results: dict[int, BootResult] = {}
    for ell in row_list:
        w_obs = float(np.sum(fit.gamma_beta[ell] ** 2))
        gamma_tilde = fit.gamma_beta.copy()
        gamma_tilde[ell] = 0.0
        center = np.empty((L, T))
        for m in range(T):
            center[:, m] = stats.W[m] @ (gamma_tilde @ fit.factors[:, m])
        rng = np.random.default_rng(seed + ell)
        boot = np.empty(n_sims)
        for b in range(n_sims):
            btix = _block_index(rng, T, block_len)
            eta = _multipliers(rng, T, dof, multiplier_variance)
            pseudo = _pseudo_stats(stats, center, resid, btix, eta)
            fb = fit_ipca(
                pseudo, K=K, alpha=False, weighting=weighting, tol=tol, max_iter=max_iter,
                gamma0=fit.gamma_beta, F0=fit.factors,
            )
            boot[b] = float(np.sum(fb.gamma_beta[ell] ** 2))
        results[ell] = BootResult(
            statistic=w_obs, p_value=float(np.mean(w_obs < boot)), n_sims=n_sims,
            boot_stats=boot, magnitude=float(np.sqrt(np.mean(fit.gamma_beta[ell] ** 2))),
        )
    return results


# ---------------------------------------------------------------------------
# Gold-spec config loader (§8) + provenance (§10)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentificationConfig:
    orthonormal_gamma: bool
    descending_factor_moments: bool
    nonneg_factor_means: bool
    applied_per_iteration: bool
    sign_eigvec_rule: str
    sign_fallback_mean_eps: float
    near_tie_moment_eps: float


@dataclass(frozen=True)
class ModelConfig:
    K: int
    alpha: bool

    def __post_init__(self) -> None:
        if self.K < 1:
            raise ValueError(f"model.K must be >= 1; got {self.K}")


@dataclass(frozen=True)
class EstimationConfig:
    mode: str
    month_weighting: str
    tolerance: float
    max_iter_in_sample: int
    max_iter_recursive: int
    convergence_metric: str
    init: str
    warm_start_recursion: bool
    identification: IdentificationConfig

    def __post_init__(self) -> None:
        if self.mode not in ("in_sample", "recursive"):
            raise ValueError(f"estimation.mode must be in_sample | recursive; got {self.mode!r}")
        if self.month_weighting not in ("per_month_normalized", "observation_weighted"):
            raise ValueError(f"estimation.month_weighting invalid: {self.month_weighting!r}")
        if self.tolerance <= 0:
            raise ValueError("estimation.tolerance must be > 0")


@dataclass(frozen=True)
class BootstrapConfig:
    sims: int
    dof: int
    alpha_block_len: int
    beta_block_len: int
    multiplier_variance: str
    alpha_residual_source: str
    seed: int

    def __post_init__(self) -> None:
        if self.multiplier_variance not in ("kpp_faithful", "unit"):
            raise ValueError(f"bootstrap.multiplier_variance invalid: {self.multiplier_variance!r}")
        if self.dof <= 2:
            raise ValueError("bootstrap.dof must be > 2 (finite t variance)")


@dataclass(frozen=True)
class DataContractConfig:
    L: int
    rank_map: str
    demean_after_map: bool
    characteristic_lag_months: int
    family: str

    def __post_init__(self) -> None:
        if self.family not in ("raw", "corr"):
            raise ValueError(f"data_contract.family must be raw | corr; got {self.family!r}")
        if self.characteristic_lag_months != 1:
            raise ValueError("data_contract.characteristic_lag_months must be 1 (next-return convention)")


@dataclass(frozen=True)
class ScalingConfig:
    lane: str
    dts_constant: float
    dts_floor: float
    spread_validity_bp: list[float]
    duration_floor_years: float
    vol_floor: float

    def __post_init__(self) -> None:
        if self.lane not in ("DTSScaled25", "VOLScaled010", "Unscaled"):
            raise ValueError(f"scaling.lane invalid: {self.lane!r}")


@dataclass(frozen=True)
class StrategiesConfig:
    tangency: dict[str, Any]
    spread: dict[str, Any]
    costs_bp_oneway: float
    smoothing_gamma_grid: list[float]


@dataclass(frozen=True)
class WindowConfig:
    name: str
    oos_burn_in_months: int


@dataclass(frozen=True)
class IPCAConfig:
    """Typed view of ``kpp_ipca.yaml`` (§8). Mirrors ``run_config.py``: frozen dataclasses,
    ``from_dict``/``from_yaml``/``from_path``, ``to_yaml``/``hash()`` for §10 provenance.

    Location-agnostic: ``from_path`` takes any path, so the gold spec can live under
    ``agents/quant/library/configs/`` (default) or elsewhere without code change.
    """

    model: ModelConfig
    estimation: EstimationConfig
    bootstrap: BootstrapConfig
    data_contract: DataContractConfig
    scaling: ScalingConfig
    strategies: StrategiesConfig
    window: WindowConfig
    context_source: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IPCAConfig:
        try:
            ident = IdentificationConfig(**data["estimation"]["identification"])
            est = {k: v for k, v in data["estimation"].items() if k != "identification"}
            return cls(
                model=ModelConfig(**data["model"]),
                estimation=EstimationConfig(identification=ident, **est),
                bootstrap=BootstrapConfig(**data["bootstrap"]),
                data_contract=DataContractConfig(**data["data_contract"]),
                scaling=ScalingConfig(**data["scaling"]),
                strategies=StrategiesConfig(**data["strategies"]),
                window=WindowConfig(**data["window"]),
                context_source=str(data["context_values"]["source"]),
            )
        except KeyError as exc:
            raise KeyError(f"kpp_ipca config missing required key: {exc}") from None

    @classmethod
    def from_yaml(cls, text: str) -> IPCAConfig:
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_path(cls, path: str | Path) -> IPCAConfig:
        return cls.from_yaml(Path(path).read_text())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["context_values"] = {"source": d.pop("context_source")}
        return d

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)

    def hash(self) -> str:
        """Stable SHA-256 over the canonical resolved YAML (the §10 config hash)."""
        return hashlib.sha256(self.to_yaml().encode("utf-8")).hexdigest()


def build_run_log(
    config: IPCAConfig,
    *,
    fits: list[IPCAFit],
    train_end: int,
    window_bounds: tuple[int, int],
    dropped_months: list[int],
    below_floor_shares: dict[str, float],
    lib_versions: dict[str, str],
    comparability_label: str | None = None,
) -> dict[str, Any]:
    """Assemble the §10 run record: config hash + resolved YAML, seed, window + wall bounds,
    per-estimation iteration counts and warm/cold flags, below-floor scaling shares, the
    received-and-re-logged dropped-month list, K22 arm, bootstrap arm, library versions.

    ``comparability_label`` stamps non-comparable interface-validation runs (WS-B shakedown).
    """
    log: dict[str, Any] = {
        "config_hash": config.hash(),
        "resolved_config": config.to_dict(),
        "bootstrap_seed": config.bootstrap.seed,
        "window_name": config.window.name,
        "window_bounds": list(window_bounds),
        "train_end": train_end,
        "month_weighting_arm": config.estimation.month_weighting,   # K22 arm
        "bootstrap_multiplier_arm": config.bootstrap.multiplier_variance,
        "demean_after_map": config.data_contract.demean_after_map,
        "estimations": [
            {"n_iter": f.n_iter, "converged": f.converged, "init_kind": f.init_kind,
             "tol_final": f.tol_final, "weighting": f.weighting}
            for f in fits
        ],
        "below_floor_shares": below_floor_shares,
        "dropped_months": dropped_months,
        "lib_versions": lib_versions,
    }
    if comparability_label is not None:
        log["comparability_label"] = comparability_label
    return log


def context_table(
    computed: dict[str, float],
    context_values: dict[str, float],
    *,
    comparability_label: str | None = None,
) -> str:
    """Render the §10.4 context-value comparison as markdown (§10 acceptance harness).

    ``computed`` = the run's metrics; ``context_values`` = the published reference numbers (KPP
    §10.4 — placeholders until ``kelly_2023_ipca.md`` is authored; NOT gates, POLICY D1). A missing
    context entry renders ``—``. ``comparability_label`` stamps non-comparable runs at the top so the
    table can never be mistaken for a like-for-like target. Built against synthetic placeholders
    per §10; the WS-B shakedown calls this on real (non-comparable) data.
    """
    lines: list[str] = []
    if comparability_label is not None:
        lines += [f"**{comparability_label}**", ""]
    lines += ["| metric | computed | context (§10.4) |", "|--------|----------|-----------------|"]
    for name, value in computed.items():
        ctx = context_values.get(name)
        lines.append(f"| {name} | {value} | {'—' if ctx is None else ctx} |")
    return "\n".join(lines) + "\n"
