"""Shared synthetic data-generating processes + recovery metrics for RQ3 instrument validation.

These are the EXACT planted-truth generators and recovery metrics that certify the recovery of a
known signal — extracted VERBATIM from their originating test batteries so a results EXPORTER
(``scripts/run_rq3_validation_export.py``) and the tests share one source of truth. The tests
import these back under their original names, so the tests staying green is proof the extraction is
byte-equivalent.

Sources:
  * ``build_quality_dgp`` — from ``tests/unit/test_characteristic_sort_recovery.py`` (bond-quality
    → next-month return; the characteristic-sort planted long-short alpha DGP).
  * ``make_ipca_panel`` / ``max_principal_angle_deg`` / ``factor_alignment_r2`` — from
    ``tests/synthetic/test_ipca_battery.py`` (the KPP IPCA certification DGP + subspace/factor
    recovery metrics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agents.quant.library.ipca import rank_transform


# ---------------------------------------------------------------------------
# Characteristic-sort DGP: bond quality drives next-month return.
# ---------------------------------------------------------------------------

def build_quality_dgp(
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
# IPCA DGP + recovery metrics (KPP certification battery).
# ---------------------------------------------------------------------------

def make_ipca_panel(
    rng: np.random.Generator,
    *,
    L: int = 30,
    K: int = 3,
    T: int = 264,
    n_range: tuple[int, int] = (800, 3000),
    snr: float = 10.0,
    gamma_alpha: np.ndarray | None = None,
    distinct_moments: bool = True,
):
    """Simulate a panel from the IPCA model. L counts the constant; L-1 instruments.

    snr = var(signal)/var(per-bond noise). Returns (Z_list, R_list, months, truth).
    """
    n_inst = L - 1
    Qg, _ = np.linalg.qr(rng.standard_normal((L, K)))
    gamma_beta = Qg[:, :K]
    var_f = np.linspace(K, 1.0, K) if distinct_moments else np.ones(K)
    mu_f = 0.3 * np.sqrt(var_f)
    months = np.arange(1, T + 1)
    asof = months - 1
    Z_list: list[np.ndarray] = []
    R_list: list[np.ndarray] = []
    raw_list: list[np.ndarray] = []
    F_true = np.empty((K, T))
    for m in range(T):
        n = int(rng.integers(n_range[0], n_range[1] + 1))
        raw = rng.standard_normal((n, n_inst))
        ranked = np.column_stack([rank_transform(raw[:, j]) for j in range(n_inst)])
        Z = np.column_stack([ranked, np.ones(n)])
        f = mu_f + np.sqrt(var_f) * rng.standard_normal(K)
        F_true[:, m] = f
        signal = Z @ gamma_beta @ f
        if gamma_alpha is not None:
            signal = signal + Z @ gamma_alpha[:, 0]
        sig_sd = float(signal.std())
        eps_sd = (sig_sd if sig_sd > 0 else 1.0) / np.sqrt(snr)
        R = signal + eps_sd * rng.standard_normal(n)
        Z_list.append(Z)
        R_list.append(R)
        raw_list.append(raw)
    truth = {
        "gamma_beta": gamma_beta, "F_true": F_true, "var_f": var_f,
        "raw": raw_list, "months": months, "asof": asof,
    }
    return Z_list, R_list, months, truth


def max_principal_angle_deg(B1: np.ndarray, B2: np.ndarray) -> float:
    Q1, _ = np.linalg.qr(B1)
    Q2, _ = np.linalg.qr(B2)
    s = np.clip(np.linalg.svd(Q1.T @ Q2, compute_uv=False), -1.0, 1.0)
    return float(np.degrees(np.arccos(s.min())))


def factor_alignment_r2(F_hat: np.ndarray, F_true: np.ndarray) -> float:
    T = F_hat.shape[1]
    X = np.column_stack([np.ones(T), F_hat.T])
    beta, *_ = np.linalg.lstsq(X, F_true.T, rcond=None)
    pred = X @ beta
    ss_res = ((F_true.T - pred) ** 2).sum(axis=0)
    ss_tot = ((F_true.T - F_true.T.mean(axis=0)) ** 2).sum(axis=0)
    return float((1.0 - ss_res / ss_tot).min())
