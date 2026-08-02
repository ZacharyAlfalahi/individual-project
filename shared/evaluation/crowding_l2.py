"""Layer-2 crowding — recursive-OOS IPCA factor spanning.

Build recursive OUT-OF-SAMPLE IPCA factors (fit IPCA on data through t, predict t+1 — no
look-ahead by construction; the instruments are rank-transformed per month, so nothing crosses the
window boundary), then span a candidate's returns by them. This is a materially stronger crowding
lens than Layer-1 (BBW-4 + str + mom6): if the recursive-OOS IPCA factors absorb the candidate's
alpha, the "survivor" is really the model's own factor. The IN-SAMPLE IPCA fit stays ruled out
(circularity) — only the recursive OOS path is used.

Reuses the audited IPCA module (`build_sufficient_stats`, `fit_ipca_recursive`) and the audited
`regress_on_benchmark`; nothing under agents/quant/library/ is modified.
"""

from __future__ import annotations

import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark
from agents.quant.library.ipca import build_sufficient_stats, fit_ipca_recursive


def recursive_ipca_factors(
    Z: list, R: list, months, *, K: int, burn_in: int = 36, month_to_date: dict | None = None,
) -> pd.DataFrame:
    """Build the recursive-OOS IPCA factor frame: `date` + one column per factor (ipca_1..ipca_K),
    one row per OOS prediction month. `Z`/`R` are the per-month instrument matrices and return
    vectors; `month_to_date` maps the integer month ids to real dates for the join (defaults to the
    ids). Every row is a t+1 OOS realisation of a Γ̂ estimated only on data through t."""
    stats = build_sufficient_stats(Z, R, months)
    rec = fit_ipca_recursive(stats, K=K, burn_in=burn_in)
    f = rec.f_oos.T                                          # (J, K)
    dates = ([month_to_date[int(m)] for m in rec.oos_months]
             if month_to_date is not None else list(rec.oos_months))
    frame = pd.DataFrame({"date": dates})
    for k in range(K):
        frame[f"ipca_{k + 1}"] = f[:, k]
    return frame


def ipca_crowding_diagnostic(
    candidate_returns: pd.Series, ipca_factors: pd.DataFrame, *, nw_lags: int | None = None,
) -> dict:
    """Span the candidate by the recursive-OOS IPCA factors (spanning regression). A small,
    insignificant `ipca_alpha` means the candidate is absorbed by — crowded into — the IPCA
    factors; a surviving `ipca_alpha` means it is not. Measurement only, never gating."""
    reg = regress_on_benchmark(candidate_returns, ipca_factors, nw_lags)
    return {
        "ipca_alpha": reg["alpha"],
        "ipca_alpha_t": reg["alpha_t"],
        "ipca_n_obs": float(reg["n_obs"]),
    }
