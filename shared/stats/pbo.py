"""
shared/stats/pbo.py — CSCV probability of backtest overfitting (Bailey & López de Prado),
net-new (WS-D).

PURPOSE. Net-new for the P4 optimisation-overfitting workstream (WS-D). Measurement-only:
PBO is reported beside whatever selection the caller has already made and never gates,
ranks, or repairs anything.

GEOMETRY + CONVENTION. Folds come from `cpcv_folds` — the project's CPCV geometry, reused
verbatim — and every Sharpe comes from `summarize_returns` (ddof=1, mean/sd *
sqrt(months_per_year)), the SAME functional the Auditor and CPCV stats use. No new
convention. With the defaults purge=0 and embargo=0 the fold set is the textbook CSCV of
Bailey et al.; the P4 runner passes the scientist-protocol purge/embargo values, so the
procedure is copied from the registered protocol, not re-chosen here.

DEFINITION. Per fold: rank all candidates by in-sample (train) Sharpe and take the IS-best
(ties broken by first column order); its relative rank among the out-of-sample (test)
Sharpes is omega = rank / (N + 1) (average rank, ascending — the OOS-best has
omega = N/(N+1)), with logit lambda = ln(omega / (1 - omega)). PBO is the fraction of
folds with lambda < 0: how often the in-sample winner lands in the bottom half OOS.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns
from shared.stats.cpcv import cpcv_folds


@dataclass(frozen=True)
class PBOResult:
    n_candidates: int
    n_obs: int
    n_folds: int                       # C(n_groups, test_groups)
    n_groups: int
    test_groups: int
    purge: int
    embargo: int
    lambdas: tuple[float, ...]         # one logit per fold, in cpcv_folds order
    pbo: float                         # fraction of folds with lambda < 0
    median_oos_relative_rank: float    # median omega of the IS-best across folds
    is_best_counts: dict[str, int]     # candidate id -> times selected IS-best

    def to_dict(self) -> dict:
        return {
            "n_candidates": self.n_candidates,
            "n_obs": self.n_obs,
            "n_folds": self.n_folds,
            "n_groups": self.n_groups,
            "test_groups": self.test_groups,
            "purge": self.purge,
            "embargo": self.embargo,
            "pbo": self.pbo,
            "median_oos_relative_rank": self.median_oos_relative_rank,
            "is_best_counts": dict(self.is_best_counts),
        }


def _sharpe_vector(
    values: np.ndarray,
    rows: np.ndarray,
    label: str,
    fold_no: int,
    candidate_ids: list[str],
    months_per_year: int,
) -> np.ndarray:
    """Per-candidate Sharpe over the given rows via `summarize_returns` (positional
    Series — the functional only uses the index for date reporting, so a RangeIndex is
    fine). Fail-loud on any candidate whose slice cannot carry a rank."""
    out = np.empty(len(candidate_ids))
    for j, cid in enumerate(candidate_ids):
        col = values[rows, j]
        # A constant slice has sd = 0 in exact arithmetic, but float rounding of the
        # slice mean can leave sd ~ 1e-18 and an astronomical yet FINITE Sharpe, which
        # the isfinite check below would wave through. ptp == 0 is the exact test for
        # that degeneracy.
        if col.size and np.ptp(col) == 0:
            raise ValueError(
                f"degenerate {label} Sharpe for candidate '{cid}' in fold {fold_no}: "
                f"slice is constant (sd = 0 in exact arithmetic) — PBO ranks are undefined"
            )
        summ = summarize_returns(pd.Series(col), nw_lags=None, months_per_year=months_per_year)
        sharpe = float(summ["sharpe"])
        if not np.isfinite(sharpe):
            raise ValueError(
                f"non-finite {label} Sharpe for candidate '{cid}' in fold {fold_no} — "
                f"PBO ranks are undefined"
            )
        out[j] = sharpe
    return out


def pbo_cscv(
    returns_matrix: pd.DataFrame,
    *,
    n_groups: int = 8,
    test_groups: int = 2,
    purge: int = 0,
    embargo: int = 0,
    months_per_year: int = 12,
) -> PBOResult:
    """CSCV probability of backtest overfitting for a T x N candidate-return matrix
    (rows = months in time order, columns = candidate ids).

    Per (train, test) fold from `cpcv_folds`: the IS-best candidate by train Sharpe (ties
    broken by first column order) is ranked among the test Sharpes; omega = rank/(N+1)
    and lambda = ln(omega/(1-omega)). PBO = fraction of folds with lambda < 0.

    Fail-loud (ValueError, naming the offender) on: fewer than two candidates, any NaN
    cell, or any IS/OOS slice whose Sharpe cannot carry a rank — a constant slice (sd = 0
    in exact arithmetic) or an otherwise non-finite Sharpe. Pure function — no I/O, no
    randomness.
    """
    n_obs, n_candidates = returns_matrix.shape
    if n_candidates < 2:
        raise ValueError(f"PBO needs >= 2 candidate columns to rank; got {n_candidates}")

    # NaN cells would be silently dropped inside summarize_returns, desynchronising the
    # train/test slices across candidates — refuse them outright.
    has_nan = returns_matrix.isna().any(axis=0)
    if bool(has_nan.any()):
        offenders = [str(c) for c in returns_matrix.columns[has_nan]]
        raise ValueError(f"returns_matrix contains NaN cells in candidate(s) {offenders}")

    candidate_ids = [str(c) for c in returns_matrix.columns]
    values = returns_matrix.to_numpy(dtype=float)
    folds = cpcv_folds(
        n_obs, n_groups=n_groups, test_groups=test_groups, purge=purge, embargo=embargo
    )

    lambdas: list[float] = []
    omegas: list[float] = []
    is_best_counts: dict[str, int] = {cid: 0 for cid in candidate_ids}
    for fold_no, (train, test) in enumerate(folds):
        is_sharpes = _sharpe_vector(values, train, "IS", fold_no, candidate_ids, months_per_year)
        oos_sharpes = _sharpe_vector(values, test, "OOS", fold_no, candidate_ids, months_per_year)

        best = int(np.argmax(is_sharpes))  # argmax ties resolve to the first column
        is_best_counts[candidate_ids[best]] += 1
        ranks = pd.Series(oos_sharpes).rank(method="average")  # ascending: OOS-best = N
        omega = float(ranks.iloc[best]) / (n_candidates + 1)
        omegas.append(omega)
        lambdas.append(log(omega / (1.0 - omega)))

    lam = np.array(lambdas)
    return PBOResult(
        n_candidates=n_candidates,
        n_obs=n_obs,
        n_folds=len(folds),
        n_groups=n_groups,
        test_groups=test_groups,
        purge=purge,
        embargo=embargo,
        lambdas=tuple(lambdas),
        pbo=float((lam < 0).mean()),
        median_oos_relative_rank=float(np.median(omegas)),
        is_best_counts=is_best_counts,
    )
