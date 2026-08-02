"""G4 — robustness (spec §9/§10). Runs POST-BH-FDR (on G3 survivors only). Reuses the wrapped
stats + the Layer-1 crowding diagnostic; the audited engine / library are untouched.

  * CPCV (shared/stats) over the candidate's returns — purge = the per-candidate information span,
    embargo = holding_period months (F8/A4). The gate boolean `cpcv_qualified` is the committed
    G4 criterion: a positive MEDIAN OOS-fold Sharpe (the informative statistic for a fit-free
    candidate — see cpcv.py).
  * Deflated Sharpe (n_trials = the strategy's m) and the Layer-1 crowding diagnostic (BBW-4 +
    str + mom6) are MEASUREMENTS reported beside the gate, never gating.

Populates the measurements block; sets `cpcv_qualified`.
"""

from __future__ import annotations

import math

import pandas as pd

from agents.quant.library.characteristic_sort import summarize_returns

from shared.evaluation.crowding import crowding_diagnostic
from shared.evaluation.crowding_l2 import ipca_crowding_diagnostic
from shared.evaluation.thresholds import load_crowding_config
from shared.stats import cpcv_evaluate, deflated_sharpe_ratio

from ..schemas.evaluation import Measurements
from .gate_result import GateOutcome


def robustness_g4(
    candidate_returns: pd.Series,
    gross,
    *,
    n_trials: int,
    information_span: int,
    holding_period: int,
    sr_std: float,
    months_per_year: int = 12,
    nw_lags: int | None = None,
    n_groups: int = 8,
    test_groups: int = 2,
    crowding_config=None,
    crowding_factors=None,
    ipca_factors=None,
):
    embargo = max(1, holding_period)
    cpcv = cpcv_evaluate(candidate_returns, n_groups=n_groups, test_groups=test_groups,
                         purge=information_span, embargo=embargo, months_per_year=months_per_year)
    cpcv_qualified = math.isfinite(cpcv.median_sharpe) and cpcv.median_sharpe > 0.0

    # Deflated Sharpe — per-period Sharpe deflated against the m-trial max (measurement only).
    summ = summarize_returns(candidate_returns, nw_lags, months_per_year)
    sr = summ["sharpe"]
    if math.isfinite(sr) and summ["n_months"] >= 2:
        dsr = deflated_sharpe_ratio(sr / math.sqrt(months_per_year), summ["n_months"],
                                    n_trials, sr_std=sr_std)
    else:
        dsr = float("nan")

    # Layer-1 crowding (measurement only). Real run loads the factor bundle from the config;
    # tests inject `crowding_factors`.
    config = crowding_config if crowding_config is not None else load_crowding_config()
    crowd = dict(crowding_diagnostic(candidate_returns, config=config, factors=crowding_factors))
    # Layer 2 (recursive-OOS IPCA spanning) — a SECOND lens, merged in beside Layer 1 when the
    # recursive-OOS IPCA factor frame is supplied (measurement only, never gating).
    if ipca_factors is not None:
        crowd.update(ipca_crowding_diagnostic(candidate_returns, ipca_factors, nw_lags=nw_lags))

    measurements = Measurements(gross=gross, cpcv=cpcv.to_dict(), deflated_sharpe=dsr, crowding=crowd)
    return (GateOutcome("G4", passed=cpcv_qualified, booleans={"cpcv_qualified": cpcv_qualified}),
            measurements)
