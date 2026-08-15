"""G3 — development inference (spec §9). The candidate's primary test: benchmark alpha vs BBW-4
with NW-HAC (reuse `regress_on_benchmark`), a two-sided p from the HAC t-stat, and BH-FDR WITHIN
the family (reuse the wrapped `run_fdr`). Every valid proposal in the family is counted (§9). G3
always runs — it computes performance and sets `bh_survived`; a non-survivor is the terminal
NO_DEVELOPMENT_EVIDENCE outcome, not a refusal. Populates the gross measurements block.
"""

from __future__ import annotations

from statistics import NormalDist

import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark, summarize_returns

from shared.stats import run_fdr

from ..schemas.evaluation import GrossMeasurements
from ..schemas.outcomes import RefusalCode  # noqa: F401 (documents G3 has no refusal code)
from .gate_result import GateOutcome

_NORMAL = NormalDist()


def two_sided_p(t_stat: float) -> float:
    """Two-sided p-value from a t-stat (asymptotic normal — a NW-HAC t is asymptotically N(0,1))."""
    if t_stat != t_stat:                                     # NaN
        return float("nan")
    return 2.0 * (1.0 - _NORMAL.cdf(abs(t_stat)))


def inference_g3(
    candidate_returns: pd.Series,
    bbw4_factors: pd.DataFrame,
    *,
    proposal_id: str,
    family_pvalues: dict | None = None,
    q: float = 0.10,
    nw_lags: int | None = None,
    months_per_year: int = 12,
    direction: int = 1,
):
    """Return (GateOutcome, GrossMeasurements). `family_pvalues` are the OTHER valid proposals'
    raw p-values (this proposal is added to the family for the joint BH-FDR).

    SC-SCI-8 / D-Q17 — `bh_survived` is SIGN-AWARE: the two-sided BH rejection (`bh_rejected`) AND
    the alpha in the strategy's GATING DIRECTION (`direction`, DERIVED from the realised parent
    premium sign, not the paper's claim; +1, or -1 for a genuinely negative-premium parent). The
    two-sided test and BH family are unchanged; the survivor LABEL is narrowed to correctly-signed
    rejections (strictly conservative), consistent with G4's directional CPCV gate. A
    rejected-but-wrong-signed extension (it reliably WORSENED the strategy) is bh_rejected=True,
    bh_survived=False. (str is NOT a negative-premium case: it realises +momentum on the corrected
    dev panel; DRR's -0.99 reversal is a published claim, so str's derived direction is +1.)"""
    reg = regress_on_benchmark(candidate_returns, bbw4_factors, nw_lags)
    alpha, alpha_t = reg["alpha"], reg["alpha_t"]
    p_raw = two_sided_p(alpha_t)
    summ = summarize_returns(candidate_returns, nw_lags, months_per_year)

    family = dict(family_pvalues or {})
    family[proposal_id] = p_raw                              # all valid proposals counted (§9)
    report = run_fdr(family, q, scope="within_strategy")
    decision = report.decisions[proposal_id]
    bh_survived = decision.rejected and (direction * alpha > 0.0)

    gross = GrossMeasurements(
        mean_return=summ["average"], sharpe=summ["sharpe"], alpha_bbw4=alpha,
        t_stat=alpha_t, p_raw=p_raw, p_bh=decision.adjusted_p, bh_rejected=decision.rejected,
    )
    return (GateOutcome("G3", passed=bh_survived, booleans={"bh_survived": bh_survived}), gross)
