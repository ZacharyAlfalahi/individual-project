"""The Experimentalist orchestrator — runs a ScientistCase's proposals through the gate stack
G0 -> G5 and emits one EvaluationRecord per proposal (final_outcome DERIVED from the twelve
booleans). A failed gate stops the proposal and no further performance is computed (prohibition 9).

G3's BH-FDR is JOINT across the family (spec §9: "all valid proposals counted"), so it is a batch
over ALL audit-clean proposals, not a per-proposal step: G0->G2 filter first, then one BH-FDR over
the survivors' raw p-values, then G4 robustness on the BH survivors, then G5 lexicographic
selection of up to `cap`. G6 (holdout) is a separate one-time access, not run here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from agents.quant.library.characteristic_sort import regress_on_benchmark, summarize_returns

from shared.stats import run_fdr

from ..schemas.equivalence import equivalence_key
from ..schemas.evaluation import (
    BOOLEAN_FIELDS,
    Booleans,
    EvaluationRecord,
    GrossMeasurements,
    Measurements,
)
from .audit_checks import audit_g2, load_reporting_delays
from .compiler import compile_g1a
from .execution_verifier import execute_g1b
from .inference import two_sided_p
from .robustness import robustness_g4
from .selector import select_g5
from .validator import validate_g0


@dataclass(frozen=True)
class ExperimentReport:
    records: tuple                       # EvaluationRecord per proposal (input order)
    advanced: tuple                      # proposal_ids advanced to holdout (G5)
    funnel: dict                         # final_outcome -> count (denominators visible, §12)
    wrong_signed: int = 0                # SC-SCI-8: BH-rejected but wrong-signed (worsened) — a
                                         #   reported RQ4 finding, NOT a separate outcome label


def _booleans(g0_bools=None, **over) -> Booleans:
    d = {f: False for f in BOOLEAN_FIELDS}
    if g0_bools:
        d.update(g0_bools)
    d.update(over)
    return Booleans(**d)


def _info_span(compiled, signal_lookback: int, holding_period: int) -> int:
    lag = compiled.panel_transform.lag_months if compiled.panel_transform else 0
    return signal_lookback + holding_period + lag


def run_experimentalist(
    case,
    proposals,
    library,
    *,
    panel: pd.DataFrame,
    base_rulebook: dict,
    bbw4_factors: pd.DataFrame,
    holding_period: int,
    signal_lookback: int,
    available_variables,
    macro=None,
    m: int = 6,
    sr_std: float = 0.5,
    q: float = 0.10,
    cap: int = 3,
    direction: int = 1,               # strategy's CLAIMED premium sign (+1; -1 for str reversal)
    crowding_config=None,
    crowding_factors=None,
    reporting_delays=None,
    nw_lags: int | None = None,
    months_per_year: int = 12,
) -> ExperimentReport:
    reporting_delays = reporting_delays if reporting_delays is not None else load_reporting_delays()
    seen: set = set()
    stored: dict = {}                    # proposal_id -> (Booleans, refusal_code, Measurements)
    audit_clean: list = []               # (proposal, candidate_returns, compiled, g0_bools)

    # ---- Phase A: G0 -> G1a -> G1b -> G2 (per proposal; stop at first failure) --------------
    for p in proposals:
        g0 = validate_g0(p, case, library, seen_keys=seen, available_variables=available_variables)
        if not g0.passed:
            stored[p.proposal_id] = (_booleans(g0.booleans), g0.refusal_code, Measurements())
            continue
        seen.add(equivalence_key(p))
        template = library.templates[p.template_ref]
        g1a, compiled = compile_g1a(p, template, case, holding_period=holding_period,
                                    available_variables=available_variables)
        if not g1a.passed:
            stored[p.proposal_id] = (_booleans(g0.booleans, compiled=False), g1a.refusal_code, Measurements())
            continue
        g1b, exec_res = execute_g1b(compiled, panel, base_rulebook, macro=macro)
        if not g1b.passed:
            stored[p.proposal_id] = (_booleans(g0.booleans, compiled=True, execution_verified=False),
                                     g1b.refusal_code, Measurements())
            continue
        g2 = audit_g2(compiled, reporting_delays=reporting_delays)
        if not g2.passed:
            stored[p.proposal_id] = (_booleans(g0.booleans, compiled=True, execution_verified=True,
                                               audit_clean=False), g2.refusal_code, Measurements())
            continue
        audit_clean.append((p, exec_res.candidate_returns, compiled, dict(g0.booleans)))

    # ---- Phase B: G3 JOINT BH-FDR over all audit-clean proposals ----------------------------
    reg_by_id: dict = {}
    family: dict = {}
    for p, cand, compiled, _ in audit_clean:
        reg = regress_on_benchmark(cand, bbw4_factors, nw_lags)
        p_raw = two_sided_p(reg["alpha_t"])
        family[p.proposal_id] = p_raw
        reg_by_id[p.proposal_id] = reg
    fdr = run_fdr(family, q) if family else None

    # ---- Phase C: G3 outcome + G4 robustness (BH survivors only) ----------------------------
    survivors: list = []
    wrong_signed = 0                                  # SC-SCI-8: rejected but in the WRONG direction
    for p, cand, compiled, g0_bools in audit_clean:
        reg = reg_by_id[p.proposal_id]
        decision = fdr.decisions[p.proposal_id]
        summ = summarize_returns(cand, nw_lags, months_per_year)
        # SC-SCI-8 — sign-aware survivor: two-sided BH rejection AND alpha in the claimed direction.
        bh_survived = decision.rejected and (direction * reg["alpha"] > 0.0)
        gross = GrossMeasurements(mean_return=summ["average"], sharpe=summ["sharpe"],
                                  alpha_bbw4=reg["alpha"], t_stat=reg["alpha_t"],
                                  p_raw=family[p.proposal_id], p_bh=decision.adjusted_p,
                                  bh_rejected=decision.rejected)
        if not bh_survived:                           # NO_DEVELOPMENT_EVIDENCE
            if decision.rejected:                     # rejected but wrong sign = reliably WORSENED
                wrong_signed += 1
            stored[p.proposal_id] = (
                _booleans(g0_bools, compiled=True, execution_verified=True, audit_clean=True,
                          bh_survived=False), None, Measurements(gross=gross))
            continue
        try:
            g4, meas = robustness_g4(
                cand, gross, n_trials=m,
                information_span=_info_span(compiled, signal_lookback, holding_period),
                holding_period=holding_period, sr_std=sr_std, direction=direction,
                crowding_config=crowding_config, crowding_factors=crowding_factors,
                months_per_year=months_per_year, nw_lags=nw_lags)
            cpcv_q = g4.booleans["cpcv_qualified"]
        except Exception:                              # per-proposal isolation: one cannot crash the batch
            cpcv_q, meas = False, Measurements(gross=gross)
        stored[p.proposal_id] = (
            _booleans(g0_bools, compiled=True, execution_verified=True, audit_clean=True,
                      bh_survived=True, cpcv_qualified=cpcv_q), None, meas)
        if cpcv_q:
            survivors.append({"proposal_id": p.proposal_id,
                              "median_cpcv_sharpe": meas.cpcv.get("median_sharpe"), "n_changes": 1})

    # ---- Phase D: G5 lexicographic selection ------------------------------------------------
    advanced = select_g5(survivors, cap=cap)

    records = tuple(
        EvaluationRecord(proposal_id=p.proposal_id, booleans=stored[p.proposal_id][0],
                         refusal_code=stored[p.proposal_id][1], measurements=stored[p.proposal_id][2])
        for p in proposals
    )
    funnel = dict(Counter(r.final_outcome.value for r in records))
    return ExperimentReport(records=records, advanced=tuple(advanced), funnel=funnel,
                            wrong_signed=wrong_signed)
