"""Experimentalist orchestrator — proposals flow through G0->G5, EvaluationRecords are assembled
with DERIVED final_outcomes, the BH-FDR is joint across the family, and the funnel denominators
are visible. Uses a synthetic panel + factors."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.outcomes import Outcome, RefusalCode  # noqa: E402
from agents.scientist.schemas.proposal import decode_proposal  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402
from shared.evaluation.thresholds import CrowdingConfig  # noqa: E402

LIB = load_library()
FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")
MONTHS = pd.date_range("2004-08-31", periods=60, freq="ME")
AVAILABLE = {"baa_aaa_spread", "vix", "term_spread", "rating", "investment_grade", "var_5pct",
             "gamma_illiq", "bond_vol", "size", "time_to_maturity"}
CASE = ScientistCase(
    case_id="case_x", strategy_id="str", corrected_quant_config_ref="qc://c",
    corrected_run_ref="run://c", audit_report_ref="a://c", failed_check_ids=("lib_gap",),
    failed_check_verdicts={"lib_gap": "FAIL"}, applicable_toggles=("lib_gap", "lab_trim"),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False),
)


def _panel(n=20):
    rows = []
    for b in range(n):
        for i, d in enumerate(MONTHS):
            spread = 0.002 + 0.0005 * (i % 5)
            rows.append(dict(cusip=f"B{b:02d}", date=d, ret=spread * (b - (n - 1) / 2), size=100.0,
                             score=float(b), investment_grade=(b >= n // 2),
                             rating=float((b * 7) % n), gamma_illiq=float(n - b)))
    return pd.DataFrame(rows)


def _bbw4():
    rng = np.random.default_rng(7)
    return pd.DataFrame({"date": MONTHS, **{f: rng.normal(scale=0.01, size=len(MONTHS))
                                            for f in ("mktb", "drf", "crf", "lrf")}})


def _crowding_factors():
    rng = np.random.default_rng(8)
    return pd.DataFrame({"date": MONTHS, **{f: rng.normal(scale=0.02, size=len(MONTHS))
                                            for f in FACTORS}})


def _cfg():
    return CrowdingConfig(factor_set=FACTORS, hac_lag_rule="newey_west_auto", min_obs=10,
                          bundles={f: (f"{f}.parquet", f"{f}_corr") for f in FACTORS})


def _proposal(pid, mechanism, template, variable, form):
    raw = {
        "proposal_id": pid, "case_id": "case_x", "parent_strategy_id": "str",
        "mechanism_ref": mechanism, "template_ref": template, "rationale": "x", "prediction": "y",
        "config_delta": {"conditioning_variable": variable, "conditioning_lag_months": 1,
                         "interaction_form": form},
        "required_inputs": [variable],
        "generation": {"source": "random_eligible", "seed": 0, "model": "d", "prompt_version": "v0",
                       "library_version": "x", "generated_at": "2026-08-02T00:00:00Z"},
    }
    dec = decode_proposal(raw)
    assert dec.ok, dec.error
    return dec.proposal


def _run(proposals):
    return run_experimentalist(
        CASE, proposals, LIB, panel=_panel(), base_rulebook={"score": "score", "groups": 2,
        "weighting": "equal", "min_bonds": 4, "signal_lag": 0, "nw_lags": 0},
        bbw4_factors=_bbw4(), holding_period=1, signal_lookback=1, available_variables=AVAILABLE,
        m=6, cap=3, crowding_config=_cfg(), crowding_factors=_crowding_factors(), nw_lags=0)


def test_full_stack_flow_and_funnel():
    valid = _proposal("good", "mech_008", "ex_ante_universe_conditioning_v1", "gamma_illiq",
                      "restrict_top_liquidity_tercile")
    unknown = _proposal("bad", "mech_999", "ex_ante_universe_conditioning_v1", "gamma_illiq",
                        "restrict_top_liquidity_tercile")
    dup = _proposal("dup", "mech_008", "ex_ante_universe_conditioning_v1", "gamma_illiq",
                    "restrict_top_liquidity_tercile")            # same equivalence key as 'good'
    report = _run([valid, unknown, dup])

    by_id = {r.proposal_id: r for r in report.records}
    # the unknown mechanism and the duplicate are G0 refusals (INVALID_PROPOSAL).
    assert by_id["bad"].final_outcome is Outcome.INVALID_PROPOSAL
    assert by_id["bad"].refusal_code is RefusalCode.UNKNOWN_MECHANISM
    assert by_id["dup"].refusal_code is RefusalCode.DUPLICATE_PROPOSAL
    # the valid proposal cleared G0-G2 (audit-clean) and G3 computed performance.
    assert by_id["good"].booleans.audit_clean
    assert by_id["good"].measurements.gross is not None
    assert by_id["good"].final_outcome is not Outcome.INVALID_PROPOSAL
    # funnel denominators are visible and total the input count.
    assert sum(report.funnel.values()) == 3
    assert len(report.records) == 3


def test_joint_bh_runs_and_records_are_complete():
    # two valid proposals -> BH-FDR is joint over both; every proposal gets a record.
    p1 = _proposal("m1", "mech_008", "ex_ante_universe_conditioning_v1", "gamma_illiq",
                   "restrict_top_liquidity_tercile")
    p2 = _proposal("m2", "mech_009", "ex_ante_universe_conditioning_v1", "rating",
                   "restrict_investment_grade")
    report = _run([p1, p2])
    assert len(report.records) == 2
    for r in report.records:
        assert r.booleans.audit_clean and r.measurements.gross is not None
        assert r.measurements.gross.p_bh is not None          # joint BH adjusted-p populated
