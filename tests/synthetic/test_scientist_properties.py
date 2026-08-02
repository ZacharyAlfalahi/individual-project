"""Scientist §14 property tests + the 8 synthetic end-to-end state-machine cases (every terminal
outcome reachable, exactly one terminal outcome per state). These pass BEFORE any corpus run."""

import itertools
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.schemas.toggle import TOGGLE_IDS  # noqa: E402
from agents.scientist.experimentalist.outcome import derive_outcome  # noqa: E402
from agents.scientist.experimentalist.validator import validate_g0  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.equivalence import equivalence_key  # noqa: E402
from agents.scientist.schemas.evaluation import BOOLEAN_FIELDS, Booleans  # noqa: E402
from agents.scientist.schemas.outcomes import Outcome, RefusalCode  # noqa: E402
from agents.scientist.schemas.proposal import decode_proposal  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402
from shared.stats import run_fdr  # noqa: E402

LIB = load_library()
AVAILABLE = {"baa_aaa_spread", "vix", "term_spread", "rating", "investment_grade", "var_5pct",
             "gamma_illiq", "bond_vol", "size", "time_to_maturity"}
CASE = ScientistCase(
    case_id="c", strategy_id="str", corrected_quant_config_ref="qc", corrected_run_ref="r",
    audit_report_ref="a", failed_check_ids=("lib_gap",), failed_check_verdicts={"lib_gap": "FAIL"},
    applicable_toggles=("lib_gap", "lab_trim"),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False))


def _bools(**over):
    d = {f: True for f in BOOLEAN_FIELDS}
    d.update(over)
    return Booleans(**d)


def _proposal(**over):
    raw = {"proposal_id": "p", "case_id": "c", "parent_strategy_id": "str",
           "mechanism_ref": "mech_003", "template_ref": "lagged_binary_regime_interaction_v1",
           "rationale": "x", "prediction": "y",
           "config_delta": {"conditioning_variable": "baa_aaa_spread", "conditioning_lag_months": 1,
                            "interaction_form": "binary_above_historical_median"},
           "required_inputs": ["baa_aaa_spread"],
           "generation": {"source": "random_eligible", "seed": 0, "model": "d", "prompt_version": "v",
                          "library_version": "x", "generated_at": "2026-08-02T00:00:00Z"}}
    raw.update(over)
    return decode_proposal(raw).proposal


# ---- §14 properties -----------------------------------------------------------------------

def test_toggle_on_to_off_is_always_rejected():
    for t in TOGGLE_IDS:                                   # property over EVERY bias toggle
        out = validate_g0(_proposal(required_inputs=[t]), CASE, LIB, seen_keys=set(),
                          available_variables=AVAILABLE)
        assert not out.passed and out.refusal_code is RefusalCode.TOGGLE_REVERSAL


def test_duplicate_never_enters_the_executed_set():
    p = _proposal()
    out = validate_g0(p, CASE, LIB, seen_keys={equivalence_key(p)}, available_variables=AVAILABLE)
    assert out.refusal_code is RefusalCode.DUPLICATE_PROPOSAL
    assert derive_outcome(_bools(not_duplicate=False)) is Outcome.INVALID_PROPOSAL


def test_bh_adjusted_p_is_monotone_in_rank():
    rng = np.random.default_rng(0)
    for _ in range(20):
        pvals = {f"k{i}": float(p) for i, p in enumerate(rng.uniform(0, 1, 8))}
        rep = run_fdr(pvals, 0.10)
        ordered = sorted(rep.decisions.values(), key=lambda d: d.rank)
        adj = [d.adjusted_p for d in ordered]
        assert all(adj[i] <= adj[i + 1] + 1e-12 for i in range(len(adj) - 1))   # monotone from top


def test_exactly_one_terminal_outcome_for_every_state():
    for combo in itertools.product([True, False], repeat=len(BOOLEAN_FIELDS)):
        assert isinstance(derive_outcome(Booleans(**dict(zip(BOOLEAN_FIELDS, combo)))), Outcome)


def test_audit_clean_never_decides_without_execution_verified():
    # audit_clean must imply execution_verified for the outcome to reflect audit — if execution
    # failed, the outcome is EXECUTION_FAILURE regardless of audit_clean.
    assert derive_outcome(_bools(execution_verified=False, audit_clean=True)) is Outcome.EXECUTION_FAILURE


# ---- 8 synthetic end-to-end state-machine cases -------------------------------------------

def test_eight_state_machine_cases():
    cases = {
        "invalid_schema": (_bools(schema_valid=False), Outcome.INVALID_PROPOSAL),
        "invalid_duplicate": (_bools(not_duplicate=False), Outcome.INVALID_PROPOSAL),
        "exec_not_compiled": (_bools(compiled=False), Outcome.EXECUTION_FAILURE),
        "exec_not_verified": (_bools(execution_verified=False), Outcome.EXECUTION_FAILURE),
        "audit_failure": (_bools(audit_clean=False), Outcome.AUDIT_FAILURE),
        "no_dev_evidence": (_bools(bh_survived=False), Outcome.NO_DEVELOPMENT_EVIDENCE),
        "not_advanced": (_bools(cpcv_qualified=False), Outcome.DEVELOPMENT_SURVIVOR_NOT_ADVANCED),
        "holdout_evaluated": (_bools(), Outcome.HOLDOUT_EVALUATED),
    }
    assert len(cases) == 8
    for name, (bools, expected) in cases.items():
        assert derive_outcome(bools) is expected, name
