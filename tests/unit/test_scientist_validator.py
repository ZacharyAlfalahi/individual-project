"""G0 proposal-integrity gate (spec §9): a valid proposal passes with all six G0 booleans True,
and each of the eight refusal codes is reachable, first-fail-wins."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.experimentalist.validator import validate_g0  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.equivalence import equivalence_key  # noqa: E402
from agents.scientist.schemas.outcomes import RefusalCode  # noqa: E402
from agents.scientist.schemas.proposal import decode_proposal  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402

LIB = load_library()
AVAILABLE = {"baa_aaa_spread", "vix", "term_spread", "rating", "investment_grade", "var_5pct",
             "gamma_illiq", "bond_vol", "size", "time_to_maturity"}
CASE = ScientistCase(
    case_id="case_x", strategy_id="str", corrected_quant_config_ref="qc://c",
    corrected_run_ref="run://c", audit_report_ref="a://c", failed_check_ids=("lib_gap",),
    failed_check_verdicts={"lib_gap": "FAIL"}, applicable_toggles=("lib_gap", "lab_trim"),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False),
)


def _proposal(**over):
    raw = {
        "proposal_id": "prop_test", "case_id": "case_x", "parent_strategy_id": "str",
        "mechanism_ref": "mech_003", "template_ref": "lagged_binary_regime_interaction_v1",
        "rationale": "credit regime conditioning", "prediction": "improves risk-adjusted result",
        "config_delta": {"conditioning_variable": "baa_aaa_spread", "conditioning_lag_months": 1,
                         "interaction_form": "binary_above_historical_median"},
        "required_inputs": ["baa_aaa_spread"],
        "generation": {"source": "random_eligible", "seed": 0, "model": "d", "prompt_version": "v0",
                       "library_version": "x", "generated_at": "2026-08-01T00:00:00Z"},
    }
    raw.update(over)
    dec = decode_proposal(raw)
    assert dec.ok, dec.error
    return dec.proposal


def _g0(proposal, *, seen=frozenset()):
    return validate_g0(proposal, CASE, LIB, seen_keys=set(seen), available_variables=AVAILABLE)


def test_valid_proposal_passes_g0():
    out = _g0(_proposal())
    assert out.passed
    assert all(out.booleans.values())
    assert out.refusal_code is None


@pytest.mark.parametrize("over,code,false_field", [
    ({"config_delta": {"conditioning_variable": None, "conditioning_lag_months": 1,
                       "interaction_form": "binary_above_historical_median"}},
     RefusalCode.INVALID_SCHEMA, "schema_valid"),
    ({"case_id": "wrong_case"}, RefusalCode.INVALID_SCHEMA, "schema_valid"),
    ({"mechanism_ref": "mech_999"}, RefusalCode.UNKNOWN_MECHANISM, "mechanism_authorised"),
    ({"template_ref": "signal_characteristic_interaction_v1"},  # not allowed by mech_003
     RefusalCode.UNSUPPORTED_TEMPLATE, "template_supported"),
    ({"config_delta": {"conditioning_variable": "var_5pct", "conditioning_lag_months": 1,
                       "interaction_form": "binary_above_historical_median"}},  # not in T1 enum
     RefusalCode.FIELD_OUT_OF_DOMAIN, "template_supported"),
    ({"required_inputs": ["survivorship"]}, RefusalCode.TOGGLE_REVERSAL, "toggles_preserved"),
    ({"required_inputs": ["holdout_path"]}, RefusalCode.FORBIDDEN_CHANGE, "toggles_preserved"),
    ({"required_inputs": ["not_a_real_variable"]}, RefusalCode.MISSING_INPUT, "inputs_available"),
])
def test_refusal_codes_reachable(over, code, false_field):
    out = _g0(_proposal(**over))
    assert not out.passed
    assert out.refusal_code is code
    assert out.booleans[false_field] is False


def test_duplicate_is_rejected():
    p = _proposal()
    out = _g0(p, seen={equivalence_key(p)})
    assert not out.passed
    assert out.refusal_code is RefusalCode.DUPLICATE_PROPOSAL
    assert out.booleans["not_duplicate"] is False


def test_first_failure_wins_ordering():
    # An unknown mechanism AND a bad template: mechanism check (earlier) must win.
    out = _g0(_proposal(mechanism_ref="mech_999", template_ref="nope"))
    assert out.refusal_code is RefusalCode.UNKNOWN_MECHANISM
