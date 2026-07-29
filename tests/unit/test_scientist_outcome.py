"""
Unit tests for the DERIVED terminal-outcome function (spec Appendix B).

Covers all six outcome branches AND the first-match-wins precedence — a record that is
simultaneously not-compiled AND not-audit_clean must return the EARLIER branch
(EXECUTION_FAILURE), never AUDIT_FAILURE — and asserts HOLDOUT_EVALUATED is returned ONLY when
every prior boolean is satisfied.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.scientist.experimentalist.outcome import derive_outcome  # noqa: E402
from agents.scientist.schemas.evaluation import Booleans  # noqa: E402
from agents.scientist.schemas.outcomes import Outcome  # noqa: E402

# The six G0-integrity booleans (any False => INVALID_PROPOSAL).
_G0 = (
    "schema_valid",
    "mechanism_authorised",
    "template_supported",
    "toggles_preserved",
    "inputs_available",
    "not_duplicate",
)
# The eleven booleans that must ALL be True for HOLDOUT_EVALUATED (holdout_evaluated itself is
# NOT read by derive_outcome — the outcome means "advanced to holdout").
_PRIOR_TO_HOLDOUT = _G0 + ("compiled", "execution_verified", "audit_clean", "bh_survived",
                           "cpcv_qualified")


def _bools(**overrides) -> Booleans:
    base = dict(
        schema_valid=True, mechanism_authorised=True, template_supported=True,
        toggles_preserved=True, inputs_available=True, not_duplicate=True,
        compiled=True, execution_verified=True, audit_clean=True,
        bh_survived=True, cpcv_qualified=True, holdout_evaluated=True,
    )
    base.update(overrides)
    return Booleans(**base)


# --- the six branches --------------------------------------------------------------------

def test_all_true_returns_holdout_evaluated():
    assert derive_outcome(_bools()) is Outcome.HOLDOUT_EVALUATED


@pytest.mark.parametrize("flag", _G0)
def test_any_g0_false_is_invalid_proposal(flag):
    assert derive_outcome(_bools(**{flag: False})) is Outcome.INVALID_PROPOSAL


@pytest.mark.parametrize("flag", ["compiled", "execution_verified"])
def test_compile_or_execute_false_is_execution_failure(flag):
    assert derive_outcome(_bools(**{flag: False})) is Outcome.EXECUTION_FAILURE


def test_not_audit_clean_is_audit_failure():
    assert derive_outcome(_bools(audit_clean=False)) is Outcome.AUDIT_FAILURE


def test_not_bh_survived_is_no_development_evidence():
    assert derive_outcome(_bools(bh_survived=False)) is Outcome.NO_DEVELOPMENT_EVIDENCE


def test_not_cpcv_qualified_is_development_survivor_not_advanced():
    assert derive_outcome(_bools(cpcv_qualified=False)) is Outcome.DEVELOPMENT_SURVIVOR_NOT_ADVANCED


# --- precedence: first-match-wins --------------------------------------------------------

def test_precedence_not_compiled_and_not_audit_clean_is_execution_failure():
    # Simultaneously not-compiled AND not-audit_clean -> the EARLIER branch wins.
    out = derive_outcome(_bools(compiled=False, audit_clean=False))
    assert out is Outcome.EXECUTION_FAILURE


def test_precedence_g0_beats_everything_downstream():
    out = derive_outcome(
        _bools(schema_valid=False, compiled=False, audit_clean=False,
               bh_survived=False, cpcv_qualified=False)
    )
    assert out is Outcome.INVALID_PROPOSAL


def test_precedence_audit_failure_beats_no_evidence():
    out = derive_outcome(_bools(audit_clean=False, bh_survived=False))
    assert out is Outcome.AUDIT_FAILURE


def test_precedence_no_evidence_beats_not_advanced():
    out = derive_outcome(_bools(bh_survived=False, cpcv_qualified=False))
    assert out is Outcome.NO_DEVELOPMENT_EVIDENCE


# --- HOLDOUT_EVALUATED only when every prior boolean is satisfied ------------------------

def test_holdout_evaluated_boolean_is_not_consumed_by_derivation():
    # Appendix B does not read holdout_evaluated; cpcv_qualified True (and all before) suffices.
    assert derive_outcome(_bools(holdout_evaluated=False)) is Outcome.HOLDOUT_EVALUATED


@pytest.mark.parametrize("flag", _PRIOR_TO_HOLDOUT)
def test_holdout_requires_every_prior_boolean(flag):
    # Flipping ANY of the eleven prior booleans must drop the outcome off HOLDOUT_EVALUATED.
    assert derive_outcome(_bools(**{flag: False})) is not Outcome.HOLDOUT_EVALUATED


def test_every_booleans_record_has_exactly_one_outcome():
    # Property (spec §14): every classified proposal has exactly one terminal outcome — i.e.
    # derive_outcome is total and returns a member of Outcome for any boolean assignment.
    import itertools

    fields = _PRIOR_TO_HOLDOUT + ("holdout_evaluated",)
    for combo in itertools.product([True, False], repeat=len(fields)):
        b = _bools(**dict(zip(fields, combo)))
        assert isinstance(derive_outcome(b), Outcome)
