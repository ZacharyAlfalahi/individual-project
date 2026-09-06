"""WS-C (P2) — census router: routing into refusal/compilable/eligibility
partitions, the corpus_fate refusal oracle, and fail-loud discipline."""

from __future__ import annotations

import pytest

from agents.librarian import corpus_fate as cf
from evaluation.codegen.census import (
    ELIGIBILITY_EXCLUSION_REASONS,
    CensusInput,
    CensusMember,
    CensusResult,
    P2CensusError,
    RoutingDecision,
    run_census,
)


def _spec(paper_id: str) -> dict:
    return {"header": {"paper_id": paper_id}, "part1": {"formation_structure": "sort"}}


# --- the D21 walk as the fixture-test oracle -----------------------------------
# corpus_fate.refusals() is the frozen oracle; the router must reproduce it when
# fed inputs consistent with the D21 walk (routing via fate_of).

def _route_by_corpus_fate(inp: CensusInput) -> RoutingDecision:
    fate = cf.fate_of(inp.paper_id).fate
    if fate in cf.REFUSAL_FATES:
        return RoutingDecision(compilable=False, refused=True, refusal_reason=fate)
    return RoutingDecision(compilable=True, refused=False)


def test_router_reproduces_the_corpus_fate_refusal_set():
    inputs = [CensusInput(r.paper, _spec(r.paper), True) for r in cf.all_fates()]
    result = run_census(inputs, _route_by_corpus_fate)
    got = {m.paper_id for m in result.refusal_set()}
    expected = {r.paper for r in cf.refusals()}
    assert got == expected
    # every refusal reason is a closed REFUSAL_FATES value
    for m in result.refusal_set():
        assert m.refusal_reason in cf.REFUSAL_FATES


def test_compilable_set_is_the_non_refusals():
    inputs = [CensusInput(r.paper, _spec(r.paper), True) for r in cf.all_fates()]
    result = run_census(inputs, _route_by_corpus_fate)
    got = {m.paper_id for m in result.compilable_set()}
    expected = {r.paper for r in cf.all_fates() if not r.is_refusal}
    assert got == expected
    assert result.eligibility_exclusions() == ()


# --- eligibility exclusions are typed, counted, and NEVER routed -----------------

def test_eligibility_exclusion_is_counted_and_not_routed():
    calls = {"n": 0}

    def counting_route(inp: CensusInput) -> RoutingDecision:
        calls["n"] += 1
        return RoutingDecision(compilable=True, refused=False)

    inputs = [
        CensusInput("ok_paper", _spec("ok_paper"), True),
        CensusInput("bad_text", None, False, exclusion_reason="text_acquisition_failed"),
    ]
    result = run_census(inputs, counting_route)
    assert calls["n"] == 1                       # the excluded paper was NOT routed
    excl = result.eligibility_exclusions()
    assert [m.paper_id for m in excl] == ["bad_text"]
    assert excl[0].disposition == "eligibility_excluded"
    assert excl[0].exclusion_reason == "text_acquisition_failed"
    assert excl[0].refused is False and excl[0].compilable is False


def test_eligibility_exclusion_requires_a_typed_reason():
    with pytest.raises(P2CensusError):
        CensusInput("bad", None, False)          # no exclusion_reason
    with pytest.raises(P2CensusError):
        CensusInput("ok", {}, True, exclusion_reason="should not be here")


# --- fail-loud construction guards ---------------------------------------------

def test_off_enum_refusal_reason_is_a_build_error():
    with pytest.raises(P2CensusError):
        RoutingDecision(compilable=False, refused=True, refusal_reason="route_to_the_moon")


def test_decision_cannot_be_both_compilable_and_refused():
    with pytest.raises(P2CensusError):
        RoutingDecision(compilable=True, refused=True, refusal_reason="refuse_no_strategy")


def test_member_inconsistency_is_a_build_error():
    with pytest.raises(P2CensusError):
        CensusMember("p", compilable=True, refused=True, refusal_reason=None,
                     text_quality_ok=True, extracted_spec={})
    with pytest.raises(P2CensusError):
        # eligibility-excluded but also marked compilable
        CensusMember("p", compilable=True, refused=False, refusal_reason=None,
                     text_quality_ok=False, extracted_spec=None,
                     exclusion_reason="x")


def test_duplicate_paper_is_a_build_error():
    with pytest.raises(P2CensusError):
        CensusResult((
            CensusMember("p", True, False, None, True, {}),
            CensusMember("p", True, False, None, True, {}),
        ))


def test_member_lookup_fails_loud_on_unknown_paper():
    result = run_census([CensusInput("p", _spec("p"), True)],
                        lambda i: RoutingDecision(True, False))
    assert result.member("p").paper_id == "p"
    with pytest.raises(P2CensusError):
        result.member("nobody")


def test_fate_table_covers_every_member():
    inputs = [
        CensusInput("r", _spec("r"), True),
        CensusInput("c", _spec("c"), True),
        CensusInput("x", None, False, exclusion_reason="text_quality_below_bar"),
    ]

    def route(inp: CensusInput) -> RoutingDecision:
        if inp.paper_id == "r":
            return RoutingDecision(False, True, "refuse_family_unsupported")
        return RoutingDecision(True, False)

    result = run_census(inputs, route)
    table = result.fate_table()
    assert {row["paper_id"] for row in table} == {"r", "c", "x"}
    dispo = {row["paper_id"]: row["disposition"] for row in table}
    assert dispo == {"r": "refused", "c": "compilable", "x": "eligibility_excluded"}


# --- closed eligibility-exclusion vocabulary -----------------------------------

def test_the_closeout_reason_is_in_the_closed_vocabulary():
    # the P2 coverage-boundary close-out relies on this exact reason
    assert "extraction_review_exit_no_spec" in ELIGIBILITY_EXCLUSION_REASONS


@pytest.mark.parametrize("reason", sorted(ELIGIBILITY_EXCLUSION_REASONS))
def test_every_closed_reason_is_accepted(reason):
    inp = CensusInput("p", None, False, exclusion_reason=reason)
    assert inp.exclusion_reason == reason
    m = CensusMember("p", False, False, None, False, None, exclusion_reason=reason)
    assert m.disposition == "eligibility_excluded"


def test_an_off_vocabulary_reason_is_a_build_error():
    with pytest.raises(P2CensusError, match="ELIGIBILITY_EXCLUSION_REASONS"):
        CensusInput("p", None, False, exclusion_reason="totally_made_up")
    with pytest.raises(P2CensusError, match="ELIGIBILITY_EXCLUSION_REASONS"):
        CensusMember("p", False, False, None, False, None,
                     exclusion_reason="totally_made_up")
