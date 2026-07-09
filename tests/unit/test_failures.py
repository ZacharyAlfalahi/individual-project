"""
Unit tests for the failure taxonomy (build brief §5.6, D31).

The four typed outcome events and their routing. All are events on the run
record, never tags (D24).
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline.failures import (
    FAILURE_ROUTING,
    PAPER_FAILED,
    REVIEW,
    EnumerationDisagreement,
    LocatorSystematicFailure,
    PartialParse,
    UnparseablePdf,
    route_of,
)


# --- routing table ----------------------------------------------------------

def test_routing_table_matches_d31():
    assert FAILURE_ROUTING["unparseable_pdf"] == PAPER_FAILED
    assert FAILURE_ROUTING["partial_parse"] == PAPER_FAILED
    assert FAILURE_ROUTING["locator_systematic_failure"] == REVIEW
    assert FAILURE_ROUTING["enumeration_disagreement"] == REVIEW


# --- the four events --------------------------------------------------------

def test_unparseable_pdf_routes_to_paper_failed():
    ev = UnparseablePdf(paper_id="P", detail="no text extracted")
    assert ev.kind == "unparseable_pdf"
    assert ev.routing == PAPER_FAILED
    assert route_of(ev) == PAPER_FAILED
    assert ev.to_dict()["routing"] == PAPER_FAILED


def test_partial_parse_routes_to_paper_failed():
    ev = PartialParse(paper_id="P", detail="only 3 of 20 pages", pages_recovered=3)
    assert ev.routing == PAPER_FAILED
    assert ev.to_dict()["pages_recovered"] == 3


def test_locator_systematic_failure_routes_to_review():
    ev = LocatorSystematicFailure(paper_id="P", match_rate=0.4, bar=0.8)
    assert ev.routing == REVIEW
    assert route_of(ev) == REVIEW


def test_locator_systematic_failure_validates_rate_bounds():
    with pytest.raises(LibrarianSchemaError):
        LocatorSystematicFailure(paper_id="P", match_rate=1.5, bar=0.8)


def test_enumeration_disagreement_routes_to_review():
    ev = EnumerationDisagreement(
        paper_id="P", detail="only model_a: ['X']", only_model_a=("X",)
    )
    assert ev.routing == REVIEW
    assert ev.only_model_a == ("X",)
    assert ev.to_dict()["only_model_a"] == ["X"]


# --- validation + non-event routing -----------------------------------------

def test_events_reject_empty_paper_id():
    with pytest.raises(LibrarianSchemaError):
        UnparseablePdf(paper_id="", detail="d")


def test_route_of_rejects_non_event():
    with pytest.raises(LibrarianSchemaError):
        route_of("not an event")
