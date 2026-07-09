"""
D22 planted-registry test: an out-of-registry concept is rejected, and the
correct extraction answer is the 'unrecognised' escape (with the paper's words).
"""

from agents.librarian.schema import UNRECOGNISED
from agents.librarian.validators import validate_librarian_spec

from _librarian_fixtures import (
    FakeSignalRegistry,
    build_leg,
    build_part2,
    build_spec,
    located_quote,
    signal_ref,
)


def test_planted_out_of_registry_concept_is_rejected():
    # A signal that names a concept the registry does not hold.
    leg = build_leg(sort_signal=signal_ref("bespoke_downside_beta"))
    spec = build_spec(part2=build_part2(legs=[leg]))
    errs = validate_librarian_spec(spec, registry=FakeSignalRegistry())
    assert len(errs) == 1
    assert "not in the Signal Concept Registry" in errs[0].reason
    assert UNRECOGNISED in errs[0].reason  # the error names the correct answer


def test_unrecognised_is_the_correct_answer_for_the_same_signal():
    # The same out-of-registry signal, correctly tagged 'unrecognised' with the
    # paper's own words, validates clean.
    leg = build_leg(
        sort_signal=signal_ref(
            UNRECOGNISED,
            quotes=(located_quote(text="a bespoke downside beta measure"),),
            label="bespoke downside beta",
        )
    )
    spec = build_spec(part2=build_part2(legs=[leg]))
    errs = validate_librarian_spec(spec, registry=FakeSignalRegistry())
    assert errs == []
