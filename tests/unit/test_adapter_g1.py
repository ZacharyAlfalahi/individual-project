"""
G1 -- adapter exhaustive transform tests (build brief §8).

For every transform rule, enumerate its FULL input domain (via
``iter_domain_cases``) and assert a HAND-SPECIFIED expected output. The expected
tables below are written by hand (NOT read from transform_table.yaml), so a test is
an INDEPENDENT check of the rulebook, not a tautology: if a new enum value is added
to a domain, or the YAML map drifts, a test fails. Attribution: because the input is
a known-good enumerated domain value, any failure is the adapter/table's (never
extraction's).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _librarian_fixtures import located_quote, signal_ref  # noqa: E402
from agents.librarian.adapter.signal_resolution import resolve_signal  # noqa: E402
from agents.librarian.adapter.transform_table import (  # noqa: E402
    Omit,
    Produced,
    Review,
    load_transform_table,
)
from agents.librarian.registries.silence_policy import load_silence_policy_table  # noqa: E402
from agents.librarian.schema.signal_ref import DescribedSignal, SignalRef  # noqa: E402
from agents.librarian.validators.domains import iter_domain_cases  # noqa: E402
from agents.quant.config.concept_column import load_concept_column_table  # noqa: E402

TT = load_transform_table()
CT = load_concept_column_table()


def _domain(field: str) -> tuple:
    return dict(iter_domain_cases(family="part2"))[field]


# --- long_leg -> long_group / short_group (needs n_groups) -----------------------

def test_g1_long_leg_full_domain_at_g5():
    expected = {
        "lowest_signal": {"long_group": 0, "short_group": 4},
        "highest_signal": {"long_group": 4, "short_group": 0},
        # "other" is off-menu -> no engine direction -> Review.
    }
    for value in _domain("long_leg"):
        out = TT.apply_long_leg(value, 5)
        if value == "other":
            assert isinstance(out, Review)
        else:
            assert isinstance(out, Produced) and out.value == expected[value]


def test_g1_long_leg_tracks_n_groups():
    # top group = g-1 for any g.
    for g in (2, 5, 10):
        assert TT.apply_long_leg("lowest_signal", g).value == {"long_group": 0, "short_group": g - 1}
        assert TT.apply_long_leg("highest_signal", g).value == {"long_group": g - 1, "short_group": 0}


def test_g1_long_leg_silent_n_groups_reviews():
    # No group count -> cannot place g-1 without the factory default (P5) -> Review.
    assert isinstance(TT.apply_long_leg("lowest_signal", None), Review)


# --- weighting (composite: weighting_scheme + weighting_base) ---------------------

def test_g1_weighting_full_cross_domain():
    # Hand table over the full scheme x base cross-product (+ the silent-base case).
    for scheme in _domain("weighting_scheme"):
        for base in list(_domain("weighting_base")) + [None]:
            out = TT.apply_weighting(scheme, base)
            if scheme == "equal":
                assert isinstance(out, Produced) and out.value == "equal"
            elif scheme == "value":
                if base is None:
                    assert isinstance(out, Omit)  # silent base -> factory par default
                elif base == "par":
                    assert isinstance(out, Produced) and out.value == "size"  # TRUE match
                else:  # market_value / other -> forwarded so the factory refuses
                    assert isinstance(out, Produced) and out.value == base
            else:  # scheme == "other" -> forwarded, factory OUT_OF_ENUM_WEIGHTING
                assert isinstance(out, Produced) and out.value == "other"


# --- expost_trim -----------------------------------------------------------------

def test_g1_trim_full_domain():
    for value in _domain("expost_trim"):
        out = TT.apply_trim(value)
        if value == "none":
            assert isinstance(out, Omit)
        else:  # v1 carries no bounds -> Review (never a fabricated unbounded trim)
            assert isinstance(out, Review)


# --- signal resolution over the 7 registry concepts + the escape -----------------

def test_g1_signal_resolution_over_all_concepts():
    grounded = {
        "var_5pct": "var_5pct",
        "credit_rating": "rating",
        "past_6m_cumulative_return": "mom6",
        "bpw_gamma": "gamma",
    }
    deferred = ("prior_1m_excess_return", "maturity", "size")
    for concept, column in grounded.items():
        b = resolve_signal(signal_ref(concept), CT)
        assert b.tag == "BOUND" and b.value == column
    for concept in deferred:
        b = resolve_signal(signal_ref(concept), CT)
        assert b.tag == "MISSING" and b.value is None
    # the 'unrecognised' escape -> MISSING (never AMBIGUOUS)
    escape = SignalRef(
        concept_id=signal_ref("unrecognised").concept_id,
        as_described=DescribedSignal(label="x", quotes=(located_quote(),)),
    )
    assert resolve_signal(escape, CT).tag == "MISSING"


def test_g1_adapter_never_emits_ambiguous():
    # Structural: the table is a function, so no resolution is ever AMBIGUOUS.
    for concept in ("var_5pct", "credit_rating", "past_6m_cumulative_return", "bpw_gamma",
                    "maturity", "size", "prior_1m_excess_return"):
        assert resolve_signal(signal_ref(concept), CT).tag in ("BOUND", "MISSING")


# --- combiner --------------------------------------------------------------------

def test_g1_combiner_domain():
    # The representable combiners live in the table; 'other' is deliberately absent
    # (the adapter refuses it, UNSUPPORTED_COMBINER).
    assert set(TT.combiner) == {"single_leg", "equal_average"}
    assert TT.combiner["equal_average"]["divisor"] == "available"
    assert "other" in _domain("combiner")  # the escape is a legal menu value...
    assert "other" not in TT.combiner       # ...but not a representable combiner


# --- completeness: no Part 2 field is silently unhandled --------------------------

def test_g1_every_part2_field_has_a_home():
    # Every scalar Part 2 field is either a transform input (engine hook) or has a
    # silence-policy entry (check-only). A new field with neither would fail here.
    silence = load_silence_policy_table()
    silence_fields = set(silence.policies["sort_block"]) | set(silence.policies["common"])
    homed = TT.all_inputs() | silence_fields
    for field, _ in iter_domain_cases(family="part2"):
        assert field in homed, f"Part 2 field {field!r} has no transform and no silence policy"


def test_g1_engine_fields_are_the_expected_nine():
    # The transform table's engine hooks are exactly the nine documented fields.
    assert set(TT.engine_fields()) == {
        "sort_signal", "control_axis", "n_groups", "long_leg", "signal_lag",
        "min_bonds", "holding_period", "weighting_scheme", "expost_trim",
    }
