"""
Standing-substitutions mechanism (evaluation contract §6) -- loader + adapter apply.

Covers: the byte-hash loader (absent -> empty; verify_hash), the two §6 structural
guards (DESIGN + changes_variant_status=false), the par-weighting apply path through
``adapt_spec`` (market_value -> par as DESIGN, WITHOUT variant), the empty-default
regression (behaviour byte-identical to pre-standing-subs), and the D27 bright line
(never substitute over an UNKNOWN/silent base).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest  # noqa: E402

from _adapter_fixtures import adapter_spec  # noqa: E402
from _librarian_fixtures import build_header, stated, unknown  # noqa: E402

from agents.librarian.adapter import adapt_spec  # noqa: E402
from agents.librarian.errors import LibrarianSchemaError  # noqa: E402
from agents.librarian.registries.standing_substitutions import (  # noqa: E402
    NO_STANDING_SUBS,
    STANDING_SUBS_V1_SHA256,
    Substitution,
    load_standing_substitutions,
)
from agents.quant.config import RefusalCode, to_rulebook  # noqa: E402


def _par_table():
    return load_standing_substitutions()


# --------------------------------------------------------------------------- #
# Loader + hashing
# --------------------------------------------------------------------------- #

def test_loads_the_par_weighting_convention():
    t = _par_table()
    assert t.version == "v1"
    assert len(t.substitutions) == 1
    sub = t.substitution_for("weighting_base", "market_value")
    assert sub is not None
    assert sub.id == "par_weighting_v1"
    assert sub.replacement == "par"
    assert sub.changes_variant_status is False  # §6: standing -> NOT variant


def test_recorded_hash_matches_and_absent_loads_empty(tmp_path):
    assert _par_table().verify_hash(STANDING_SUBS_V1_SHA256) is True
    # An absent file loads as the empty table (the ordinary-adapter default) --
    # its empty hash never matches the recorded sha, so a hash-requiring harness fails loud.
    empty = load_standing_substitutions(tmp_path / "nope.yaml")
    assert empty.substitutions == ()
    assert empty.verify_hash(STANDING_SUBS_V1_SHA256) is False
    assert NO_STANDING_SUBS.verify_hash(STANDING_SUBS_V1_SHA256) is False


def test_par_does_not_match_the_predicate():
    # drf STATES weighting_base=par -> no substitution fires (byte-equal without intervention).
    assert _par_table().substitution_for("weighting_base", "par") is None


def test_structural_guards_reject_non_design_and_variant():
    with pytest.raises(LibrarianSchemaError):
        Substitution("x", "weighting_base", ("market_value",), "par", "INFERRED", False, "n", "s")
    with pytest.raises(LibrarianSchemaError):
        Substitution("x", "weighting_base", ("market_value",), "par", "DESIGN", True, "n", "s")
    with pytest.raises(LibrarianSchemaError):
        Substitution("x", "weighting_base", ("market_value",), "par", "DESIGN", False, "  ", "s")


# --------------------------------------------------------------------------- #
# Apply path through adapt_spec
# --------------------------------------------------------------------------- #

def test_market_value_becomes_par_as_design_without_variant():
    spec = adapter_spec(weighting_base=stated("market_value"))
    r = adapt_spec(spec, standing_subs=_par_table())

    # The par-weighting convention fired: weighting -> size (DESIGN), rulebook by_size.
    lc = r.leg_calls[0]
    assert lc.kwargs["weighting"].value == "size"
    assert lc.kwargs["weighting"].tag == "DESIGN"  # D24: DESIGN input -> DESIGN output
    assert to_rulebook(lc.result)["weighting"] == "by_size"

    # Standing -> NOT a variant (stays in fidelity aggregates, D23); recorded for the register.
    assert r.variant is False
    assert not r.refused
    assert len(r.standing_subs_applied) == 1
    applied = r.standing_subs_applied[0]
    assert (applied.field, applied.paper_value, applied.engine_value, applied.substitution_id) == (
        "weighting_base", "market_value", "par", "par_weighting_v1",
    )


def test_empty_table_preserves_out_of_enum_refusal():
    # Without the standing table, a market-value weighting rides through to the factory,
    # which refuses OUT_OF_ENUM_WEIGHTING -- the default behaviour, unchanged.
    spec = adapter_spec(weighting_base=stated("market_value"))
    r = adapt_spec(spec)  # default = empty standing table
    assert r.refused
    assert r.leg_calls[0].result.code is RefusalCode.OUT_OF_ENUM_WEIGHTING
    assert r.standing_subs_applied == ()


def test_bright_line_never_substitutes_a_silent_base():
    # weighting_base UNKNOWN (an extraction gap) is NOT a divergence to authorise:
    # the substitution is skipped even with the par table present.
    spec = adapter_spec(weighting_base=unknown())
    r = adapt_spec(spec, standing_subs=_par_table())
    assert r.standing_subs_applied == ()  # bright line held


def test_default_adapt_is_byte_identical_regression_pin():
    # The additive change must not perturb the default path: a par-weighted spec adapts
    # exactly as before -- no standing sub applied, not a variant.
    spec = adapter_spec()  # weighting_base=par by default
    r = adapt_spec(spec)
    assert r.standing_subs_applied == ()
    assert r.variant is False
    assert to_rulebook(r.leg_calls[0].result)["weighting"] == "by_size"


# --------------------------------------------------------------------------- #
# SpecHeader / RunProvenance stamping (contract §6 temporal rule)
# --------------------------------------------------------------------------- #

def test_header_omits_standing_stamps_when_absent():
    # A header without standing stamps serialises byte-identically to pre-standing-subs.
    d = build_header().to_dict()
    assert "standing_substitutions_version" not in d
    assert "standing_substitutions_hash" not in d


def test_header_carries_standing_stamps_when_present():
    d = build_header(
        standing_substitutions_version="v1",
        standing_substitutions_hash=STANDING_SUBS_V1_SHA256,
    ).to_dict()
    assert d["standing_substitutions_version"] == "v1"
    assert d["standing_substitutions_hash"] == STANDING_SUBS_V1_SHA256
