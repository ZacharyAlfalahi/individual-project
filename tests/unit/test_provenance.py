"""Unit tests for the provenance wrappers (Evidence / Inherited / Binding)."""

import pytest

from agents.quant.config import Binding, Evidence, Inherited
from agents.quant.config.provenance import ProvenanceError
from agents.quant.library.characteristic_sort import _RESERVED_COLUMNS


# --- flavour separation is structural --------------------------------------

def test_inherited_rejects_binding_tag():
    with pytest.raises(ProvenanceError):
        Inherited(5, "BOUND", Evidence(column="x"))


def test_binding_rejects_inherited_tag():
    with pytest.raises(ProvenanceError):
        Binding("x", "STATED", Evidence(quote="q"))


# --- Inherited per-tag evidence requirements -------------------------------

def test_stated_requires_quote():
    with pytest.raises(ProvenanceError):
        Inherited(5, "STATED", Evidence())
    Inherited(5, "STATED", Evidence(quote="the paper says five"))  # ok


def test_inferred_requires_rule_id():
    with pytest.raises(ProvenanceError):
        Inherited(5, "INFERRED", Evidence(note="derived"))
    Inherited(5, "INFERRED", Evidence(rule_id="R1", rule_text="..."))  # ok


def test_design_requires_note():
    with pytest.raises(ProvenanceError):
        Inherited("size", "DESIGN", Evidence())
    Inherited("size", "DESIGN", Evidence(note="par-weighting project decision"))  # ok


def test_unknown_requires_note_and_allows_none_value():
    with pytest.raises(ProvenanceError):
        Inherited(None, "UNKNOWN", Evidence())
    Inherited(None, "UNKNOWN", Evidence(note="not stated; searched §3-§5"))  # ok


# --- Binding per-tag evidence requirements ---------------------------------

def test_missing_requires_none_value_and_note():
    with pytest.raises(ProvenanceError):
        Binding("col", "MISSING", Evidence(note="x"))       # value must be None
    with pytest.raises(ProvenanceError):
        Binding(None, "MISSING", Evidence())                # note required
    b = Binding(None, "MISSING", Evidence(note="no column for the XYZ signal"))
    assert not b.is_usable


def test_bound_requires_column_and_is_usable():
    with pytest.raises(ProvenanceError):
        Binding("col", "BOUND", Evidence())
    b = Binding("col", "BOUND", Evidence(column="col", note="why"))
    assert b.is_usable


def test_ambiguous_requires_candidates_and_chosen_equals_value():
    with pytest.raises(ProvenanceError):
        Binding("a", "AMBIGUOUS", Evidence(candidates=("a", "b"), chosen="b"))
    b = Binding("a", "AMBIGUOUS", Evidence(candidates=("a", "b"), chosen="a", note="rule"))
    assert b.is_usable


# --- F6: no binding to a reserved engine-internal column -------------------

# parametrise over the ENGINE's reserved set so the test tracks it if it changes.
@pytest.mark.parametrize("reserved", list(_RESERVED_COLUMNS))
def test_binding_to_reserved_column_rejected(reserved):
    with pytest.raises(ProvenanceError):
        Binding(reserved, "BOUND", Evidence(column=reserved))


# --- F8: Evidence serialisation is JSON/YAML-safe (tuple -> list) ----------

def test_evidence_to_dict_tuple_to_list_and_drops_none():
    ev = Evidence(candidates=("a", "b"), chosen="a", note="picked a")
    d = ev.to_dict()
    assert d["candidates"] == ["a", "b"]
    assert isinstance(d["candidates"], list)
    assert "quote" not in d  # None fields dropped


# --- L2: a list of candidates is coerced to a tuple (frozen + hashable) ----

def test_evidence_list_candidates_coerced_to_tuple_and_hashable():
    ev = Evidence(candidates=["a", "b"], chosen="a", note="x")
    assert ev.candidates == ("a", "b")
    assert isinstance(ev.candidates, tuple)
    hash(ev)  # must not raise (would raise for a list field)
