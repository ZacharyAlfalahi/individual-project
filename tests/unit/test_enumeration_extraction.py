"""
Unit tests for live enumeration extraction (WS-3, D20).

Fully offline: the two "models" are ``FakeModelClient``s seeded with a scripted
construction list (``extract_enumeration``); the canonical text is the frozen twin
of the Cluster-2 stub. No live model, no network. Covers:

  (a) agreement   -- both models return the same construction set -> agreed, the
                     shipped constructions are relocated against the frozen text;
  (b) disagreement-- different name sets -> not agreed, an EnumerationDisagreement
                     event is populated and no constructions ship;
  (c) pure parse  -- ``_constructions_from_parsed`` maps a valid row to a
                     Construction and DROPS malformed rows (blank name/quote,
                     off-menu class, invalid grid, non-dict rows).

Plus a smoke that the three hand-authored gold enumeration recipes load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.librarian.pipeline import (
    Construction,
    FakeModelClient,
    enumerate_constructions,
    load_gold_list,
)
from agents.librarian.pipeline.lister import AUXILIARY, STRATEGY
from agents.librarian.pipeline.real_client import _constructions_from_parsed

from _librarian_pipeline_fixtures import Q_RANKED, Q_VALUE_WEIGHTED, frozen_stub

_GOLD_DIR = (
    Path(__file__).resolve().parent.parent.parent / "evaluation" / "gold_specs"
)


@pytest.fixture
def frozen_ct():
    return frozen_stub()


# --- (a) agreement ----------------------------------------------------------

def test_agreement_ships_relocated_constructions(frozen_ct):
    seed = (
        Construction("Momentum", Q_RANKED, STRATEGY),
        Construction("Market", Q_VALUE_WEIGHTED, AUXILIARY),
    )
    model_a = FakeModelClient("fake-a", enumeration=seed)
    # model_b returns the SAME set in a different order (agreement is set-based).
    model_b = FakeModelClient("fake-b", enumeration=tuple(reversed(seed)))

    list_a = model_a.extract_enumeration(frozen_ct)
    list_b = model_b.extract_enumeration(frozen_ct)
    res = enumerate_constructions(frozen_ct, list_a, list_b, paper_id="SYNTH-0001")

    assert res.agreed
    assert res.disagreement is None
    assert {c.name for c in res.constructions} == {"Momentum", "Market"}
    assert {c.name for c in res.strategies} == {"Momentum"}
    assert {c.name for c in res.auxiliaries} == {"Market"}
    # each shipped construction's quote was re-located against the frozen text.
    mom = next(c for c in res.constructions if c.name == "Momentum")
    assert mom.locator is not None and mom.locator.page == 0


# --- (b) disagreement -------------------------------------------------------

def test_name_set_disagreement_routes_to_review(frozen_ct):
    model_a = FakeModelClient(
        "fake-a", enumeration=(Construction("Momentum", Q_RANKED, STRATEGY),)
    )
    model_b = FakeModelClient(
        "fake-b", enumeration=(Construction("Value", Q_VALUE_WEIGHTED, STRATEGY),)
    )

    list_a = model_a.extract_enumeration(frozen_ct)
    list_b = model_b.extract_enumeration(frozen_ct)
    res = enumerate_constructions(frozen_ct, list_a, list_b, paper_id="P")

    assert not res.agreed
    assert res.disagreement is not None
    assert res.disagreement.routing == "review"
    assert res.disagreement.only_model_a == ("Momentum",)
    assert res.disagreement.only_model_b == ("Value",)
    assert res.constructions == ()


def test_empty_enumeration_defaults_to_no_constructions(frozen_ct):
    # A FakeModelClient with no seed returns () -- the empty-list degradation.
    model = FakeModelClient("fake-a")
    assert model.extract_enumeration(frozen_ct) == ()


# --- (c) pure parse ---------------------------------------------------------

def test_constructions_from_parsed_keeps_valid_drops_malformed():
    parsed = {
        "constructions": [
            {"name": "Momentum", "quote": "ranked each month", "class": "strategy"},  # valid
            {"name": "  ", "quote": "q", "class": "strategy"},          # blank name -> drop
            {"name": "X", "quote": "   ", "class": "strategy"},         # blank quote -> drop
            {"name": "Y", "quote": "q", "class": "not_a_class"},        # off-menu class -> drop
            {"name": "G", "quote": "q", "class": "strategy",
             "grid": {"is_grid": True}},                                # grid w/o headline -> drop
            "not a dict",                                               # non-dict row -> skip
        ]
    }
    out = _constructions_from_parsed(parsed)
    assert len(out) == 1
    only = out[0]
    assert only.name == "Momentum" and only.is_strategy
    # the producer never locates -- enumerate_constructions relocates downstream.
    assert only.locator is None


def test_constructions_from_parsed_preserves_valid_grid():
    parsed = {
        "constructions": [
            {"name": "Mom", "quote": "q", "class": "strategy",
             "grid": {"is_grid": True, "headline_cell": "6m", "cells_noted": ["3m", "12m"]}},
        ]
    }
    out = _constructions_from_parsed(parsed)
    assert len(out) == 1
    assert out[0].grid.is_grid and out[0].grid.headline_cell == "6m"
    assert out[0].grid.cells_noted == ("3m", "12m")


def test_constructions_from_parsed_empty_and_missing():
    assert _constructions_from_parsed({}) == ()
    assert _constructions_from_parsed({"constructions": None}) == ()
    assert _constructions_from_parsed({"constructions": []}) == ()


def test_constructions_from_parsed_hostile_json_never_raises():
    # WS-3 robustness: a hostile/hallucinated reply must degrade to ()
    # or drop the row, never raise -- a live run must not crash on non-conforming JSON.
    assert _constructions_from_parsed({"constructions": 5}) == ()       # non-list int
    assert _constructions_from_parsed({"constructions": True}) == ()    # non-list bool
    assert _constructions_from_parsed({"constructions": "nope"}) == ()  # non-list str
    # a non-iterable cells_noted coerces to empty rather than raising; the row survives.
    out = _constructions_from_parsed(
        {"constructions": [{"name": "M", "quote": "q", "class": "strategy",
                            "grid": {"cells_noted": 7}}]}
    )
    assert len(out) == 1 and out[0].grid.cells_noted == ()
    out2 = _constructions_from_parsed(
        {"constructions": [{"name": "M", "quote": "q", "class": "strategy",
                            "grid": {"is_grid": False, "cells_noted": True}}]}
    )
    assert len(out2) == 1 and out2[0].grid.cells_noted == ()


# --- gold recipes load (WS-3 deliverable) -----------------------------------

@pytest.mark.parametrize(
    "fname,paper_id,names",
    [
        # BBW carries three strategy rows since the 2026-09-03 instructed
        # extension (CRF + LRF transcribed from their ratified golds -- see the
        # file's authoring amendment).
        ("enum_bbw_2019.yaml", "BBW_2019", {"Downside Risk Factor (DRF)", "CRF", "LRF"}),
        ("enum_jnps_2013.yaml", "JNPS_2013", {"Six-Month Momentum (mom6)"}),
        ("enum_drr_2026.yaml", "DRR_2026", {"Short-Term Reversal (str)"}),
    ],
)
def test_gold_enumeration_recipes_load(fname, paper_id, names):
    res = load_gold_list(_GOLD_DIR / fname)
    assert res.agreed and res.paper_id == paper_id
    assert {c.name for c in res.strategies} == names
