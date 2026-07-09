"""
Unit tests for the enumeration stage (build brief §5.2, D20).

Construction object shape; the grid rule; strategy-vs-auxiliary; dual-model
agreement passes; disagreement -> ``EnumerationDisagreement`` review event. All
offline against the Cluster-2 stub.
"""

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.pipeline.failures import EnumerationDisagreement
from agents.librarian.pipeline.lister import (
    AUXILIARY,
    STRATEGY,
    Construction,
    EnumerationResult,
    GridInfo,
    enumerate_constructions,
    load_gold_list,
)

from _librarian_pipeline_fixtures import (
    Q_MONTHLY,
    Q_RANKED,
    Q_VALUE_WEIGHTED,
    load_stub,
)


@pytest.fixture
def stub():
    return load_stub()


# --- Construction / GridInfo shape ------------------------------------------

def test_construction_shape_and_class():
    c = Construction(name="Momentum", quote=Q_RANKED, cls=STRATEGY)
    assert c.is_strategy
    assert c.grid.is_grid is False
    d = c.to_dict()
    assert d["name"] == "Momentum" and d["cls"] == "strategy"


def test_construction_rejects_bad_class():
    with pytest.raises(LibrarianSchemaError):
        Construction(name="x", quote=Q_RANKED, cls="not_a_class")


def test_grid_requires_headline_cell_when_is_grid():
    # D20: a grid construction must name its headline cell.
    with pytest.raises(LibrarianSchemaError):
        GridInfo(is_grid=True)
    g = GridInfo(is_grid=True, headline_cell="6m", cells_noted=("3m", "9m", "12m"))
    assert g.headline_cell == "6m" and "9m" in g.cells_noted


# --- dual-model agreement ---------------------------------------------------

def test_agreement_ships_relocated_constructions(stub):
    a = (Construction("Momentum", Q_RANKED, STRATEGY),
         Construction("Market", Q_VALUE_WEIGHTED, AUXILIARY))
    b = (Construction("Market", Q_VALUE_WEIGHTED, AUXILIARY),  # order-invariant
         Construction("Momentum", Q_RANKED, STRATEGY))
    res = enumerate_constructions(stub, a, b, paper_id="SYNTH-0001")
    assert res.agreed
    assert {c.name for c in res.constructions} == {"Momentum", "Market"}
    # strategies / auxiliaries partition the list (D20).
    assert {c.name for c in res.strategies} == {"Momentum"}
    assert {c.name for c in res.auxiliaries} == {"Market"}
    # each shipped construction's quote was re-located against the stub.
    mom = next(c for c in res.constructions if c.name == "Momentum")
    assert mom.locator is not None and mom.locator.page == 0


def test_grid_metadata_preserved_through_agreement(stub):
    grid = GridInfo(is_grid=True, headline_cell="6m", cells_noted=("3m", "12m"))
    a = (Construction("Momentum", Q_RANKED, STRATEGY, grid=grid),)
    b = (Construction("Momentum", Q_RANKED, STRATEGY, grid=grid),)
    res = enumerate_constructions(stub, a, b, paper_id="P")
    assert res.agreed
    shipped = res.constructions[0]
    assert shipped.grid.is_grid and shipped.grid.headline_cell == "6m"


# --- disagreement -----------------------------------------------------------

def test_name_set_disagreement_routes_to_review(stub):
    a = (Construction("Momentum", Q_RANKED, STRATEGY),)
    b = (Construction("Momentum", Q_RANKED, STRATEGY),
         Construction("Value", Q_VALUE_WEIGHTED, STRATEGY))
    res = enumerate_constructions(stub, a, b, paper_id="P")
    assert not res.agreed
    assert isinstance(res.disagreement, EnumerationDisagreement)
    assert res.disagreement.routing == "review"
    assert res.disagreement.only_model_b == ("Value",)
    assert res.constructions == ()


def test_class_conflict_routes_to_review(stub):
    # same name, different class -> disagreement (D20).
    a = (Construction("MKT", Q_VALUE_WEIGHTED, STRATEGY),)
    b = (Construction("MKT", Q_VALUE_WEIGHTED, AUXILIARY),)
    res = enumerate_constructions(stub, a, b, paper_id="P")
    assert not res.agreed
    assert "class conflict" in res.disagreement.detail


def test_enumeration_result_disagreement_has_no_constructions():
    with pytest.raises(LibrarianSchemaError):
        EnumerationResult(
            paper_id="P",
            constructions=(Construction("x", Q_MONTHLY, STRATEGY),),
            disagreement=EnumerationDisagreement(paper_id="P", detail="d"),
        )


# --- gold list --------------------------------------------------------------

def test_load_gold_list(tmp_path):
    gold = tmp_path / "gold.yaml"
    gold.write_text(
        "gold: true\n"
        "paper_id: BBW-2019\n"
        "constructions:\n"
        "  - name: Momentum\n"
        "    quote: ranked each month\n"
        "    class: strategy\n"
        "    grid: {is_grid: true, headline_cell: '6m', cells_noted: ['3m']}\n"
        "  - name: MKT\n"
        "    quote: value-weighted\n"
        "    class: auxiliary\n",
        encoding="utf-8",
    )
    res = load_gold_list(gold)
    assert res.agreed and res.paper_id == "BBW-2019"
    assert {c.name for c in res.strategies} == {"Momentum"}
    assert res.constructions[0].grid.headline_cell == "6m"


def test_load_gold_list_requires_gold_flag(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("paper_id: P\nconstructions: []\n", encoding="utf-8")
    with pytest.raises(LibrarianSchemaError):
        load_gold_list(bad)
