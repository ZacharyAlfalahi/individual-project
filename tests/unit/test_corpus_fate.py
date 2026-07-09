"""
The D21 corpus walk, as executable fixtures.

D21 walked every project paper through the librarian's paper-kind gates and
decided each paper's fate. ``agents/librarian/corpus_fate.py`` encodes that walk
as typed data; this test asserts the walk holds:

  (a) *completeness* -- FATE_TABLE covers exactly the D21 paper list, no missing,
      no extras (the D21 list is hard-coded here as the independent oracle);
  (b) every fate is in the closed enum;
  (c) the *specific* fates match D21 (BBW -> sort, KPP -> refuse_family_unsupported,
      HXZ -> refuse_asset_class, BPW -> refuse_no_strategy, ...);
  (d) the refusal fates partition into the RQ2 refusal typology (D21-F4).

The table is a specification/fixture, NOT a live router -- so these are
expected-answer assertions, not routing-behaviour tests.
"""

from __future__ import annotations

import pytest

from agents.librarian import corpus_fate as cf
from agents.librarian.corpus_fate import (
    FATE_TABLE,
    FATES,
    NON_REFUSAL_FATES,
    REFUSAL_FATES,
    CorpusFate,
    all_fates,
    fate_of,
    papers,
    refusals,
)
from agents.librarian.errors import LibrarianSchemaError

# ---------------------------------------------------------------------------
# The independent D21 oracle: the exact papers D21 names and the fate each got.
# Hard-coded here (NOT imported from corpus_fate) so completeness is a genuine
# cross-check of the module against the decision log, not a tautology.
# ---------------------------------------------------------------------------

D21_EXPECTED: dict[str, str] = {
    # sort family -- the strategies the pipeline replicates
    "BBW 2019": "sort",
    "Momentum in Corporate Bonds": "sort",
    "BBW 2021": "sort",
    # pass gates, excluded by curation (D21-F3)
    "DRR 2023": "pass_gates_curation_excluded",
    "DRR 2026": "pass_gates_curation_excluded",
    # refuse: family unsupported (IPCA / DNN)
    "KPP": "refuse_family_unsupported",
    "Duraj-Giesecke": "refuse_family_unsupported",
    # refuse: asset class (equity)
    "HXZ": "refuse_asset_class",
    "KPJ": "refuse_asset_class",
    # refuse: no strategy
    "HLZ": "refuse_no_strategy",
    "Giglio-Kelly-Xiu survey": "refuse_no_strategy",
    "Glasserman-Lin": "refuse_no_strategy",
    "AI Scientist": "refuse_no_strategy",
    "AlphaAgent": "refuse_no_strategy",
    "BPW": "refuse_no_strategy",
    "DPZ": "refuse_no_strategy",
}


# --- (a) completeness: exact cover, no missing, no extras ------------------------

def test_fate_table_covers_the_d21_list_exactly():
    got = set(FATE_TABLE)
    expected = set(D21_EXPECTED)
    missing = expected - got
    extra = got - expected
    assert not missing, f"D21 papers with no fate in FATE_TABLE: {sorted(missing)}"
    assert not extra, f"FATE_TABLE has papers D21 never walked: {sorted(extra)}"


def test_papers_helper_matches_the_table_keys():
    assert set(papers()) == set(FATE_TABLE)
    # walk order is preserved and deduplicated
    assert list(papers()) == [row.paper for row in all_fates()]
    assert len(papers()) == len(set(papers()))


def test_all_fates_has_one_row_per_paper():
    assert len(all_fates()) == len(FATE_TABLE) == len(D21_EXPECTED)


# --- (b) every fate is in the closed enum ---------------------------------------

def test_every_fate_is_in_the_closed_enum():
    for row in all_fates():
        assert row.fate in FATES, f"{row.paper}: fate {row.fate!r} off the closed enum"


def test_closed_enum_partitions_into_refusal_and_non_refusal():
    # the two families are disjoint and together are the whole enum
    assert REFUSAL_FATES.isdisjoint(NON_REFUSAL_FATES)
    assert REFUSAL_FATES | NON_REFUSAL_FATES == FATES


def test_constructing_an_off_enum_fate_is_a_build_error():
    with pytest.raises(LibrarianSchemaError):
        CorpusFate("Made Up", "route_to_the_moon", "not a real fate")


def test_empty_paper_or_reason_is_a_build_error():
    with pytest.raises(LibrarianSchemaError):
        CorpusFate("", "sort", "blank paper id")
    with pytest.raises(LibrarianSchemaError):
        CorpusFate("BBW 2019", "sort", "   ")


# --- (c) the specific fates match D21 -------------------------------------------

@pytest.mark.parametrize(("paper", "expected_fate"), sorted(D21_EXPECTED.items()))
def test_each_paper_has_its_d21_fate(paper, expected_fate):
    assert fate_of(paper).fate == expected_fate


def test_spot_check_representative_fates():
    # one representative per fate, spelled out (the brief's named examples)
    assert fate_of("BBW 2019").fate == "sort"
    assert fate_of("KPP").fate == "refuse_family_unsupported"
    assert fate_of("HXZ").fate == "refuse_asset_class"
    assert fate_of("BPW").fate == "refuse_no_strategy"
    assert fate_of("DRR 2023").fate == "pass_gates_curation_excluded"


def test_fate_of_unlisted_paper_raises():
    with pytest.raises(LibrarianSchemaError):
        fate_of("Some Paper Nobody Walked")


def test_every_fate_carries_a_non_empty_reason():
    for row in all_fates():
        assert row.reason.strip(), f"{row.paper}: empty reason"


# --- (d) the refusal fates partition into the RQ2 refusal typology ---------------

def test_refusals_are_exactly_the_refuse_star_fates():
    refusal_papers = {row.paper for row in refusals()}
    expected = {p for p, f in D21_EXPECTED.items() if f in REFUSAL_FATES}
    assert refusal_papers == expected


def test_rq2_refusal_typology_has_all_three_classes_populated():
    # the RQ2 preview must exercise every refusal class, else the typology is
    # untested on this corpus
    by_class: dict[str, list[str]] = {f: [] for f in REFUSAL_FATES}
    for row in refusals():
        by_class[row.fate].append(row.paper)
    for fate, members in by_class.items():
        assert members, f"refusal class {fate!r} has no member paper in the corpus"
    # the three refusal classes are exactly the RQ2 typology
    assert set(by_class) == set(REFUSAL_FATES)


def test_is_refusal_flag_agrees_with_the_fate():
    for row in all_fates():
        assert row.is_refusal == (row.fate in REFUSAL_FATES)


def test_non_refusal_papers_are_the_gate_clearers():
    non_refusal = {row.paper for row in all_fates() if not row.is_refusal}
    expected = {p for p, f in D21_EXPECTED.items() if f in NON_REFUSAL_FATES}
    assert non_refusal == expected


# --- module-level guards --------------------------------------------------------

def test_to_dict_roundtrips_the_row():
    row = fate_of("BBW 2019")
    assert row.to_dict() == {
        "paper": "BBW 2019",
        "fate": "sort",
        "reason": row.reason,
    }


def test_fate_table_and_rows_are_the_same_objects():
    # FATE_TABLE is built from the same rows all_fates() returns (no drift)
    for row in all_fates():
        assert FATE_TABLE[row.paper] is row


def test_module_exposes_the_documented_api():
    for name in ("FATE_TABLE", "fate_of", "all_fates"):
        assert hasattr(cf, name)
