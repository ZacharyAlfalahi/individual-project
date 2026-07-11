"""
The quote-fixture loading seam: graceful fallback whether or not the fixtures
directory exists ([P1]).

RQ1 scoring runs against human-authored quote fixtures (D34). These have now been
authored and moved into the default dir (orientation A#1), so the seam
(``load_quote_fixtures``) must load them from the default location with no code
change. The blank-state guarantee still holds and still matters: an absent (or
empty, or non-directory) fixture path returns ``[]``, never an error.

These tests exercise both halves: the present-directory path against the REAL
default location (the three corpus fixtures now on disk), and the absent/blank-state
path against a ``tmp_path`` we populate in-test only.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.librarian.errors import LibrarianSchemaError
from agents.librarian.evaluation import (
    DEFAULT_FIXTURE_DIR,
    QuoteFixture,
    load_quote_fixtures,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --- the present-directory state: real fixtures have landed ---------------------
# Tripwire (flipped from the old absent-state premise, A#1): the three corpus quote
# fixtures now live at the default location. This guards against their accidental
# removal or an unexpected fixture set -- if it fires, confirm the change is intended.

def test_default_fixture_dir_has_the_three_corpus_papers():
    d = REPO_ROOT / DEFAULT_FIXTURE_DIR
    assert d.is_dir(), (
        f"{DEFAULT_FIXTURE_DIR} is missing -- the real quote fixtures were removed; "
        "the seam depends on them for RQ1 scoring"
    )
    names = sorted(p.name for p in d.glob("*.yaml"))
    assert names == ["bbw_2019.yaml", "bpw_2011.yaml", "kpp_2023.yaml"], (
        f"unexpected fixture set at {DEFAULT_FIXTURE_DIR}: {names}"
    )


def test_load_from_default_returns_the_three_papers(monkeypatch):
    # run from repo root so the relative default path resolves to the fixtures dir
    monkeypatch.chdir(REPO_ROOT)
    fixtures = load_quote_fixtures()
    assert all(isinstance(f, QuoteFixture) for f in fixtures)
    papers = sorted(f.content["paper"] for f in fixtures)
    assert papers == ["BBW_2019", "BPW_2011", "KPP_2023"]


def test_load_from_explicitly_absent_dir_returns_empty(tmp_path):
    missing = tmp_path / "definitely_not_here"
    assert not missing.exists()
    assert load_quote_fixtures(missing) == []


def test_load_when_dir_is_a_file_returns_empty(tmp_path):
    # a path that exists but is not a directory is still the blank state, not a crash
    not_a_dir = tmp_path / "quote_fixtures.yaml"
    not_a_dir.write_text("x: 1\n", encoding="utf-8")
    assert load_quote_fixtures(not_a_dir) == []


def test_empty_present_dir_returns_empty(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    assert load_quote_fixtures(d) == []


# --- the present-directory path (simulated via tmp_path only) --------------------

def test_loads_yaml_fixtures_when_present(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "bbw2019.yaml").write_text(
        "paper_id: BBW-2019\nfield: n_groups\nvalue: 5\n", encoding="utf-8"
    )
    (d / "momentum.yaml").write_text(
        "paper_id: MOM\nfield: signal\nvalue: past_6m_cumulative_return\n",
        encoding="utf-8",
    )
    fixtures = load_quote_fixtures(d)
    assert len(fixtures) == 2
    assert all(isinstance(f, QuoteFixture) for f in fixtures)
    # deterministic sorted-path order
    assert [Path(f.source_path).name for f in fixtures] == [
        "bbw2019.yaml",
        "momentum.yaml",
    ]
    assert fixtures[0].content["paper_id"] == "BBW-2019"
    assert fixtures[1].content["value"] == "past_6m_cumulative_return"


def test_non_yaml_files_are_ignored(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "keep.yaml").write_text("a: 1\n", encoding="utf-8")
    (d / "README.md").write_text("# not a fixture\n", encoding="utf-8")
    (d / "notes.txt").write_text("scratch\n", encoding="utf-8")
    fixtures = load_quote_fixtures(d)
    assert len(fixtures) == 1
    assert Path(fixtures[0].source_path).name == "keep.yaml"


def test_empty_yaml_doc_is_skipped_not_loaded(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "empty.yaml").write_text("", encoding="utf-8")
    (d / "real.yaml").write_text("a: 1\n", encoding="utf-8")
    fixtures = load_quote_fixtures(d)
    assert len(fixtures) == 1
    assert Path(fixtures[0].source_path).name == "real.yaml"


# --- a present-but-broken fixture is a build error (distinct from blank) ---------

def test_malformed_yaml_fixture_raises(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "broken.yaml").write_text("a: [1, 2\n  b: {", encoding="utf-8")
    with pytest.raises(LibrarianSchemaError):
        load_quote_fixtures(d)


def test_non_mapping_top_level_fixture_raises(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "list.yaml").write_text("- 1\n- 2\n- 3\n", encoding="utf-8")
    with pytest.raises(LibrarianSchemaError):
        load_quote_fixtures(d)


def test_accepts_str_or_path_dir(tmp_path):
    d = tmp_path / "quote_fixtures"
    d.mkdir()
    (d / "f.yaml").write_text("a: 1\n", encoding="utf-8")
    from_str = load_quote_fixtures(str(d))
    from_path = load_quote_fixtures(d)
    assert len(from_str) == len(from_path) == 1
