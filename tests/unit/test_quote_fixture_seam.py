"""
The quote-fixture loading seam: graceful fallback whether or not the fixtures
directory exists ([P1]).

RQ1 scoring runs against human-authored quote fixtures (D34) that do not exist
yet -- the bake-off has not run, and authoring them is out of code's scope. The
seam (``load_quote_fixtures``) must therefore be green *today*: an absent fixture
directory is the safe blank state (P1), returning ``[]``, never an error. And it
must load fixtures the day a human lands them, with no code change at the seam.

These tests exercise both halves without authoring any real fixture in the repo:
the absent-directory path against the real (absent) default location, and the
present-directory path against a ``tmp_path`` we populate in-test only.
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


# --- the absent-directory blank state (today) -----------------------------------

def test_default_fixture_dir_is_absent_today():
    # premise of the seam: the bake-off has not run, so the real fixtures dir is
    # absent. If a human lands it, this test flips -- a deliberate tripwire.
    assert not (REPO_ROOT / DEFAULT_FIXTURE_DIR).exists(), (
        f"{DEFAULT_FIXTURE_DIR} now exists -- the seam's absent-state premise no "
        "longer holds; update the tripwire and confirm real fixtures are intended"
    )


def test_load_from_absent_default_returns_empty(monkeypatch):
    # run from repo root so the relative default path resolves to the (absent) dir
    monkeypatch.chdir(REPO_ROOT)
    assert load_quote_fixtures() == []


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
