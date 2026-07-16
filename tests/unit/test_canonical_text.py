"""
Unit tests for CanonicalText + the status gate (build brief §5.1).

Covers: the stub loads with ``status == "stub"``; ``require_frozen`` RAISES on the
stub and PASSES on a ``status: frozen`` fixture built in-test; and ``locate`` finds
a substring present in ``pages`` (returning a byte-correct L0 Locator) and returns
None for an absent one.
"""

from pathlib import Path

import pytest

from agents.quant.config import Locator

from agents.librarian.config import CanonicalText, load_canonical_text
from agents.librarian.errors import CanonicalTextNotFrozenError, LibrarianSchemaError

_STUB_PATH = Path(__file__).resolve().parent.parent.parent / (
    "agents/librarian/fixtures/canonical_text.stub.yaml"
)


@pytest.fixture(scope="module")
def stub():
    return load_canonical_text(_STUB_PATH)


def _frozen(pages=("alpha beta gamma", "delta epsilon")):
    """A minimal status: frozen CanonicalText built in-test (never a real paper)."""
    return CanonicalText(
        source_pdf="synthetic/frozen.pdf",
        source_sha256="ff" * 32,
        parser={"name": "stub-parser", "version": "0.0.0"},
        normalisation={"ladder_level": "L0", "rules": []},
        pages=pages,
        status="frozen",
    )


# --- the stub loads ---------------------------------------------------------

def test_stub_loads_with_stub_status(stub):
    assert stub.status == "stub"
    assert not stub.is_frozen
    assert len(stub.pages) == 2
    assert stub.parser["name"] == "stub-parser"


# --- the status gate --------------------------------------------------------

def test_require_frozen_raises_on_stub(stub):
    with pytest.raises(CanonicalTextNotFrozenError):
        stub.require_frozen()
    # subclasses LibrarianSchemaError -> broad ValueError handlers keep working.
    with pytest.raises(LibrarianSchemaError):
        stub.require_frozen()


def test_require_frozen_passes_on_frozen():
    ct = _frozen()
    assert ct.require_frozen() is ct
    assert ct.is_frozen


# --- locate (L0 exact substring) --------------------------------------------

def test_locate_finds_present_substring(stub):
    loc = stub.locate("quintiles")
    assert isinstance(loc, Locator)
    # byte-correct span into the located page.
    assert stub.pages[loc.page][loc.char_start:loc.char_end] == "quintiles"


def test_locate_returns_none_for_absent_substring(stub):
    assert stub.locate("this phrase is definitely not in the stub") is None


def test_locate_searches_across_pages():
    ct = _frozen(pages=("first page only", "second page has the TARGET token"))
    loc = ct.locate("TARGET")
    assert loc is not None
    assert loc.page == 1
    assert ct.pages[1][loc.char_start:loc.char_end] == "TARGET"


def test_locate_rejects_empty_quote(stub):
    with pytest.raises(LibrarianSchemaError):
        stub.locate("")


# --- locate with a ladder level (parser bake-off v2) ------------------------

def test_locate_level_l1_normalises_whitespace():
    ct = _frozen(pages=("the long short\nfactor series",))
    # the default (L0) cannot bridge the newline...
    assert ct.locate("long short factor series") is None
    # ...but L1 collapses whitespace and locates.
    loc = ct.locate("long short factor series", level="L1")
    assert isinstance(loc, Locator)
    assert loc.page == 0


def test_locate_default_follows_frozen_ladder_level():
    # locate() must DEFAULT to the text's own recipe ladder (L1 for the real frozen
    # papers), not L0. The L0 default was a footgun: real, L1-normalised quotes fail an
    # exact L0 substring, so the D9 quote gate returned spurious quote_match_failure on
    # every live extraction (RealClient BBW smoke, 2026-07-16).
    ct = CanonicalText(
        source_pdf="synthetic/frozen_l1.pdf",
        source_sha256="ab" * 32,
        parser={"name": "stub-parser", "version": "0.0.0"},
        normalisation={"ladder_level": "L1", "rules": ["whitespace_collapse"]},
        pages=("the long short\nfactor series",),  # newline only bridged at L1
        status="frozen",
    )
    assert ct.locate("long short factor series") is not None      # default -> L1
    assert ct.locate("long short factor series", level="L0") is None  # explicit L0 still can't


def test_locate_level_l2_dehyphenates_fold():
    ct = _frozen(pages=("bonds sorted into quintiles at formation",))
    # a fixture-style hyphen-space fold is bridged only at L2.
    assert ct.locate("quin- tiles", level="L1") is None
    assert ct.locate("quin- tiles", level="L2") is not None


def test_locate_declines_cross_page_matches():
    # A quote straddling a page boundary matches via the adjacent-pair fallback
    # (locate_quote reports it), but CanonicalText.locate returns None rather than
    # stamp a malformed single-page Locator (offsets index the joined pair). HIGH-1.
    from agents.librarian.config.locate import locate_quote

    ct = _frozen(pages=("the measure begins on this", "next page and finishes here"))
    quote = "begins on this next page and finishes"
    res = locate_quote(ct.pages, quote, "L1")
    assert res.matched and res.used_cross_page          # the matcher primitive locates it
    assert ct.locate(quote, level="L1") is None          # but locate() declines it


# --- construction guards ----------------------------------------------------

def test_bad_status_is_build_error():
    with pytest.raises(LibrarianSchemaError):
        CanonicalText(
            source_pdf="x.pdf",
            source_sha256="ab",
            parser={"name": "p", "version": "1"},
            normalisation={"ladder_level": "L0", "rules": []},
            pages=("page",),
            status="draft",  # not in {stub, frozen}
        )


def test_missing_status_key_is_build_error(tmp_path):
    f = tmp_path / "ct.yaml"
    f.write_text(
        "source_pdf: x.pdf\n"
        "source_sha256: ab\n"
        "parser: {name: p, version: '1'}\n"
        "normalisation: {ladder_level: L0, rules: []}\n"
        "pages: ['a page']\n",  # no status
        encoding="utf-8",
    )
    with pytest.raises(LibrarianSchemaError):
        load_canonical_text(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(LibrarianSchemaError):
        load_canonical_text(tmp_path / "nope.yaml")


def test_empty_pages_is_build_error():
    with pytest.raises(LibrarianSchemaError):
        CanonicalText(
            source_pdf="x.pdf",
            source_sha256="ab",
            parser={"name": "p", "version": "1"},
            normalisation={"ladder_level": "L0", "rules": []},
            pages=(),
            status="frozen",
        )
