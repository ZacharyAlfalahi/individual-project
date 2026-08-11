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
from agents.quant.config.provenance import ProvenanceError

from agents.librarian.config import CanonicalText, load_canonical_text
from agents.librarian.config.locate import locate_quote
from agents.librarian.config.normalise import normalise
from agents.librarian.errors import CanonicalTextNotFrozenError, LibrarianSchemaError

_REPO = Path(__file__).resolve().parent.parent.parent
_STUB_PATH = _REPO / "agents/librarian/fixtures/canonical_text.stub.yaml"
_KPP_FROZEN = _REPO / "evaluation/canonical_texts/kpp_2023.frozen.yaml"


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


def test_locate_returns_faithful_cross_page_locator():
    # A quote straddling a page seam matches via the adjacent-pair fallback. Before
    # WS-1 locate() declined it (returned None -> routed to review/UNKNOWN); it now
    # returns a faithful cross-page Locator (end_page == page + 1) whose span indexes
    # the joined pair and round-trips byte-exactly via slice_text. WS-1 (fixes the 3
    # page_break misses in parser_bakeoff_report.md).
    ct = _frozen(pages=("the measure begins on this", "next page and finishes here"))
    quote = "begins on this next page and finishes"
    res = locate_quote(ct.pages, quote, "L1")
    assert res.matched and res.used_cross_page          # the matcher primitive locates it
    loc = ct.locate(quote, level="L1")
    assert loc is not None
    assert loc.end_page == loc.page + 1                 # a faithful cross-page span
    assert ct.slice_text(loc, level="L1") == normalise(quote, "L1")  # round-trips


def test_cross_page_locator_to_dict_and_guard():
    # Additive end_page: single-page locators stay byte-identical (no end_page key);
    # a cross-page locator surfaces end_page and must span exactly one seam. WS-1.
    single = Locator(page=2, char_start=1, char_end=4)
    assert single.to_dict() == {"page": 2, "char_start": 1, "char_end": 4}
    cross = Locator(page=2, char_start=1, char_end=4, end_page=3)
    assert cross.to_dict() == {"page": 2, "char_start": 1, "char_end": 4, "end_page": 3}
    with pytest.raises(ProvenanceError):
        Locator(page=2, char_start=1, char_end=4, end_page=5)  # not page + 1


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


# --- KPP (schema v1.2): the frozen fitted-model paper + dense Table A.I ------

def test_kpp_frozen_text_loads_and_is_frozen():
    ct = load_canonical_text(_KPP_FROZEN)
    assert ct.is_frozen
    ct.require_frozen()  # does not raise
    assert ct.source_sha256 == (
        "5e5399dfae90d0e637c38e577b15ec2ae0a242e15e8d3fc3301585803031da7f"
    )
    assert len(ct.pages) == 42


def test_kpp_dense_table_ai_substring_locates_at_l1():
    # The dense Appendix-A characteristic-definition region must be locatable at L1
    # (this is what makes the 29-instrument gold authorable) -- exact substring,
    # no cross-page fallback.
    ct = load_canonical_text(_KPP_FROZEN)
    quote = "Value-at-risk is the second lowest credit excess return over the past"
    m = locate_quote(ct.pages, quote, "L1")
    assert m.matched
    assert not m.used_cross_page
