"""
Mechanism-library verification (KPP schema pilot, spec §6). Proves, for every authored entry:

  * schema validity (required keys; sources non-empty; allowed_templates all exist);
  * VERIFIED is reproducible — each source quote RE-LOCATES at its recorded extractor/level/page
    (frozen canonical text or a PyMuPDF extraction), so "VERIFIED" is never an unbacked label;
  * survey sources carry `chains_to` (the survey-chaining rule — a survey must name the primary
    it cites);
  * every entry is REACHABLE per the census — its (allowed_template x required-input variable)
    space intersects prereg/reachability_census.md's executable tuples (authored backwards).

Source PDFs live in papers/pdf/ (untracked); tests skip gracefully when a source file is absent.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.librarian.config.canonical_text import load_canonical_text  # noqa: E402
from agents.librarian.config.locate import locate_quote  # noqa: E402

LIB = REPO_ROOT / "prereg" / "mechanism_library"
TEMPLATES = REPO_ROOT / "prereg" / "templates"
CENSUS_TEMPLATES = {p.stem for p in TEMPLATES.glob("*.yaml")}

ENTRIES = sorted(LIB.glob("mech_*.yaml"))
VAR_FAMILIES = yaml.safe_load((LIB / "variable_families.yaml").read_text())["variable_families"]


def _load(p: Path) -> dict:
    return yaml.safe_load(p.read_text())


def test_at_least_the_pilot_entries_exist():
    assert len(ENTRIES) >= 3, "KPP schema pilot expects >= 3 mechanism entries"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_entry_schema_valid(path):
    m = _load(path)
    for key in ("mechanism_id", "version", "title", "claim", "sources", "applicability",
                "allowed_templates", "forbidden_uses"):
        assert key in m, f"{path.name} missing {key}"
    assert m["sources"], "at least one source required"
    assert m["applicability"]["strategy_families"], "strategy_families required"
    assert m["applicability"]["required_inputs"], "required_inputs required"
    for tid in m["allowed_templates"]:
        assert tid in CENSUS_TEMPLATES, f"{path.name} allows unknown template {tid}"
    for ri in m["applicability"]["required_inputs"]:
        assert ri["variable_family"] in VAR_FAMILIES, f"unknown variable_family {ri['variable_family']}"
        assert ri["timing_rule"] == "observed_before_formation"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_every_source_quote_relocates(path):
    m = _load(path)
    for src in m["sources"]:
        assert src["status"] == "VERIFIED"
        v = src["verification"]
        text = REPO_ROOT / v["text"]
        if not text.exists():
            pytest.skip(f"source text absent: {v['text']}")
        quote = src["quote"]
        if v["extractor"] == "frozen_canonical":
            ct = load_canonical_text(text)
            loc = ct.locate(quote)                       # uses the text's L1 ladder
            assert loc is not None, f"{path.name} {src['source_id']}: quote did not locate"
            assert loc.page == v["page"], f"{path.name} {src['source_id']}: page {loc.page} != {v['page']}"
        elif v["extractor"] == "pymupdf":
            fitz = pytest.importorskip("fitz")
            pages = tuple(p.get_text() for p in fitz.open(str(text)))
            r = locate_quote(pages, quote, v["locate_level"])
            assert r.matched, f"{path.name} {src['source_id']}: quote did not locate"
            assert r.page == v["page"], f"{path.name} {src['source_id']}: page {r.page} != {v['page']}"
        else:
            pytest.fail(f"unknown extractor {v['extractor']}")


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_survey_sources_chain_to_a_primary(path):
    m = _load(path)
    for src in m["sources"]:
        if src["kind"] == "survey":
            assert "chains_to" in src, f"{path.name} {src['source_id']}: survey source must chain_to a primary"
            assert src["chains_to"].get("source_id"), "chains_to needs a source_id"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_entry_is_reachable_per_census(path):
    # Authored-backwards check: for at least one allowed_template, at least one variable in each
    # required_input's family appears in that template's conditioning_variable enum.
    m = _load(path)
    for ri in m["applicability"]["required_inputs"]:
        fam_vars = set(VAR_FAMILIES[ri["variable_family"]])
        reachable = False
        for tid in m["allowed_templates"]:
            tmpl = _load(TEMPLATES / f"{tid}.yaml")
            enum = set(tmpl["permitted_fields"]["conditioning_variable"]["allowed"])
            if fam_vars & enum:
                reachable = True
                break
        assert reachable, f"{path.name}: required_input {ri['variable_family']} unreachable by any allowed_template"


def test_report_distinct_mechanism_count(capsys):
    # Informational: the >= 8-DISTINCT-mechanism gate (spec §6) is measured on titles/claims,
    # not entry count. Here we just surface the count for the pilot.
    titles = {(_load(p)["title"]).strip().lower() for p in ENTRIES}
    print(f"\ndistinct mechanism titles: {len(titles)} across {len(ENTRIES)} entries")
    assert len(titles) == len(ENTRIES), "pilot entries must be conceptually distinct (no duplicate titles)"
