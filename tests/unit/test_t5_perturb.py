"""
T5 STRUCTURAL perturber (evaluation/harness/perturb.py) -- unit tests.

These tests exercise the deterministic text transformation only; they NEVER run a
model or the pipeline (perturb.py cannot). They are allowed to write to pytest's
``tmp_path`` (they are tests, not the pipeline). Each test builds items in memory
via ``build_one`` / ``generate(write=False)`` so it does not depend on the
committed frozen/sheet artefacts.
"""

from __future__ import annotations

import yaml
import pytest

from agents.librarian.config.canonical_text import load_canonical_text
from agents.librarian.config.locate import locate_quote
from agents.librarian.config.normalise import normalise
from evaluation.harness import perturb
from evaluation.harness.perturb import (
    DECOYS,
    PerturbError,
    build_one,
    build_one_semantic,
    load_deletion_valid,
    load_semantic_items,
)

_NBSP = " "
_CYRILLIC_O = "о"


def _pages(frozen_text: str) -> tuple[str, ...]:
    return tuple(yaml.safe_load(frozen_text)["pages"])


# ---------------------------------------------------------------------------
# Determinism: same input -> identical bytes.
# ---------------------------------------------------------------------------

def test_build_one_is_byte_deterministic():
    a = build_one("drf", "n_groups", 1)
    b = build_one("drf", "n_groups", 1)
    assert a["frozen_text"] == b["frozen_text"]
    assert a["sheet_text"] == b["sheet_text"]


def test_build_is_byte_deterministic_across_classes():
    # One item per structural class -- same input must yield identical bytes.
    for anchor, field, cls in (
        ("drf", "n_groups", 1),
        ("mom6", "signal_lag", 2),
        ("str", "weighting_scheme", 4),
        ("mom6", "sort_kind", 7),
    ):
        x = build_one(anchor, field, cls)
        y = build_one(anchor, field, cls)
        assert x["frozen_text"] == y["frozen_text"], (anchor, field, cls)
        assert x["sheet_text"] == y["sheet_text"], (anchor, field, cls)


def test_expected_structural_item_counts():
    # Counted without the (slow) full generation: 69 STATED targets across the
    # three anchors -> classes 1/2/4; 6 deletion_valid fields -> class 7.
    total_targets = 0
    for anchor, fname in perturb.ANCHORS.items():
        raw = yaml.safe_load(
            open(f"evaluation/canonical_texts/{fname}", encoding="utf-8").read()
        )
        pages = tuple(str(p) for p in raw["pages"])
        total_targets += len(perturb.iter_targets(anchor, pages))
    assert total_targets == 69
    assert len(load_deletion_valid()) == 6
    assert total_targets * 3 + len(load_deletion_valid()) == 213


# ---------------------------------------------------------------------------
# Class 7 (deletion): the gold quote no longer locates anywhere.
# ---------------------------------------------------------------------------

def test_class7_gold_quote_no_longer_locates():
    for anchor, field in sorted(load_deletion_valid()):
        item = build_one(anchor, field, 7)
        pages = _pages(item["frozen_text"])
        sheet = yaml.safe_load(item["sheet_text"])
        quote = sheet["gold_quote"]
        # Deletion is answer-changing: the gold sentence must be gone (existence).
        assert not locate_quote(pages, quote, "L1").matched, (anchor, field)
        assert sheet["expected_outcome"] == "abstained_gold_silent"
        assert sheet["regime"] == "answer_changing"


def test_class7_refuses_field_not_in_deletion_valid():
    valid = load_deletion_valid()
    # n_groups is a heavily-restated marker -> NOT deletion_valid for any anchor.
    assert ("drf", "n_groups") not in valid
    with pytest.raises(PerturbError):
        build_one("drf", "n_groups", 7)


def test_class7_deletes_all_copies_across_pages():
    # mom6 strategy_side is stated twice -> both copies must be removed.
    item = build_one("mom6", "strategy_side", 7)
    assert "removed=2" in item["transform_id"]
    pages = _pages(item["frozen_text"])
    quote = yaml.safe_load(item["sheet_text"])["gold_quote"]
    assert not locate_quote(pages, quote, "L1").matched


# ---------------------------------------------------------------------------
# Class 4 (distractor): decoy present AND the original quote still locates.
# ---------------------------------------------------------------------------

def test_class4_decoy_present_and_original_still_locates():
    for anchor in ("drf", "mom6", "str"):
        for field in ("n_groups", "weighting_scheme", "sort_signal"):
            item = build_one(anchor, field, 4)
            pages = _pages(item["frozen_text"])
            sheet = yaml.safe_load(item["sheet_text"])
            decoy = DECOYS[anchor][field]
            assert locate_quote(pages, decoy, "L1").matched, (anchor, field, "decoy")
            assert locate_quote(pages, sheet["gold_quote"], "L1").matched, (anchor, field, "orig")
            assert sheet["regime"] == "invariant"


def test_class4_decoy_is_foreign_to_the_clean_text():
    # A decoy that already appears in the clean paper is not a distractor.
    clean = load_canonical_text("evaluation/canonical_texts/bbw_2019.frozen.yaml")
    for field, decoy in DECOYS["drf"].items():
        assert not locate_quote(clean.pages, decoy, "L1").matched, field


def test_class4_inserts_on_a_different_page_than_the_gold_quote():
    item = build_one("drf", "n_groups", 4)  # gold quote on page 14 (idx 13)
    assert "page=0" in item["transform_id"]  # decoy goes to page 0 (title page)


# ---------------------------------------------------------------------------
# Class 1 (orthographic): the span changed deterministically; the yaml still loads.
# ---------------------------------------------------------------------------

def test_class1_span_bytes_changed_with_homoglyphs(tmp_path):
    item = build_one("drf", "n_groups", 1)
    pages = _pages(item["frozen_text"])
    joined = "".join(pages)
    # The documented transform injects a NBSP and a Cyrillic 'o'.
    assert _NBSP in joined
    assert _CYRILLIC_O in joined
    # And the perturbed frozen text still loads as a frozen, non-empty canonical text.
    p = tmp_path / "c1.frozen.yaml"
    p.write_text(item["frozen_text"], encoding="utf-8")
    ct = load_canonical_text(str(p))
    assert ct.is_frozen and len(ct.pages) > 0


def test_class1_changes_only_the_evidence_page():
    clean = yaml.safe_load(
        open("evaluation/canonical_texts/bbw_2019.frozen.yaml", encoding="utf-8").read()
    )
    item = build_one("drf", "n_groups", 1)
    pages = _pages(item["frozen_text"])
    changed = [i for i, (a, b) in enumerate(zip(pages, clean["pages"])) if a != b]
    assert changed == [13]  # gold page 14 (1-indexed) == index 13


# ---------------------------------------------------------------------------
# Class 2 (layout): the layout changed but the quote is still present; yaml loads.
# ---------------------------------------------------------------------------

def test_class2_quote_still_locates_and_yaml_loads(tmp_path):
    item = build_one("mom6", "signal_lag", 2)
    pages = _pages(item["frozen_text"])
    quote = yaml.safe_load(item["sheet_text"])["gold_quote"]
    res = locate_quote(pages, quote, "L1")
    assert res.matched  # still present across the (two) pages
    p = tmp_path / "c2.frozen.yaml"
    p.write_text(item["frozen_text"], encoding="utf-8")
    ct = load_canonical_text(str(p))
    assert ct.is_frozen and len(ct.pages) == len(pages)


def test_class2_is_a_crosspage_seam_split():
    # The tail of the evidence moves to the next page; per-page match fails, the
    # cross-page fallback succeeds.
    item = build_one("drf", "formation_structure", 2)
    pages = _pages(item["frozen_text"])
    quote = yaml.safe_load(item["sheet_text"])["gold_quote"]
    res = locate_quote(pages, quote, "L1")
    assert res.matched and res.used_cross_page


# ---------------------------------------------------------------------------
# Metadata fidelity: the recipe blocks are carried through byte-identically.
# ---------------------------------------------------------------------------

def test_metadata_blocks_are_byte_identical_to_clean_anchor():
    clean_text = open(
        "evaluation/canonical_texts/bbw_2019.frozen.yaml", encoding="utf-8"
    ).read()

    def meta_region(text: str) -> str:
        lines = text.splitlines(keepends=True)
        s = next(i for i, ln in enumerate(lines) if ln.startswith("source_pdf:"))
        e = next(i for i, ln in enumerate(lines) if ln.startswith("pages:"))
        return "".join(lines[s:e])

    item = build_one("drf", "weighting_scheme", 1)
    assert meta_region(item["frozen_text"]) == meta_region(clean_text)


def test_perturbation_provenance_block_present():
    item = build_one("str", "n_groups", 4)
    raw = yaml.safe_load(item["frozen_text"])
    prov = raw["perturbation"]
    assert prov["anchor"] == "str"
    assert prov["field"] == "n_groups"
    assert prov["class"] == 4
    assert prov["base_source_sha256"] == raw["source_sha256"]
    assert prov["generator"] == "evaluation/harness/perturb.py"


# ---------------------------------------------------------------------------
# The L1 -> L0 span mapper (the hard part) round-trips on every STATED field.
# ---------------------------------------------------------------------------

def test_span_mapper_roundtrips_on_every_stated_field():
    for anchor, fname in perturb.ANCHORS.items():
        raw = yaml.safe_load(
            open(f"evaluation/canonical_texts/{fname}", encoding="utf-8").read()
        )
        pages = tuple(str(p) for p in raw["pages"])
        for tgt in perturb.iter_targets(anchor, pages):
            rt = normalise(pages[tgt.page_idx][tgt.l0_start:tgt.l0_end], "L1")
            assert rt == tgt.q_l1, (anchor, tgt.short_key)


# ===========================================================================
# SEMANTIC classes (3/5/6): hand-authored VERBATIM text, deterministic apply.
# These read the committed semantic_perturbations.yaml (the final wording)
# but build items in memory -- no model, no pipeline, no dependence on the emitted
# frozen/sheet artefacts. build_one_semantic re-loads a gold per call (~1s), so the
# whole-set assertions consume a MODULE-SCOPED fixture that builds all 44 ONCE.
# ===========================================================================

def _build_all_semantic() -> dict[str, dict]:
    """Build all 44 semantic items ONCE, reusing each anchor's resolved targets.
    Returns {stem: build_dict + 'item'}."""
    out: dict[str, dict] = {}
    items = load_semantic_items()
    for anchor in perturb.ANCHORS:
        raw, base_sha = perturb._load_raw(anchor)
        pages_l0 = tuple(str(p) for p in raw["pages"])
        targets = perturb._targets_by_key(anchor, pages_l0)
        for it in [i for i in items if i["anchor"] == anchor]:
            b = perturb.build_semantic_item(anchor, it, targets, raw, base_sha)
            out[b["stem"]] = {**b, "item": it}
    return out


@pytest.fixture(scope="module")
def semantic_built() -> dict[str, dict]:
    return _build_all_semantic()


def test_semantic_item_count_is_44():
    from collections import Counter
    items = load_semantic_items()
    assert len(items) == 44
    assert Counter(it["anchor"] for it in items) == {"drf": 11, "mom6": 18, "str": 15}
    assert Counter(int(it["class"]) for it in items) == {3: 14, 5: 15, 6: 15}


def test_semantic_build_is_byte_deterministic():
    # Per-item determinism on a representative sample (incl. the shared item).
    for anchor, field, cls in (
        ("drf", "sort_signal", 3),
        ("mom6", "n_groups", 5),
        ("drf", "long_leg+weighting_scheme", 3),
    ):
        a = build_one_semantic(anchor, field, cls)
        b = build_one_semantic(anchor, field, cls)
        assert a["frozen_text"] == b["frozen_text"], (anchor, field, cls)
        assert a["sheet_text"] == b["sheet_text"], (anchor, field, cls)


def test_all_44_semantic_items_regenerate_byte_identical(semantic_built):
    # Determinism over the FULL set: build a second time and compare bytes per item.
    second = _build_all_semantic()
    assert set(second) == set(semantic_built)
    for stem, b in semantic_built.items():
        assert (b["frozen_text"], b["sheet_text"]) == (
            second[stem]["frozen_text"], second[stem]["sheet_text"]), stem


def test_class3_paraphrase_removes_all_clean_quote_copies(semantic_built):
    n = 0
    for b in semantic_built.values():
        if int(b["item"]["class"]) != 3:
            continue
        n += 1
        pages = _pages(b["frozen_text"])
        sheet = yaml.safe_load(b["sheet_text"])
        # T5-PRE-3 P1 replace-all: the clean gold quote is gone EVERYWHERE.
        assert not locate_quote(pages, sheet["gold_quote"], "L1").matched, b["stem"]
        # ...and the paraphrase is present.
        assert locate_quote(pages, sheet["perturbed_text"], "L1").matched, b["stem"]
        assert sheet["regime"] == "invariant_degrading"
        assert sheet["expected_outcome"] == "paraphrase_correct_or_abstain"
    assert n == 14  # 3 drf + 6 mom6 + 5 str


def test_class3_replaces_both_copies_of_a_mom6_duplicate_field(semantic_built):
    # mom6 sort_signal / long_leg / weighting_scheme are stated on p9 AND p10 ->
    # replace-all must remove BOTH copies (transform reports replaced=2).
    for field in ("sort_signal", "long_leg", "weighting_scheme"):
        b = semantic_built[f"mom6__{field}__c3_paraphrase"]
        assert "replaced=2" in b["transform_id"], field
        pages = _pages(b["frozen_text"])
        gold_quote = yaml.safe_load(b["sheet_text"])["gold_quote"]
        assert not locate_quote(pages, gold_quote, "L1").matched, field


def test_class3_nonduplicate_field_replaces_exactly_one_copy(semantic_built):
    for stem in ("drf__n_groups__c3_paraphrase", "str__n_groups__c3_paraphrase",
                 "mom6__holding_period__c3_paraphrase"):
        assert "replaced=1" in semantic_built[stem]["transform_id"], stem


def test_class5_and_6_keep_original_and_add_inserted_text(semantic_built):
    n = 0
    for b in semantic_built.values():
        if int(b["item"]["class"]) not in (5, 6):
            continue
        n += 1
        pages = _pages(b["frozen_text"])
        sheet = yaml.safe_load(b["sheet_text"])
        # original STILL locates (left intact) AND the inserted sentence is present.
        assert locate_quote(pages, sheet["gold_quote"], "L1").matched, b["stem"]
        assert locate_quote(pages, sheet["perturbed_text"], "L1").matched, b["stem"]
        assert sheet["regime"] == "answer_changing"
    assert n == 30  # 15 contradiction + 15 injection


def test_class5_expected_outcome_is_abstain_only(semantic_built):
    sheet = yaml.safe_load(semantic_built["mom6__signal_lag__c5_contradiction"]["sheet_text"])
    assert sheet["expected_outcome"] == "contradiction_abstain"
    assert sheet["pass_set"] == ["abstained"]


def test_class6_sheet_carries_injected_value_and_vector(semantic_built):
    sheet = yaml.safe_load(semantic_built["mom6__n_groups__c6_injection"]["sheet_text"])
    assert sheet["injected_value"] == 5
    assert sheet["vector"] == "reader_note"
    assert sheet["expected_outcome"] == "injection_correct_or_abstain"


def test_semantic_metadata_blocks_are_byte_identical_to_clean_anchor(semantic_built):
    clean_text = open(
        "evaluation/canonical_texts/jnps_2013.frozen.yaml", encoding="utf-8"
    ).read()

    def meta_region(text: str) -> str:
        lines = text.splitlines(keepends=True)
        s = next(i for i, ln in enumerate(lines) if ln.startswith("source_pdf:"))
        e = next(i for i, ln in enumerate(lines) if ln.startswith("pages:"))
        return "".join(lines[s:e])

    frozen = semantic_built["mom6__signal_lag__c5_contradiction"]["frozen_text"]
    assert meta_region(frozen) == meta_region(clean_text)


def test_semantic_perturbation_provenance_block_present(semantic_built):
    raw = yaml.safe_load(semantic_built["str__sort_signal__c6_injection"]["frozen_text"])
    prov = raw["perturbation"]
    assert prov["anchor"] == "str"
    assert prov["field"] == "sort_signal"
    assert prov["class"] == 6
    assert prov["base_source_sha256"] == raw["source_sha256"]
    assert prov["injected_value"] == "long_term_reversal"
    assert prov["generator"] == "evaluation/harness/perturb.py"


def test_semantic_shared_drf_item_covers_both_fields(semantic_built):
    b = semantic_built["drf__long_leg+weighting_scheme__c3_paraphrase"]
    sheet = yaml.safe_load(b["sheet_text"])
    assert sheet["covers_fields"] == ["long_leg", "weighting_scheme"]
    assert sheet["true_value"] == {"long_leg": "highest_signal", "weighting_scheme": "value"}
    assert "replaced=1" in b["transform_id"]


def test_perturbed_text_is_never_the_clean_quote_and_frozen_loads(semantic_built, tmp_path):
    for b in semantic_built.values():
        sheet = yaml.safe_load(b["sheet_text"])
        assert normalise(sheet["perturbed_text"], "L1") != normalise(sheet["gold_quote"], "L1"), b["stem"]
    # spot-check that a perturbed frozen text still loads as a frozen canonical text.
    p = tmp_path / "sem.frozen.yaml"
    p.write_text(semantic_built["str__n_groups__c5_contradiction"]["frozen_text"], encoding="utf-8")
    ct = load_canonical_text(str(p))
    assert ct.is_frozen and len(ct.pages) > 0


def test_build_one_semantic_rejects_unknown_item():
    with pytest.raises(PerturbError):
        build_one_semantic("drf", "holding_period", 3)  # drf holding_period is UNKNOWN -> no item
