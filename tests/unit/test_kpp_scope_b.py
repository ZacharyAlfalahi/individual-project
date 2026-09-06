"""Scope B (2026-09-04): the fitted-model extraction route.

Pins: the two new field-type renders + the menu-filled instruments run-template;
the int_set normalisation and decoding branches; the D9-extension instruments
merge (agreement + quote gate + dial rules); and the KPP fake-phase end-to-end
(emit -> typed round-trip -> score with the 8/11 denominator statement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText  # noqa: E402
from agents.librarian.errors import LibrarianSchemaError  # noqa: E402
from agents.librarian.pipeline import load_prompt_manifest  # noqa: E402
from agents.librarian.pipeline.form_filler import normalise  # noqa: E402
from agents.librarian.pipeline.real_client import (  # noqa: E402
    PromptBuilder,
    _answer_from_parsed,
)
from agents.librarian.registries import load_signal_concept_registry  # noqa: E402
from scripts import run_librarian  # noqa: E402
from scripts.run_librarian import _merge_instruments  # noqa: E402


@pytest.fixture(scope="module")
def builder():
    return PromptBuilder.load(load_signal_concept_registry(), load_prompt_manifest())


# --- renders -----------------------------------------------------------------

def test_estimation_enum_renders_menu_and_definition(builder):
    m = load_prompt_manifest()
    text = builder.render(m.query_for("intercept_spec"), "IPCA")
    assert "FIELD: intercept_spec" in text
    assert "restricted" in text and "unrestricted" in text and "other" in text
    assert "alpha" in text                              # the gloss reached the prompt
    assert "never more than about 200 characters" in text   # D39 carried


def test_int_set_renders_range(builder):
    m = load_prompt_manifest()
    text = builder.render(m.query_for("n_factors_tested"), "IPCA")
    assert "RANGE:" in text and "K sweep" in text
    assert "[1,2,3,4,5,6]" in text                      # the registered expansion rule


def test_instruments_prompt_fills_all_29_ids(builder):
    text = builder.instruments_prompt()
    assert "{registry_menu}" not in text
    assert text.count(" -- ") >= 29                     # 29 registry rows rendered
    assert "unrecognised" in text and "EXACT and BINARY" in text
    # Wrap-safe: the tie-breaker sentence spans a line break in the frozen text.
    assert "better omitted than invented" in " ".join(text.split())


# --- int_set decoding + normalisation ---------------------------------------

def test_normalise_int_set_sorts_and_dedups():
    assert normalise("n_factors_tested", [3, 1, 2, 3], "int_set") == (1, 2, 3)
    with pytest.raises(LibrarianSchemaError):
        normalise("n_factors_tested", [1, "two"], "int_set")
    with pytest.raises(LibrarianSchemaError):
        normalise("n_factors_tested", [True, 2], "int_set")   # bool is not an int
    with pytest.raises(LibrarianSchemaError):
        normalise("n_factors_tested", [], "int_set")


def test_answer_from_parsed_int_set_branch():
    ok = _answer_from_parsed("n_factors_tested", "int_set",
                             {"answered": True, "value": [5, 1, 3], "quote": "K = 1..5"},
                             "m")
    assert ok.answered and ok.raw == (1, 3, 5)
    bad = _answer_from_parsed("n_factors_tested", "int_set",
                              {"answered": True, "value": "1-5", "quote": "K"}, "m")
    assert bad.answered is False and bad.parse_failed is True


# --- the instruments merge ---------------------------------------------------

def _ct(text="the paper uses credit rating and bond age as instruments"):
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "1"},
        normalisation={"ladder_level": "L0", "rules": []},
        pages=(text,), status="stub")


def _row(cid, quote="credit rating", **kw):
    return {"concept_id": cid, "label": kw.get("label", cid), "quote": quote,
            "source_class": kw.get("source_class", "bond"),
            "transform": kw.get("transform"), "lag": kw.get("lag")}


def test_merge_ships_only_agreed_and_located():
    ct = _ct()
    inst, note = _merge_instruments(
        [_row("rating"), _row("age", quote="bond age"), _row("only_a")],
        [_row("rating"), _row("age", quote="bond age")],
        ct)
    ids = [i.concept_id.value for i in inst.instruments]
    assert ids == ["age", "rating"]                    # only_a dropped (one-sided)
    assert all(i.concept_id.tag == "STATED" for i in inst.instruments)
    assert "shipped=2" in note


def test_merge_quote_gate_drops_and_notes():
    inst, note = _merge_instruments(
        [_row("rating", quote="NOT IN TEXT")],
        [_row("rating", quote="ALSO NOT IN TEXT")],
        _ct())
    assert inst is None                                 # zero shipped -> None sibling
    assert "rating(quote gate)" in note


def test_merge_dial_disagreement_goes_unknown():
    inst, _ = _merge_instruments(
        [_row("rating", source_class="bond")],
        [_row("rating", source_class="equity")],
        _ct())
    ref = inst.instruments[0]
    assert ref.source_class.tag == "UNKNOWN"
    assert ref.transform.tag == "UNKNOWN"               # both-null never STATED


def test_merge_counts_unrecognised_rows():
    _, note = _merge_instruments(
        [_row("unrecognised"), _row("rating")], [_row("rating")], _ct())
    assert "unrecognised-rows=1" in note


# --- KPP end-to-end (fake phase, offline) ------------------------------------

def test_kpp_fake_run_emits_scores_and_round_trips(tmp_path):
    out = tmp_path / "kpp"
    assert run_librarian.main(["--paper", "kpp", "--phase", "fake",
                               "--out", str(out)]) == 0
    d = json.loads((out / "spec_0.json").read_text(encoding="utf-8"))
    assert d["header"]["strategy_label"]["value"] == "IPCA"
    assert "estimation" in d and "instruments" not in d   # no rows shipped by fakes
    # Rubric freeze PR-2 (2026-09-04): the prose fields are now ASKED via the
    # method_summary mechanism; the fake pair produces no located prose answer,
    # so they degrade to genuine silence (not_stated), never not_extracted.
    for prose in ("return_variable", "characteristic_preprocessing",
                  "managed_portfolio_construction"):
        assert d["estimation"][prose]["evidence"]["unknown_reason"] == "not_stated"
    assert d["part2"]["legs"][0]["sort_kind"]["evidence"]["unknown_reason"] == "not_extracted"

    from agents.librarian.pipeline.spec_loader import spec_from_dict
    assert spec_from_dict(d).to_dict() == d              # round-trip law

    from scripts.run_kpp_score import main as score_main
    out_json = tmp_path / "score.json"
    assert score_main(["--run-dir", str(out), "--out", str(out_json)]) == 0
    s = json.loads(out_json.read_text(encoding="utf-8"))
    assert s["summary"]["outcome_tally"].get("excluded_prose") == 3
    assert "8 of 11" in s["summary"]["denominator_statement"]
    assert s["summary"]["never_pooled"] is True


# --- review-gate regressions (2026-09-04) ------------------------------------

_EST_DEFS = _REPO_ROOT / "agents" / "librarian" / "data" / "estimation_definitions.yaml"
_EST_DEFS_SHA256 = "597ae3cae59ae68df561e6ba8c90449fd6acb344c5341f65a5a0355b16e359bf"  # frozen sha of estimation_definitions.yaml (rubric freeze)


def test_estimation_definitions_are_frozen():
    """Review m2: the estimation glosses reach rendered prompts, so the file is
    byte-pinned like the sort definitions -- an edit without re-stamping fails
    HERE, never as silently changed extraction."""
    import hashlib

    import yaml as _yaml

    b = _EST_DEFS.read_bytes()
    assert hashlib.sha256(b).hexdigest() == _EST_DEFS_SHA256, (
        "estimation_definitions.yaml edited without re-stamping the pin above")
    assert _yaml.safe_load(b)["version"] == "v1"


def test_sort_definition_maps_stay_byte_equal(builder):
    """Review M1: the estimation glosses live in SEPARATE builder maps; the
    frozen sort _definitions/_domains2 stay byte-equal to their sources."""
    from agents.librarian.registries import load_field_definitions

    assert builder._definitions == dict(load_field_definitions().definitions)
    assert "model_family" not in builder._definitions
    assert "model_family" in builder._estimation_definitions


def test_merge_degrades_nonstr_label_to_concept_id():
    """Review m3: a hostile non-str label never crashes the merge."""
    inst, _ = _merge_instruments(
        [dict(_row("rating"), label=123)], [dict(_row("rating"), label=None)], _ct())
    assert inst.instruments[0].as_described.label == "rating"


def test_merge_prefers_first_locatable_duplicate():
    """Review m4: duplicate concept_ids within one list keep the locatable row."""
    inst, _ = _merge_instruments(
        [_row("rating", quote="NOT IN TEXT"), _row("rating", quote="credit rating")],
        [_row("rating", quote="credit rating")], _ct())
    assert inst is not None and inst.instruments[0].concept_id.value == "rating"


def test_n_factors_tested_ships_as_json_native_list(tmp_path):
    """Review n7: the spec value is a LIST at the boundary (representation-stable
    to_dict), while the merge compared tuples internally."""
    out = tmp_path / "kpp"
    assert run_librarian.main(["--paper", "kpp", "--phase", "fake",
                               "--out", str(out)]) == 0
    d = json.loads((out / "spec_0.json").read_text(encoding="utf-8"))
    v = d["estimation"]["n_factors_tested"]["value"]
    assert v is None or isinstance(v, list)            # fake pair abstains -> None


# --- rubric composition plumbing (kpp_prose_rubric.md §2.2; INERT until frozen) --

def test_prose_judgement_flips_to_comparable_bool():
    from evaluation.harness.estimation_compare import (
        Comparability, compare_estimation_value)

    judged = compare_estimation_value("return_variable", "g", "r",
                                      rubric_judgement=True)
    assert judged.comparability == Comparability.COMPARABLE and judged.equal is True
    unjudged = compare_estimation_value("return_variable", "g", "r")
    assert unjudged.comparability == Comparability.NO_POLICY and unjudged.equal is None
    with pytest.raises(ValueError):
        compare_estimation_value("model_family", "g", "r", rubric_judgement=True)


def test_scorer_judgement_applies_only_to_shipped_prose(tmp_path):
    """A judgement grades a SHIPPED answer; an abstained/never-asked prose row
    keeps its coverage channel (D37 conv. 3) even when a judgement exists."""
    import dataclasses

    from agents.quant.config import Evidence, Inherited
    from agents.quant.config.provenance import Locator
    from evaluation.gold_specs.gold_loader import load_gold_spec
    from evaluation.harness.estimation_scoring import score_kpp

    out = tmp_path / "kpp"
    assert run_librarian.main(["--paper", "kpp", "--phase", "fake",
                               "--out", str(out)]) == 0
    from agents.librarian.pipeline.spec_loader import spec_from_dict
    run_spec = spec_from_dict(json.loads((out / "spec_0.json").read_text()))
    gold = load_gold_spec("kpp")

    # The fake run's prose rows are never-asked -> judgement must NOT bite.
    s = score_kpp(gold, run_spec, rubric_judgements={"return_variable": True})
    prose = [c for c in s.cells if c.key == "return_variable"]
    assert prose[0].outcome.value == "excluded_prose"

    # Ship a prose answer, then the judgement grades it.
    shipped = dataclasses.replace(
        run_spec.estimation,
        return_variable=Inherited(
            "excess returns scaled by DtS", "STATED",
            Evidence(quote="returns divided by Duration times Spread",
                     locator=Locator(page=37, char_start=0, char_end=44))))
    run2 = dataclasses.replace(run_spec, estimation=shipped)
    s2 = score_kpp(gold, run2, rubric_judgements={"return_variable": False})
    prose2 = [c for c in s2.cells if c.key == "return_variable"]
    assert prose2[0].outcome.value == "wrong"
