"""Tests for typed StrategySpec deserialization.

The correctness spine is the ROUND-TRIP LAW: for every emitted spec dict d,
``spec_from_dict(d).to_dict() == d`` — pinned here over (a) the real paid-run
artefacts on disk (machine-local, skipif absent) and (b) a CI-stable typed spec
built from the committed T4(b) planted key, where full OBJECT equality is also
asserted. Strictness (unknown/missing keys fail loud, schema guards re-run) is
pinned on tampered dicts.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.pipeline.spec_loader import (  # noqa: E402
    SpecDeserialisationError,
    spec_from_dict,
)
from agents.quant.config.provenance import ProvenanceError  # noqa: E402

_RUN_DIRS = [
    _REPO_ROOT / "runs" / "corpus_anchors_report" / "bbw",
    _REPO_ROOT / "runs" / "corpus_anchors_report" / "jnps",
    _REPO_ROOT / "runs" / "corpus_anchors_report" / "drr",
    _REPO_ROOT / "runs" / "bbw_4anchor_report",
    _REPO_ROOT / "runs" / "corpus_corpus_report" / "bbw2021",
]
_T5_ROOT = _REPO_ROOT / "runs" / "t5"
_KEY = _REPO_ROOT / "evaluation" / "synthetic" / "planted_key_synth_2026.yaml"

_needs_runs = pytest.mark.skipif(
    not (_RUN_DIRS[0] / "spec_0.json").exists(),
    reason="paid run dirs absent (runs/ is gitignored, machine-local)",
)


def _emitted_spec_dicts():
    dirs = list(_RUN_DIRS)
    if _T5_ROOT.is_dir():
        dirs += sorted(p for p in _T5_ROOT.iterdir() if p.is_dir())[:10]
    for d in dirs:
        for p in sorted(d.glob("spec_*.json")):
            yield p, json.loads(p.read_text(encoding="utf-8"))


# --- the round-trip law -------------------------------------------------------

@_needs_runs
def test_round_trip_law_over_the_real_emitted_runs():
    n = 0
    for path, d in _emitted_spec_dicts():
        spec = spec_from_dict(d)
        assert spec.to_dict() == d, f"round-trip drift for {path}"
        n += 1
    assert n >= 8                                       # anchors + 4anchor + bbw2021 (+T5)


def test_round_trip_object_equality_on_key_derived_spec():
    """CI-stable: the committed planted key builds a full typed spec; the loaded
    object must equal the original EXACTLY (frozen dataclass equality), and the
    dict round-trips."""
    from scripts.run_t4b_kat import _key_derived_spec

    key = yaml.safe_load(_KEY.read_text(encoding="utf-8"))
    spec = _key_derived_spec(key)
    d = spec.to_dict()
    loaded = spec_from_dict(d)
    assert loaded == spec
    assert loaded.to_dict() == d


# --- strictness ---------------------------------------------------------------

def _key_spec_dict():
    from scripts.run_t4b_kat import _key_derived_spec

    key = yaml.safe_load(_KEY.read_text(encoding="utf-8"))
    return _key_derived_spec(key).to_dict()


def test_unknown_key_fails_loud_naming_the_path():
    d = copy.deepcopy(_key_spec_dict())
    d["part2"]["legs"][0]["surprise"] = 1
    with pytest.raises(SpecDeserialisationError, match=r"legs\[0\].*surprise"):
        spec_from_dict(d)


def test_missing_key_fails_loud():
    d = copy.deepcopy(_key_spec_dict())
    del d["part2"]["weighting_scheme"]
    with pytest.raises(SpecDeserialisationError, match="weighting_scheme"):
        spec_from_dict(d)


def test_unknown_evidence_key_fails_loud():
    d = copy.deepcopy(_key_spec_dict())
    d["part1"]["asset_class"]["evidence"]["confidence"] = 0.9
    with pytest.raises(SpecDeserialisationError, match="confidence"):
        spec_from_dict(d)


def test_schema_guards_rerun_on_load():
    """The constructors re-validate: a STATED field stripped of its quote is a
    provenance violation at load time, never a silently-accepted artefact."""
    d = copy.deepcopy(_key_spec_dict())
    label = d["header"]["strategy_label"]
    assert label["tag"] == "STATED"
    label["evidence"].pop("quote")
    with pytest.raises(ProvenanceError):
        spec_from_dict(d)


def test_conditional_omissions_round_trip():
    """Evidence drops Nones; the header emits standing-subs stamps only when
    present; Locator drops end_page when None — all must round-trip exactly."""
    d = copy.deepcopy(_key_spec_dict())
    spec = spec_from_dict(d)
    out = spec.to_dict()
    assert ("standing_substitutions_version" in out["header"]) == \
        ("standing_substitutions_version" in d["header"])
    assert out == d


# --- the unblocked consumer ----------------------------------------------------

@_needs_runs
def test_adapt_spec_accepts_a_loaded_emitted_spec():
    """The deferral this build closes: adapt_spec over a REAL emitted spec.
    The outcome (bound or typed refusal) is data — asserted typed, not pinned."""
    from agents.librarian.adapter import adapt_spec

    d = json.loads((_RUN_DIRS[0] / "spec_0.json").read_text(encoding="utf-8"))
    result = adapt_spec(spec_from_dict(d))
    assert hasattr(result, "refused") and isinstance(result.refused, bool)
    assert hasattr(result, "refusals")


# --- dormant paths (review-gap probes made executable, 2026-09-03): shapes no
# on-disk emission carries yet -- estimation blocks, cross-page locators,
# parameterised SignalRefs, binding-style evidence, unicode/falsy values. ------

def _unk(note="searched; silent"):
    from agents.quant.config import Evidence, Inherited

    return Inherited(None, "UNKNOWN", Evidence(note=note, unknown_reason="not_stated"))


def _stated(value, quote, *, end_page=None):
    from agents.quant.config import Evidence, Inherited
    from agents.quant.config.provenance import Locator

    return Inherited(value, "STATED", Evidence(
        quote=quote, locator=Locator(page=3, char_start=5, char_end=5 + len(quote),
                                     end_page=end_page)))


def _min_part2():
    from agents.librarian.schema.strategy_spec import (
        _COMMON_INHERITED_FIELDS, Combiner, Leg, Part2)
    from agents.librarian.schema.signal_ref import DescribedSignal, SignalRef

    leg = Leg(
        sort_signal=SignalRef(
            concept_id=_stated("var_5pct", "the 5% VaR"),
            as_described=DescribedSignal(label="downside risk"),
            # Dormant: NO current registry concept carries parameters.
            parameters={"window_months": _stated(36, "past 36 months")},
        ),
        control_axis=None,
        **{n: _unk() for n in ("sort_kind", "bucketing_method", "n_groups",
                               "stripe_aggregation", "control_missing_policy",
                               "long_leg", "signal_transform", "control_n_groups")},
    )
    return Part2(**{n: _unk() for n in _COMMON_INHERITED_FIELDS},
                 legs=(leg,), combiner=Combiner(kind=_unk()))


def _min_header():
    from agents.librarian.schema.strategy_spec import SpecHeader

    return SpecHeader(paper_id="X_1", strategy_label=_stated("S", "the S strategy"),
                      registry_version="v", registry_hash="h",
                      silence_table_version="s", canonical_text_hash="c")


def _min_part1():
    from agents.librarian.schema.strategy_spec import MethodSummary, Part1
    from agents.librarian.schema.signal_ref import LocatedQuote

    return Part1(formation_structure=_unk(), asset_class=_unk(),
                 method_summary=MethodSummary(
                     summary=_stated("does — sørts • bonds", "sørts • bonds"),
                     quotes=(LocatedQuote(text="sørts • bonds", page=2,
                                          char_start=0, char_end=13),)))


def test_dormant_estimation_spec_round_trips():
    """Fitted-model shape (estimation + instruments present, paper_facts None):
    the exact future-emission layout no run has produced yet."""
    from agents.librarian.schema.estimation_fields import (
        ESTIMATION_FIELDS, INSTRUMENT_INHERITED_FIELDS)
    from agents.librarian.schema.signal_ref import DescribedSignal
    from agents.librarian.schema.strategy_spec import (
        EstimationBlock, InstrumentRef, InstrumentSet, StrategySpec)

    spec = StrategySpec(
        header=_min_header(), part1=_min_part1(), part2=_min_part2(),
        paper_facts=None,
        estimation=EstimationBlock(**{n: _unk() for n in ESTIMATION_FIELDS}),
        instruments=InstrumentSet(instruments=(
            InstrumentRef(concept_id=_stated("rating", "credit rating"),
                          as_described=DescribedSignal(label="rating"),
                          **{n: _unk() for n in INSTRUMENT_INHERITED_FIELDS}),
        )),
    )
    d = spec.to_dict()
    assert "estimation" in d and "instruments" in d and d["paper_facts"] is None
    loaded = spec_from_dict(d)
    assert loaded == spec
    assert loaded.to_dict() == d


def test_dormant_cross_page_locator_and_binding_evidence_round_trip():
    """end_page (cross-page quote) + candidates/chosen/column evidence keys."""
    from agents.quant.config import Evidence
    from agents.librarian.schema.strategy_spec import StrategySpec

    spec = StrategySpec(header=_min_header(), part1=_min_part1(), part2=_min_part2())
    d = spec.to_dict()
    # Splice a cross-page STATED + a binding-style evidence into the dict (the
    # exact shapes the schema's to_dicts emit for these dormant cases).
    d["part1"]["asset_class"] = {
        "value": "corporate_bonds", "tag": "STATED",
        "evidence": {"quote": "bonds straddling a page",
                     "locator": {"page": 3, "char_start": 5, "char_end": 28,
                                 "end_page": 4}},
    }
    d["part2"]["tie_break_policy"] = {
        "value": None, "tag": "UNKNOWN",
        "evidence": Evidence(note="ambiguous columns", unknown_reason="not_stated",
                             candidates=("a", "b"), chosen="a", column="a").to_dict(),
    }
    loaded = spec_from_dict(d)
    assert loaded.to_dict() == d
    assert loaded.part1.asset_class.evidence.locator.end_page == 4
    assert loaded.part2.tie_break_policy.evidence.candidates == ("a", "b")


def test_dormant_unicode_and_parameterised_signal_round_trip():
    from agents.librarian.schema.strategy_spec import StrategySpec

    spec = StrategySpec(header=_min_header(), part1=_min_part1(), part2=_min_part2())
    d = spec.to_dict()
    loaded = spec_from_dict(d)
    assert loaded == spec
    assert loaded.to_dict() == d
    p = loaded.part2.legs[0].sort_signal.parameters
    assert p["window_months"].value == 36                # parameterised ref survives
    assert "sørts • bonds" in loaded.part1.method_summary.summary.evidence.quote
