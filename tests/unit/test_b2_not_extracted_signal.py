"""B2 (2026-09-02): the `not_extracted` registered reason + the per-model
format-failure trace signal.

Pins the two distinctions the §3.6 gate depends on:
  * never-asked (`UNKNOWN/not_extracted`, a statement about THIS RUN) is bucketed
    NOT_ASKED by REASON -- the note-prefix sniffing survives only as a fallback
    for pre-B2 run dirs;
  * a parse/schema failure (`ModelAnswer.parse_failed` -> `ModelTrace.parse_failed`)
    is a distinct trace signal, never conflated with the model reporting genuine
    paper silence.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.pipeline import real_client as rc                      # noqa: E402
from agents.librarian.pipeline.form_filler import fill_field                 # noqa: E402
from agents.librarian.pipeline.model_client import FakeModelClient, ModelAnswer  # noqa: E402
from agents.librarian.errors import LibrarianSchemaError                     # noqa: E402
from agents.librarian.validators.tag_reason import load_tag_reason_registry  # noqa: E402
from evaluation.harness.run_artefacts import RunField, _not_extracted_fields  # noqa: E402


# ---------------------------------------------------------------------------
# The registry row.
# ---------------------------------------------------------------------------

def test_registry_has_not_extracted_row():
    reg = load_tag_reason_registry()
    assert reg.has("UNKNOWN", "not_extracted")


def test_driver_emits_the_registered_reason():
    """The driver's _not_extracted helper now carries the registered reason
    (the note keeps the historical prefix for humans + old-run fallback)."""
    from scripts.run_librarian import _not_extracted

    inh = _not_extracted("smoke --limit")
    assert inh.tag == "UNKNOWN"
    assert inh.evidence.unknown_reason == "not_extracted"
    assert inh.evidence.note.startswith("not extracted")


# ---------------------------------------------------------------------------
# Scorer-side detection: reason primary, note-prefix fallback.
# ---------------------------------------------------------------------------

def _spec_with(evidence: dict) -> dict:
    return {"part2": {"holding_period": {"tag": "UNKNOWN", "value": None, "evidence": evidence}}}


def test_detection_by_reason_alone():
    """No note prefix needed once the reason is registered (new-style specs)."""
    spec = _spec_with({"note": "budget-skipped", "unknown_reason": "not_extracted"})
    assert _not_extracted_fields(spec) == {"holding_period"}


def test_detection_fallback_by_note_prefix():
    """Pre-B2 run dirs rode not_stated + the note marker; they must still score
    identically."""
    spec = _spec_with({"note": "not extracted (smoke --limit)", "unknown_reason": "not_stated"})
    assert _not_extracted_fields(spec) == {"holding_period"}


def test_genuine_silence_is_not_detected():
    spec = _spec_with({"note": "neither model found a stated value", "unknown_reason": "not_stated"})
    assert _not_extracted_fields(spec) == set()


# ---------------------------------------------------------------------------
# parse_failed: producer paths (real_client).
# ---------------------------------------------------------------------------

def test_unusable_shape_sets_parse_failed():
    """answered:true with a missing quote/value is a SCHEMA failure."""
    a = rc._answer_from_parsed("weighting_scheme", "enum",
                               {"answered": True, "value": "value"}, "m-1")   # no quote
    assert a.answered is False and a.parse_failed is True
    b = rc._answer_from_parsed("weighting_scheme", "enum",
                               {"answered": True, "quote": "q"}, "m-1")       # no value
    assert b.answered is False and b.parse_failed is True


def test_genuine_model_silence_is_not_parse_failed():
    a = rc._answer_from_parsed("weighting_scheme", "enum", {"answered": False}, "m-1")
    assert a.answered is False and a.parse_failed is False


def test_parse_failed_contradicting_answered_is_a_build_error():
    with pytest.raises(LibrarianSchemaError):
        ModelAnswer(field="f", answered=True, raw="v", quote="q", parse_failed=True)


# ---------------------------------------------------------------------------
# parse_failed: threads through the D9 merge into the trace dict.
# ---------------------------------------------------------------------------

class _CT:
    """Minimal locate-able stand-in (fill_field only needs .locate)."""
    def locate(self, quote):
        return None


def test_parse_failed_threads_into_trace():
    fld = "weighting_scheme"
    a = FakeModelClient("fa", {fld: ModelAnswer(field=fld, answered=False, parse_failed=True)})
    b = FakeModelClient("fb", {fld: ModelAnswer(field=fld, answered=False)})

    out = fill_field(fld, a, b, _CT())
    rec = out.trace.to_dict()
    assert rec["model_a"]["parse_failed"] is True
    assert rec["model_b"]["parse_failed"] is False
    # The merged reason is unchanged by design (both silent -> not_stated); the
    # SIGNAL lives per-model in the trace, additive to the four §3.6 mechanisms.
    assert rec["final_reason"] == "not_stated"


def test_runfield_carries_parse_failed_condition():
    kw = dict(field="f", a_quote=None, b_quote=None, a_located=False, b_located=False,
              a_model_id="a", b_model_id="b", normalised_a=None, normalised_b=None,
              final_tag="UNKNOWN", shipped_reason="not_stated", ship_choice=None,
              not_extracted=False)
    rf = RunField(a_answered=False, b_answered=False, a_parse_failed=True, **kw)
    assert "parse_failed" in rf.conditions
    clean = RunField(a_answered=False, b_answered=False, **kw)
    assert "parse_failed" not in clean.conditions   # pre-B2 traces default False
