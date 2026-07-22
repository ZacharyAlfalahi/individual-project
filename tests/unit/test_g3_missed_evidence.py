"""
G3 missed-evidence decomposition -- the §3.6 gate input (contract v1.1, D38).

The decomposition exists because v1's §3.6 bundled mechanisms with different
remedies and routed them all to a retrieval loop. These tests pin the partition
(mutually exclusive, exhaustive) and the classification rules, because a
misclassification here would route an architecture decision to the wrong remedy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.harness.gold_calibration import score_anchor  # noqa: E402
from evaluation.harness.missed_evidence import (  # noqa: E402
    MECHANISM_REMEDY,
    Mechanism,
    _classify,
    decompose,
    render_decomposition,
)
from evaluation.harness.run_artefacts import RunField, load_run  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_BBW = _ROOT / "runs" / "g3_2026-07-21_postfix" / "bbw"
_JNPS = _ROOT / "runs" / "g3_2026-07-21_postfix" / "jnps"

_needs_bbw = pytest.mark.skipif(not (_BBW / "trace_0.json").exists(),
                                reason="post-fix BBW dev run absent (runs/ is gitignored)")
_needs_jnps = pytest.mark.skipif(not (_JNPS / "trace_0.json").exists(),
                                 reason="post-fix JNPS dev run absent (runs/ is gitignored)")


class _Key:
    def __init__(self, name="n_groups"):
        self.name = name


class _Row:
    def __init__(self, gold_value=5, gold_tag="STATED", name="n_groups"):
        self.key = _Key(name)
        self.gold_value = gold_value
        self.gold_tag = gold_tag


def _rf(**over) -> RunField:
    base = dict(field="n_groups", a_answered=True, b_answered=True, a_quote="q", b_quote="q",
                a_located=True, b_located=True, a_model_id="a", b_model_id="b",
                normalised_a=5, normalised_b=5, final_tag="UNKNOWN",
                shipped_reason="quote_match_failure", ship_choice=None, not_extracted=False)
    base.update(over)
    return RunField(**base)


# --- classification ----------------------------------------------------------

def test_neither_model_answered_is_not_retrieved():
    m, _ = _classify(_Row(), _rf(a_answered=False, b_answered=False,
                                 normalised_a=None, normalised_b=None))
    assert m is Mechanism.NOT_RETRIEVED


def test_right_value_whose_quote_did_not_locate_is_gate_lost():
    """The model found the evidence and was RIGHT; only the locator refused it.
    Routing this to a retrieval loop is what v1 got wrong."""
    m, note = _classify(_Row(), _rf(a_located=False, b_answered=False, normalised_b=None))
    assert m is Mechanism.GATE_LOST
    assert "did not locate" in note


def test_gate_lost_is_detected_per_model_not_pairwise():
    """A correct model_a whose quote failed still counts even when model_b was
    present and wrong -- otherwise the mechanism would be masked by the pair."""
    m, _ = _classify(_Row(), _rf(a_located=False, normalised_b=99))
    assert m is Mechanism.GATE_LOST


def test_right_value_that_located_but_was_not_shipped_is_merge_refused():
    """The unnamed v1 case: a model held the gold value, it located, and D9
    refused it because the other model was silent. Neither retrieval nor
    self-consistency addresses abstention."""
    m, note = _classify(_Row(), _rf(b_answered=False, normalised_b=None))
    assert m is Mechanism.MERGE_REFUSED
    assert "model_b was silent" in note


def test_no_model_had_the_gold_value_is_value_wrong():
    m, _ = _classify(_Row(), _rf(normalised_a=99, normalised_b=99))
    assert m is Mechanism.VALUE_WRONG


def test_every_mechanism_has_a_declared_remedy():
    for m in Mechanism:
        assert MECHANISM_REMEDY[m].strip()


# --- the partition -----------------------------------------------------------

@_needs_bbw
@_needs_jnps
def test_the_decomposition_is_exhaustive_and_exclusive():
    """Shares must sum to the total, or the '"driven by" clause has no
    unambiguous referent."""
    for anchor, run in (("drf", _BBW), ("mom6", _JNPS)):
        art = load_run(run)
        d = decompose(score_anchor(anchor, run, artefacts=art), art)
        assert sum(d.counts.values()) == d.total
        assert sum(d.shares[m.value].numerator for m in Mechanism) == d.total


@_needs_bbw
def test_drf_decomposition_golden():
    art = load_run(_BBW)
    d = decompose(score_anchor("drf", _BBW, artefacts=art), art)
    assert d.total == 14
    assert d.counts == {"gate_lost": 6, "merge_refused": 5, "value_wrong": 3}
    assert d.dominant is Mechanism.GATE_LOST


@_needs_jnps
def test_mom6_decomposition_golden():
    art = load_run(_JNPS)
    d = decompose(score_anchor("mom6", _JNPS, artefacts=art), art)
    assert d.total == 16
    assert d.counts == {"merge_refused": 8, "value_wrong": 6,
                        "not_retrieved": 1, "gate_lost": 1}
    assert d.dominant is Mechanism.MERGE_REFUSED


@_needs_bbw
@_needs_jnps
def test_the_dominant_mechanism_is_paper_dependent():
    """Which is why contract v1.1 routes on the POOLED decomposition: this is one
    architecture decision for one pipeline, not one per paper."""
    dom = {}
    for anchor, run in (("drf", _BBW), ("mom6", _JNPS)):
        art = load_run(run)
        dom[anchor] = decompose(score_anchor(anchor, run, artefacts=art), art).dominant
    assert dom["drf"] is not dom["mom6"]


@_needs_bbw
def test_react_target_is_the_smallest_component():
    """The finding that motivated D38: v1 routed the whole bundle to ReAct, whose
    actual target (not_retrieved) is near-absent."""
    art_b, art_j = load_run(_BBW), load_run(_JNPS)
    total = notret = 0
    for anchor, run, art in (("drf", _BBW, art_b), ("mom6", _JNPS, art_j)):
        d = decompose(score_anchor(anchor, run, artefacts=art), art)
        total += d.total
        notret += d.counts.get("not_retrieved", 0)
    assert notret / total < 0.10


def test_a_tie_reports_no_dominant_rather_than_breaking_it():
    """A tie means the evidence does not identify a remedy. Contract v1.1 routes
    that to Z, never to a default, so `dominant` must return None."""
    from evaluation.harness.missed_evidence import MissedEvidenceDecomposition, MissedField

    fields = (
        MissedField("f1", Mechanism.GATE_LOST, 1, 1, None, ""),
        MissedField("f2", Mechanism.MERGE_REFUSED, 1, 1, None, ""),
    )
    d = MissedEvidenceDecomposition(anchor_id="x", total=2, fields=fields, shares={})
    assert d.dominant is None


@_needs_bbw
def test_render_names_the_remedy_for_every_mechanism():
    art = load_run(_BBW)
    out = render_decomposition(decompose(score_anchor("drf", _BBW, artefacts=art), art))
    for m in Mechanism:
        assert m.value in out
    assert "LOCATOR remediation" in out and "PAIR remediation" in out
