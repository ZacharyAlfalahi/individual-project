"""B5b: corpus coverage scoring against the CI-5 labels (evaluation/harness/
t3_coverage.py) -- fixture-driven; the real numbers land after the paid run."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.harness.t3_coverage import (            # noqa: E402
    CoverageLabelError,
    load_coverage_labels,
    score_corpus_coverage,
)

LABELS = load_coverage_labels()
IMPLEMENT = sorted(LABELS.of("implement"))
REFUSE = sorted(LABELS.of("refuse"))
EXCLUDED = sorted(LABELS.of("excluded"))


def _row(key, outcome, codes=()):
    return {"paper_id": key[0], "name": key[1], "outcome": outcome,
            "refusal_codes": list(codes)}


def test_registered_labels_load_and_pin():
    assert LABELS.denominator == 32
    assert (len(IMPLEMENT), len(REFUSE), len(EXCLUDED)) == (27, 5, 1)


def test_clean_sweep_all_implemented_all_refused_correctly():
    observed = ([_row(k, "executed") for k in IMPLEMENT]
                + [_row(k, "refused", ["REVIEW_REQUIRED"]) for k in REFUSE])
    out = score_corpus_coverage(observed)
    assert out["C_end_to_end_full"] == 27 / 32
    assert out["FRR"] == 0.0
    assert out["refusal_recall"] == 1.0
    assert out["n_FIR_events"] == 0
    assert out["n_unobserved"] == 0


def test_fir_event_is_listed_not_only_counted():
    observed = ([_row(k, "executed") for k in IMPLEMENT]
                + [_row(REFUSE[0], "executed")]                       # the safety failure
                + [_row(k, "refused", ["REVIEW_REQUIRED"]) for k in REFUSE[1:]])
    out = score_corpus_coverage(observed)
    assert out["n_FIR_events"] == 1
    assert out["FIR_events"] == [list(REFUSE[0])]
    assert out["refusal_recall"] == 4 / 5


def test_frr_counts_only_should_implement_refusals():
    observed = ([_row(IMPLEMENT[0], "refused", ["MISSING_BINDING"])]  # false refusal
                + [_row(k, "executed") for k in IMPLEMENT[1:]]
                + [_row(k, "refused", ["REVIEW_REQUIRED"]) for k in REFUSE])
    out = score_corpus_coverage(observed)
    assert out["FRR"] == 1 / 27
    assert out["false_refusals"] == [list(IMPLEMENT[0])]
    # the refusal layer came from the typed code (binding), via the §5.2 classifier
    assert out["layered_observed"]["refusals_by_layer"]["binding"] == 1


def test_unobserved_rows_charge_the_full_denominator():
    """A paper_failed construction shrinks NOTHING: it is a listed coverage loss."""
    observed = [_row(k, "executed") for k in IMPLEMENT[:20]]          # 12 never observed
    out = score_corpus_coverage(observed)
    assert out["n_unobserved"] == 12
    assert out["denominator"] == 32
    assert out["C_end_to_end_full"] == 20 / 32


def test_phantom_construction_fails_loud():
    with pytest.raises(CoverageLabelError):
        score_corpus_coverage([_row(("DFPS_2026", "Made Up Factor"), "executed")])


def test_untyped_refusal_fails_loud():
    with pytest.raises(CoverageLabelError):
        score_corpus_coverage([_row(IMPLEMENT[0], "refused", [])])


def test_excluded_row_is_echoed_never_scored():
    observed = ([_row(k, "executed") for k in IMPLEMENT]
                + [_row(k, "refused", ["REVIEW_REQUIRED"]) for k in REFUSE]
                + [_row(EXCLUDED[0], "refused", ["REVIEW_REQUIRED"])])
    out = score_corpus_coverage(observed)
    assert out["excluded_observed"] == [list(EXCLUDED[0])]
    assert out["denominator"] == 32                     # unchanged by the excluded row
    assert out["refusal_recall"] == 1.0                 # the excluded row scored nowhere
