"""Inc2-B — Benjamini-Hochberg over the confirmatory family (§7.2)."""

from __future__ import annotations

import pytest

from agents.auditor.checks.fdr import (
    benjamini_hochberg,
    corpus_confirmatory_fdr,
    run_fdr,
)


def test_bh_hand_computed_rejections():
    # Classic BH example: m=5, q=0.05.
    #   sorted p: 0.009, 0.01, 0.03, 0.04, 0.20
    #   thresholds k/m*q: 0.01, 0.02, 0.03, 0.04, 0.05
    #   largest k with p_(k) <= k/m*q: p_(4)=0.04 <= 0.04 => reject ranks 1..4
    p = {"a": 0.009, "b": 0.01, "c": 0.03, "d": 0.04, "e": 0.20}
    dec = benjamini_hochberg(p, 0.05)
    assert dec["a"].rejected and dec["b"].rejected
    assert dec["c"].rejected and dec["d"].rejected
    assert not dec["e"].rejected


def test_bh_adjusted_p_is_monotone_in_rank():
    p = {"a": 0.001, "b": 0.02, "c": 0.03, "d": 0.5}
    dec = benjamini_hochberg(p, 0.05)
    ordered = sorted(dec.values(), key=lambda d: d.rank)
    adj = [d.adjusted_p for d in ordered]
    assert adj == sorted(adj)  # non-decreasing with rank
    assert all(0.0 <= a <= 1.0 for a in adj)


def test_bh_all_null_none_rejected():
    p = {"a": 0.4, "b": 0.6, "c": 0.9}
    dec = benjamini_hochberg(p, 0.05)
    assert not any(d.rejected for d in dec.values())


def test_bh_nan_treated_as_one_never_rejected():
    p = {"a": 0.001, "b": float("nan")}
    dec = benjamini_hochberg(p, 0.05)
    assert dec["a"].rejected
    assert not dec["b"].rejected and dec["b"].p_value == 1.0


def test_bh_less_conservative_than_bonferroni():
    # BH should reject at least as many as Bonferroni (p <= q/m).
    p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    m, q = 4, 0.05
    bonf = {k: (v <= q / m) for k, v in p.items()}
    dec = benjamini_hochberg(p, q)
    for k in p:
        assert dec[k].rejected or not bonf[k]  # BH rejects a superset of Bonferroni


def test_run_fdr_report_counts_and_serialises():
    p = {frozenset({"meas_err"}): 0.001, frozenset({"lib_gap"}): 0.5}
    report = run_fdr(p, 0.1)
    assert report.n_family == 2
    assert report.n_rejected == 1
    d = report.to_dict()
    assert d["q"] == 0.1
    assert "meas_err" in d["decisions"]


def test_run_fdr_rejects_invalid_q():
    with pytest.raises(ValueError):
        benjamini_hochberg({"a": 0.1}, 1.5)


# --------------------------------------------------------------------------
# S4 — corpus confirmatory FDR spans the locked anchors; per-strategy is diagnostic
# --------------------------------------------------------------------------

def test_within_strategy_scope_label():
    report = run_fdr({frozenset({"meas_err"}): 0.01}, 0.1)
    assert report.scope == "within_strategy"
    assert report.to_dict()["scope"] == "within_strategy"


def test_corpus_fdr_builds_union_family_across_anchors():
    per_strategy = {
        "mom6": {frozenset({"lab_trim"}): 0.001, frozenset({"meas_err"}): 0.4},
        "drf": {frozenset({"meas_err"}): 0.002, frozenset({"lib_gap"}): 0.6},
    }
    report = corpus_confirmatory_fdr(per_strategy, 0.1)
    # union family = 2 strategies × 2 coordinates = 4 tests, keyed by (strategy, coord)
    assert report.scope == "corpus_confirmatory"
    assert report.n_family == 4
    labels = set(report.to_dict()["decisions"])
    assert "mom6::lab_trim" in labels and "drf::meas_err" in labels


def test_corpus_fdr_adjustment_spans_all_anchors():
    # The BH denominator is the FULL union size (4), so a p that would reject in a
    # 2-test family may not in the 4-test corpus family — the point of §7.2.
    per_strategy = {
        "mom6": {frozenset({"lab_trim"}): 0.02, frozenset({"meas_err"}): 0.9},
        "drf": {frozenset({"meas_err"}): 0.9, frozenset({"lib_gap"}): 0.9},
    }
    corpus = corpus_confirmatory_fdr(per_strategy, 0.05)
    within = run_fdr({frozenset({"lab_trim"}): 0.02, frozenset({"meas_err"}): 0.9}, 0.05)
    # smallest p 0.02: within-family (m=2) BH threshold 0.025 => reject;
    # corpus-family (m=4) BH threshold 0.0125 => NOT reject.
    assert within.decisions[frozenset({"lab_trim"})].rejected
    assert not corpus.decisions[("mom6", frozenset({"lab_trim"}))].rejected
