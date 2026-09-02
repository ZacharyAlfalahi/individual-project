"""B5a: the T3 coverage labels (evaluation/gold_specs/t3_coverage_labels.yaml)
must stay an exact, bijective encoding of the pre-registered CI-5 partition:
32 = 27 implement + 5 refuse (+ 1 excluded, outside both denominators), keyed by
construction NAME over the two admitted enum golds' strategy rows.

Following the test_t2_selection_prereg.py convention: the registered document is
the authority; this test makes drift between the labels, the enum golds, and the
CI-5 counts a test failure rather than a silent re-registration."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

_LABELS = _REPO_ROOT / "evaluation" / "gold_specs" / "t3_coverage_labels.yaml"


def _load():
    return yaml.safe_load(_LABELS.read_text(encoding="utf-8"))


def _enum_strategy_names(enum_path: Path) -> list[str]:
    d = yaml.safe_load(enum_path.read_text(encoding="utf-8"))
    return [c["name"] for c in d["constructions"] if c["class"] == "strategy"]


def test_ci5_counts_pin():
    """27 implement + 5 refuse + 1 excluded = 33 strategy constructions; the
    denominator is 32. These are the registered CI-5 numbers -- any change is a
    re-registration, never a drive-by edit."""
    labels = _load()
    counts = Counter(
        label
        for paper in labels["papers"].values()
        for label in paper["constructions"].values()
    )
    assert counts == {"implement": 27, "refuse": 5, "excluded": 1}
    assert sum(counts.values()) == 33
    assert counts["implement"] + counts["refuse"] == 32   # the denominator


def test_bijection_with_the_enum_golds():
    """Every enum-gold strategy row has exactly one label; every label keys a
    real strategy row (no phantom names, no missed constructions)."""
    labels = _load()
    for paper_id, block in labels["papers"].items():
        enum_names = _enum_strategy_names(_REPO_ROOT / block["enum_gold"])
        label_names = list(block["constructions"])
        assert sorted(label_names) == sorted(enum_names), paper_id
        assert len(label_names) == len(set(label_names)), f"{paper_id}: duplicate name"


def test_registered_refuse_set_is_the_ci5_five():
    """The five should-refuse constructions are named in the CI-5 registration;
    pin them so the refusal-recall frame cannot drift."""
    labels = _load()
    refuse = {name
              for paper in labels["papers"].values()
              for name, lab in paper["constructions"].items() if lab == "refuse"}
    assert refuse == {
        "Stock systematic risk (SR) quintile portfolios",
        "Stock idiosyncratic risk (IR) quintile portfolios",
        "Firm-level value factor (VALfirm / bev_mev)",
        "Firm-level equity momentum factor (EQMOMfirm / seas_1_1na)",
        "Firm-level leverage factor (LEVfirm / at_me)",
    }


def test_the_153_family_is_the_only_exclusion():
    labels = _load()
    excluded = [name
                for paper in labels["papers"].values()
                for name, lab in paper["constructions"].items() if lab == "excluded"]
    assert excluded == ["Firm-level factors from 153 equity signals (Jensen et al. 2022)"]


def test_labels_are_a_closed_vocabulary():
    labels = _load()
    legal = {"implement", "refuse", "excluded"}
    for paper in labels["papers"].values():
        assert set(paper["constructions"].values()) <= legal
