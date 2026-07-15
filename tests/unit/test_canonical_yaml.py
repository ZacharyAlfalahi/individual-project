"""
Canonical-YAML rulebook serialisation (G2, D30): sorted-key determinism +
_apply_defaults key-set reconciliation + a failing byte-compare.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

import pytest  # noqa: E402

from evaluation.harness.canonical_yaml import (  # noqa: E402
    assert_rulebook_byte_equal,
    canonical_rulebook_yaml,
    rulebooks_equal,
)

# A produced-shaped rulebook (min_bonds / trim_rule) and a golden-shaped one (nw_lags):
# the same strategy, different key sets -- _apply_defaults must reconcile them.
_PRODUCED = {
    "score": "xret", "groups": 10, "weighting": "by_size", "long_group": 9,
    "short_group": 0, "signal_lag": 0, "min_bonds": 10, "trim_rule": {"method": "none"},
}
_GOLDEN = {
    "score": "xret", "groups": 10, "weighting": "by_size", "long_group": 9,
    "short_group": 0, "signal_lag": 0, "nw_lags": None,
}


def test_sorted_key_determinism():
    a = {"score": "x", "groups": 5, "weighting": "by_size"}
    b = {"weighting": "by_size", "groups": 5, "score": "x"}  # different insertion order
    assert canonical_rulebook_yaml(a) == canonical_rulebook_yaml(b)


def test_apply_defaults_reconciles_key_sets():
    assert rulebooks_equal(_PRODUCED, _GOLDEN)
    assert_rulebook_byte_equal(_PRODUCED, _GOLDEN)  # no raise


def test_byte_equal_raises_on_real_mismatch():
    with pytest.raises(AssertionError):
        assert_rulebook_byte_equal({"score": "x", "groups": 5}, {"score": "x", "groups": 10})
