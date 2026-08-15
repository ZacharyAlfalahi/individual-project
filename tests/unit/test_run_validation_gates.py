"""
RQ2 anchor-fidelity aggregator tests (contract §7 gates 1-2, v1.6 re-scope).

Covers the repointed construction+rulebook verdict (incl. the holding_period
invariant to_rulebook omits) and PINS the differential's inertness: the gate
verdict is byte-identical with the differential diagnostic present or absent, and
the gate-path module (round_trip) does not import the differential computation.
Without these two assertions, "the differential is non-gating" is a claim about
today's code, not a property of it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO), str(_REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.harness import round_trip  # noqa: E402
from evaluation.harness.round_trip import (  # noqa: E402
    gate12_verdict,
    load_verified_standing_subs,
)

import run_validation_gates as rvg  # noqa: E402


@pytest.fixture(scope="module")
def subs():
    return load_verified_standing_subs()


def test_all_anchors_pass_gates_1_2(subs):
    v = rvg.compute_construction_rulebook_verdict(subs)
    assert v["construction_rulebook_pass"] is True
    for a in rvg.ANCHORS:
        av = v["anchors"][a]
        assert av["pass"] is True, (a, av)
        assert av["rulebook_byte_equal"] is True
        assert av["holding_period_match"] is True


def test_holding_period_invariant_checked(subs):
    # mom6 carries the only non-default holding period (STATED 6): the explicit
    # check must see it, since to_rulebook omits holding_period from byte-equality.
    v = gate12_verdict("mom6", subs)
    assert v["produced_holding_period"] == [6]
    assert v["expected_holding_period"] == 6
    assert v["holding_period_match"] is True
    for single in ("str", "drf"):
        assert gate12_verdict(single, subs)["produced_holding_period"] == [1]
    assert gate12_verdict("crf", subs)["produced_holding_period"] == [1, 1, 1]


def test_holding_period_mismatch_fails(subs, monkeypatch):
    # A corrupted holding expectation must flip the verdict -- proving the check
    # is load-bearing, not decorative (byte-equality alone would still pass).
    monkeypatch.setitem(round_trip._EXPECTED_HOLDING_PERIOD, "mom6", 1)
    v = gate12_verdict("mom6", subs)
    assert v["rulebook_byte_equal"] is True      # rulebook still byte-equal
    assert v["holding_period_match"] is False     # but holding now mismatches
    assert v["pass"] is False


def _gate_portion(report: dict) -> str:
    """The gate-bearing fields, JSON-serialised, excluding the non-gating
    diagnostic block and the wall-clock timestamp."""
    keep = {k: report[k] for k in report
            if k not in ("differential_diagnostic", "run_timestamp")}
    return json.dumps(keep, sort_keys=True, indent=2)


def test_verdict_byte_identical_with_and_without_differential(subs):
    # Inertness (PBO/G4 precedent): the gate verdict must be byte-identical
    # whether the differential diagnostic is populated or absent.
    v = rvg.compute_construction_rulebook_verdict(subs)
    populated = rvg.assemble_report(
        v, rvg.read_differential_diagnostic(), "construction_rulebook")
    absent = rvg.assemble_report(v, {}, "construction_rulebook")
    assert _gate_portion(populated) == _gate_portion(absent)


def test_gate_path_does_not_import_differential():
    # The gate-path module (round_trip) must not import the differential
    # computation -- else "non-gating" is a convention, not a property.
    src = Path(round_trip.__file__).read_text()
    for mod in ("build_str_decomposition", "run_mom6_lab_gate", "run_leadlag_gate"):
        assert mod not in src, f"round_trip references the differential computation ({mod})"
