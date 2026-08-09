"""one-shot holdout marker state machine + pre-registered failure protocol (spec §1.1, §4)."""

from __future__ import annotations

import pytest

from agents.scientist.experimentalist.oneshot_holdout.marker import (
    FRESH_BUILD,
    RESTART_BUILD,
    RESUME_EVALUATE,
    Marker,
    MarkerError,
    RehearsalMarker,
    RerunRefused,
    plan_next,
)


def _recs(*states):
    return [{"state": s, "ts": "t"} for s in states]


def test_fresh_build_on_empty_history():
    assert plan_next([]) == FRESH_BUILD


def test_one_restart_permitted_after_crash_in_stage1():
    # STAGE1_STARTED without STAGE1_COMPLETE = a build crash before any result — one restart.
    assert plan_next(_recs("STAGE1_STARTED")) == RESTART_BUILD


def test_second_restart_refused():
    with pytest.raises(RerunRefused):
        plan_next(_recs("STAGE1_STARTED", "STAGE1_STARTED"))


def test_resume_evaluate_after_build_complete():
    assert plan_next(_recs("STAGE1_STARTED", "STAGE1_COMPLETE")) == RESUME_EVALUATE


def test_no_rerun_after_evaluation_began():
    with pytest.raises(RerunRefused):
        plan_next(_recs("STAGE1_STARTED", "STAGE1_COMPLETE", "STAGE2_STARTED"))


def test_no_rerun_after_complete_and_no_force():
    with pytest.raises(RerunRefused):
        plan_next(_recs("STAGE1_STARTED", "STAGE1_COMPLETE", "STAGE2_STARTED", "COMPLETE"))


def test_unknown_state_is_rejected():
    with pytest.raises(MarkerError):
        plan_next([{"state": "WAT"}])


def test_marker_append_is_append_only_and_round_trips(tmp_path):
    # NB: avoid the substring "holdout" in the path (the conftest guard blocks it).
    m = Marker(tmp_path / "oneshot_marker.jsonl")
    assert m.records() == []
    m.append("STAGE1_STARTED", ts="t1", extra={"code_hash": "abc"})
    m.append("STAGE1_COMPLETE", ts="t2")
    recs = m.records()
    assert [r["state"] for r in recs] == ["STAGE1_STARTED", "STAGE1_COMPLETE"]
    assert recs[0]["code_hash"] == "abc"
    assert m.plan_next() == RESUME_EVALUATE


def test_rehearsal_marker_green(tmp_path):
    rm = RehearsalMarker(tmp_path / "rehearsal_marker.jsonl")
    assert rm.is_green() is False
    rm.write_green(ts="t", extra={"seed_start": "2018-10"})
    assert rm.is_green() is True
