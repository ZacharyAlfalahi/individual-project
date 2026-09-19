"""one-shot holdout integration (spec §5, §6): the full --rehearsal pass on synthetic development-shaped data,
and the real-path marker state machine (gate monkeypatched; the synthetic builder never reads
/data/holdout/). This is the §6 "full --rehearsal" integration test."""

from __future__ import annotations

import dataclasses

import pytest

from agents.scientist.experimentalist import holdout
from agents.scientist.experimentalist.oneshot_holdout.marker import Marker, RehearsalMarker, RerunRefused
from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import REHEARSAL_WINDOW, OneshotHoldoutConfig, run_oneshot_holdout
from agents.scientist.experimentalist.oneshot_holdout.windows import registered_window

from tests.unit._oneshot_holdout_fixtures import (
    synthetic_benchmarks,
    synthetic_panel_builder,
    synthetic_survivors,
    valid_checklist_cfg,
)

PRIORS = {"wide": 0.02, "moderate": 0.005, "sceptical": 0.0025}


def _rehearsal_cfg(tmp_path):
    win = REHEARSAL_WINDOW
    return OneshotHoldoutConfig(
        rehearsal=True,
        ts="2026-08-09T00:00:00",
        checklist=valid_checklist_cfg(tmp_path, require_rehearsal=False),
        quarantine_dir=tmp_path / "q",
        marker_path=tmp_path / "oneshot_marker.jsonl",
        rehearsal_marker_path=tmp_path / "rehearsal_marker.jsonl",
        panel_builder=synthetic_panel_builder(),
        survivors=synthetic_survivors(win),
        benchmarks=synthetic_benchmarks(win),
        priors=PRIORS,
    )


def test_full_rehearsal_is_green_and_writes_marker(tmp_path):
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import _tag_reachable_from_head
    _cfg = valid_checklist_cfg(tmp_path)
    if not _tag_reachable_from_head(_cfg.release_tag, _cfg.repo_root):
        pytest.skip("release tag not shipped with the repository")
    holdout._reset_single_access_for_tests()
    cfg = _rehearsal_cfg(tmp_path)
    report = run_oneshot_holdout(cfg)

    assert report.green is True and report.rehearsal is True
    assert report.window.n_months == 45
    assert report.stage1.all_seeded_green()                   # seeding validation (§5)
    assert report.seed_start == "2014-10"                     # 2018-01 − (36 + 3)
    # the registered window present for each survivor
    rec = report.results[0].benchmarks["bbw4"]
    assert rec.full["window_label"] == "full_45m"
    # rehearsal-green marker written; the real-run marker was NOT started
    assert RehearsalMarker(cfg.rehearsal_marker_path).is_green() is True
    assert Marker(cfg.marker_path).records() == []
    # the gate never opened during a rehearsal
    assert holdout._ACCESS["count"] == 0
    assert report.manifest["holdout_processed"] is False


def test_rehearsal_refuses_when_a_construction_is_nan_at_first_month(tmp_path):
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import _tag_reachable_from_head
    _cfg = valid_checklist_cfg(tmp_path)
    if not _tag_reachable_from_head(_cfg.release_tag, _cfg.repo_root):
        pytest.skip("release tag not shipped with the repository")
    holdout._reset_single_access_for_tests()
    # A builder whose frames are entirely NaN fails the seeding validation.
    import numpy as np
    import pandas as pd

    def broken_builder(*, seed_start, window):
        idx = pd.period_range(seed_start, window.end, freq="M").to_timestamp("M")
        return {"drf": pd.DataFrame({"date": idx, "drf": np.full(len(idx), np.nan)})}

    cfg = dataclasses.replace(_rehearsal_cfg(tmp_path), panel_builder=broken_builder)
    from agents.scientist.experimentalist.oneshot_holdout.run_oneshot_holdout import RehearsalNotGreen
    with pytest.raises(RehearsalNotGreen):
        run_oneshot_holdout(cfg)


def test_real_path_state_machine_completes_then_refuses_rerun(tmp_path, monkeypatch):
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import _tag_reachable_from_head
    _cfg = valid_checklist_cfg(tmp_path)
    if not _tag_reachable_from_head(_cfg.release_tag, _cfg.repo_root):
        pytest.skip("release tag not shipped with the repository")
    holdout._reset_single_access_for_tests()
    monkeypatch.setattr(holdout, "prereg_tag_present", lambda *a, **k: True)
    monkeypatch.setattr(holdout, "env_unlock_set", lambda *a, **k: True)

    win = registered_window(("2022-01", "2025-09", 45))
    cfg = OneshotHoldoutConfig(
        rehearsal=False,
        ts="2026-08-09T00:00:00",
        checklist=valid_checklist_cfg(tmp_path, require_rehearsal=True),
        quarantine_dir=tmp_path / "q",
        marker_path=tmp_path / "oneshot_marker.jsonl",
        rehearsal_marker_path=tmp_path / "rehearsal_marker.jsonl",
        panel_builder=synthetic_panel_builder(),
        survivors=synthetic_survivors(win),
        benchmarks=synthetic_benchmarks(win),
        priors=PRIORS,
        holdout_reader_open=lambda: None,        # a no-op stands in for the gated real reader
    )

    report = run_oneshot_holdout(cfg)
    assert report.rehearsal is False and report.manifest["holdout_processed"] is True
    states = [r["state"] for r in Marker(cfg.marker_path).records()]
    assert states == ["STAGE1_STARTED", "STAGE1_COMPLETE", "STAGE2_STARTED", "COMPLETE"]
    assert holdout._ACCESS["count"] == 1                       # single access consumed

    # Re-invocation is refused by the failure protocol (no --force).
    with pytest.raises(RerunRefused):
        run_oneshot_holdout(cfg)
    holdout._reset_single_access_for_tests()


def test_real_path_derives_stage2_inputs_from_stage1(tmp_path, monkeypatch):
    """The stage-2 derivation seam: with ``derive_inputs`` set, the real path evaluates the survivors/benchmarks the
    deriver returns from the stage-1 result (not the config fields) and records its derivation."""
    from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import _tag_reachable_from_head
    _cfg = valid_checklist_cfg(tmp_path)
    if not _tag_reachable_from_head(_cfg.release_tag, _cfg.repo_root):
        pytest.skip("release tag not shipped with the repository")
    holdout._reset_single_access_for_tests()
    monkeypatch.setattr(holdout, "prereg_tag_present", lambda *a, **k: True)
    monkeypatch.setattr(holdout, "env_unlock_set", lambda *a, **k: True)
    win = registered_window(("2022-01", "2025-09", 45))
    seen = {}

    def derive(stage1):
        seen["names"] = sorted(a.name for a in stage1.artefacts)
        survivors = [dataclasses.replace(s, survivor_id=f"derived:{s.survivor_id}") for s in synthetic_survivors(win)]
        return survivors, synthetic_benchmarks(win), {"note": "derived from stage 1"}

    cfg = OneshotHoldoutConfig(
        rehearsal=False, ts="2026-08-09T00:00:00",
        checklist=valid_checklist_cfg(tmp_path, require_rehearsal=True),
        quarantine_dir=tmp_path / "q", marker_path=tmp_path / "oneshot_marker.jsonl",
        rehearsal_marker_path=tmp_path / "rehearsal_marker.jsonl",
        panel_builder=synthetic_panel_builder(), survivors=[], benchmarks={}, priors=PRIORS,
        holdout_reader_open=lambda: None, derive_inputs=derive,
    )
    report = run_oneshot_holdout(cfg)
    assert seen["names"]                                          # the deriver saw the stage-1 artefacts
    assert [r.survivor_id for r in report.results] == [f"derived:{s.survivor_id}" for s in synthetic_survivors(win)]
    assert report.manifest["survivor_ids"] == [r.survivor_id for r in report.results]
    assert report.manifest["stage2_input_derivation"] == {"note": "derived from stage 1"}
    holdout._reset_single_access_for_tests()
