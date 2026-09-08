"""one-shot holdout pre-run gate checklist (spec §1.3) — happy path + each item refuses individually."""

from __future__ import annotations

import dataclasses

import pytest

from agents.scientist.experimentalist.oneshot_holdout.gate_checklist import (
    OneshotHoldoutGateError,
    check_e9_cost_decision,
    check_manifest_wired,
    check_p3_artefact,
    check_release_tag,
    check_rehearsal_green,
    check_scsci12_approved,
    check_thresholds_fingerprint,
    check_window,
    run_pre_run_checklist,
    _tag_reachable_from_head,
)

from tests.unit._oneshot_holdout_fixtures import valid_checklist_cfg


def test_checklist_happy_path_returns_window(tmp_path):
    cfg = valid_checklist_cfg(tmp_path)
    if not _tag_reachable_from_head(cfg.release_tag, cfg.repo_root):
        pytest.skip(f"release tag {cfg.release_tag} not shipped with the repository")
    result = run_pre_run_checklist(cfg)
    assert (result.window.start, result.window.end, result.window.n_months) == ("2022-01", "2025-09", 45)
    assert len(result.checks_passed) == 8


def test_window_check_passes_against_real_protocol(tmp_path):
    window = check_window(valid_checklist_cfg(tmp_path))
    assert window.n_months == 45


def test_release_tag_refused_when_absent(tmp_path):
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), release_tag="no-such-tag-xyz")
    with pytest.raises(OneshotHoldoutGateError):
        check_release_tag(cfg)


def test_scsci12_refused_when_not_approved(tmp_path):
    # A temp protocol with a valid window but SC-SCI-12 status DRAFT.
    proto = tmp_path / "protocol.yaml"
    proto.write_text(
        "windows:\n  evaluation_holdout:\n    start: 2022-01\n    end: 2025-09\n    n_months: 45\n"
        "amendments:\n  - id: SC-SCI-12\n    status: DRAFT\n    summary: not yet verified\n"
    )
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), protocol_path=proto)
    with pytest.raises(OneshotHoldoutGateError):
        check_scsci12_approved(cfg)


def test_scsci12_discharge_rejects_negations_even_when_approved(tmp_path):
    # A fail-open substring match would pass "UNMET" (contains MET) or "not yet verified".
    for bad_summary in ("the frontier-completeness condition is UNMET", "evidence not yet verified"):
        proto = tmp_path / "protocol_neg.yaml"
        proto.write_text(
            "windows:\n  evaluation_holdout:\n    start: 2022-01\n    end: 2025-09\n    n_months: 45\n"
            f"amendments:\n  - id: SC-SCI-12\n    status: APPROVED\n    summary: {bad_summary}\n"
        )
        cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), protocol_path=proto)
        with pytest.raises(OneshotHoldoutGateError):
            check_scsci12_approved(cfg)


def test_thresholds_fingerprint_refused_on_mismatch(tmp_path):
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), thresholds_fingerprint="deadbeef" * 8)
    with pytest.raises(OneshotHoldoutGateError):
        check_thresholds_fingerprint(cfg)


def test_p3_artefact_refused_when_absent_without_fallback(tmp_path):
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), p3_artefact_path=None, p3_fallback_reason=None)
    with pytest.raises(OneshotHoldoutGateError):
        check_p3_artefact(cfg)


def test_p3_artefact_fallback_permitted_when_reason_logged(tmp_path):
    cfg = dataclasses.replace(
        valid_checklist_cfg(tmp_path), p3_artefact_path=None,
        p3_fallback_reason="dev G3 extension family empty at G5->G6 boundary; sigma=0.005 fallback",
    )
    check_p3_artefact(cfg)                       # does not raise


def test_e9_refused_when_undecided(tmp_path):
    cfg = dataclasses.replace(
        valid_checklist_cfg(tmp_path), e9_cost_model_id=None, e9_gross_returns_scoping=False,
    )
    with pytest.raises(OneshotHoldoutGateError):
        check_e9_cost_decision(cfg)


def test_manifest_refused_when_unwired(tmp_path):
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), manifest_writer_wired=False)
    with pytest.raises(OneshotHoldoutGateError):
        check_manifest_wired(cfg)


def test_rehearsal_green_required_for_real_run(tmp_path):
    cfg = dataclasses.replace(
        valid_checklist_cfg(tmp_path), require_rehearsal=True, rehearsal_marker_path=tmp_path / "absent.jsonl",
    )
    with pytest.raises(OneshotHoldoutGateError):
        check_rehearsal_green(cfg)


def test_rehearsal_check_skipped_inside_rehearsal(tmp_path):
    cfg = dataclasses.replace(valid_checklist_cfg(tmp_path), require_rehearsal=False)
    check_rehearsal_green(cfg)                    # does not raise
