"""Reporter driver / CLI (docs/reporter/reporter_spec_v0.2.md §11): scaffold -> check -> render -> publish, no real data.

Mirrors test_run_auditor_driver.py: builds a synthetic repo, drives the pure cli functions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.reporter import cli
from agents.reporter.legal_states import PipelineDisposition

_TOGGLES = ("meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim")


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def _make_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "runs/g3/bbw/spec_0.json", {"paper_facts": None})
    qd = tmp_path / "results/quant/run_x"
    _write(qd / "drf.json", {"summary": {"sharpe": 0.5, "t_stat": 2.0, "average": 0.001}})
    _write(qd / "coverage.json", {"status": "run"})
    _write(qd / "run_log.json", {"git_commit": "a" * 40, "git_short": "aaaaaaa"})
    ad = tmp_path / "results/auditor/run_y"
    _write(
        ad / "drf_core.json",
        {
            "audit_scope": "COMPLETE",
            "runnable_toggles": list(_TOGGLES),
            "shapley": {
                "shapley_share_of_registered_endpoint_gap": {t: 0.0 for t in _TOGGLES}
            },
            "saturated_bases": {"doe_effects": {t: 0.0 for t in _TOGGLES}},
        },
    )
    _write(ad / "run_log.json", {"git_commit": "1111111", "auditor_prereg_tag": "tag-1"})
    return tmp_path


def _scaffold_pointer_file(tmp_path: Path) -> None:
    text = cli.scaffold(
        reporter_run_id="bbw_drf",
        paper_id="BBW",
        strategy_id="drf",
        phase="F",
        spec="runs/g3/bbw/spec_0.json",
        quant_dir="results/quant/run_x",
        audit_dir="results/auditor/run_y",
        repo_root=tmp_path,
    )
    pointer = tmp_path / "reporter" / "runs" / "bbw_drf.yaml"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(text)


def test_scaffold_includes_present_artefacts(tmp_path):
    _make_repo(tmp_path)
    text = cli.scaffold(
        reporter_run_id="bbw_drf",
        paper_id="BBW",
        strategy_id="drf",
        phase="F",
        spec="runs/g3/bbw/spec_0.json",
        quant_dir="results/quant/run_x",
        audit_dir="results/auditor/run_y",
        repo_root=tmp_path,
    )
    for key in ("spec", "quant_result", "quant_coverage", "audit_report", "audit_run_log"):
        assert f"  {key}:" in text


def test_check_run_verifies(tmp_path):
    _make_repo(tmp_path)
    _scaffold_pointer_file(tmp_path)
    doc = cli.check_run("bbw_drf", repo_root=tmp_path)
    assert doc.disposition is PipelineDisposition.AUDIT_COMPLETE_NO_EXTENSION
    assert doc.claims


def test_render_run_writes_staging(tmp_path):
    _make_repo(tmp_path)
    _scaffold_pointer_file(tmp_path)
    staging = cli.render_run(
        "bbw_drf", repo_root=tmp_path, staging_root=tmp_path / "stage"
    )
    assert (staging / "note.md").exists()
    assert (staging / "claims.json").exists()


def test_publish_all(tmp_path):
    _make_repo(tmp_path)
    _scaffold_pointer_file(tmp_path)
    summary = cli.publish_all(repo_root=tmp_path, out_root=tmp_path / "out")
    assert summary["n_runs"] == 1
    assert (tmp_path / "out" / "registry" / "runs.jsonl").exists()
    assert (tmp_path / "out" / "runs" / "bbw_drf" / "note.md").exists()


def test_check_missing_run_raises(tmp_path):
    from agents.reporter.manifest import ManifestError

    with pytest.raises(ManifestError):
        cli.check_run("nonexistent", repo_root=tmp_path)
