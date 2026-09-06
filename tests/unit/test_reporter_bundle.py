"""R2 bundle + loader (docs/reporter/reporter_spec_v0.2.md §5, C9): stage records, reportability, git normalisation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from agents.reporter.bundle import (
    CodeVersion,
    ReportabilityStatus,
    StageStatus,
    derive_reportability,
    load_bundle,
    load_phase_stacks,
)
from agents.reporter.manifest import parse_manifest_text
from shared.reporting.claims import ArtefactType

_CV = CodeVersion(full="reporterfullsha", short="reporter")

_PHASE_F_IDS, _PHASE_D_IDS = load_phase_stacks()
_PF_STR = ",".join(sorted(_PHASE_F_IDS))  # the recorded reportable stack
_PD_STR = ",".join(sorted(_PHASE_D_IDS))  # the recorded free-dev stack


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))
    return path


def _repo(tmp_path: Path, *, audit_scope="COMPLETE", with_refusal=False, model_ids: str | None = None):
    files: dict[str, Path] = {}
    header = {"model_ids": model_ids} if model_ids else {}
    files["spec"] = _write(tmp_path / "runs/g3/bbw/spec_0.json", {"header": header})
    files["quant_result"] = _write(
        tmp_path / "results/quant/run_x/drf.json", {"summary": {"sharpe": 0.4}}
    )
    files["quant_run_log"] = _write(
        tmp_path / "results/quant/run_x/run_log.json",
        {"git_commit": "a" * 40, "git_short": "aaaaaaa"},
    )
    audit_obj = {"strategy_label": "drf"}
    if audit_scope is not None:
        audit_obj["audit_scope"] = audit_scope
    files["audit_report"] = _write(
        tmp_path / "results/auditor/run_y/drf_core.json", audit_obj
    )
    files["audit_run_log"] = _write(
        tmp_path / "results/auditor/run_y/run_log.json",
        {"git_commit": "1111111", "auditor_prereg_tag": "auditor-prereg"},
    )
    if with_refusal:
        files["config_refusal"] = _write(
            tmp_path / "results/quant/run_x/drf_refusal.json",
            {"code": "MISSING_BINDING", "strategy_id": "drf"},
        )
    return files


def _pointer(tmp_path: Path, files: dict[str, Path], *, phase="D", stages_block=""):
    lines = [
        "reporter_run_id: bbw_drf",
        "schema_version: 1",
        "paper_id: BBW",
        "strategy_id: drf",
        f"phase: {phase}",
        "join_rationale: asserted by author; no mechanical join exists",
        "artefacts:",
    ]
    for key, p in files.items():
        lines.append(f"  {key}:")
        lines.append(f"    path: {p.relative_to(tmp_path)}")
        lines.append(f'    sha256: "{_sha(p)}"')
    return "\n".join(lines) + "\n" + stages_block


def _load(tmp_path, files, **kw):
    m = parse_manifest_text(
        _pointer(tmp_path, files, **kw), repo_root=tmp_path, verify_hashes=True
    )
    return load_bundle(m, repo_root=tmp_path, code_version=_CV)


def test_stage_records_from_present_artefacts(tmp_path):
    b = _load(tmp_path, _repo(tmp_path))
    assert b.stages["extraction"].status is StageStatus.SUCCEEDED
    assert b.stages["execution"].status is StageStatus.SUCCEEDED
    assert b.stages["compilation"].status is StageStatus.SUCCEEDED
    assert b.stages["audit"].status is StageStatus.SUCCEEDED
    # Not derivable from disk (C9) -> UNOBSERVED, with an explicit reason.
    assert b.stages["scientist"].status is StageStatus.UNOBSERVED
    assert b.stages["scientist"].evidence.kind == "absent"
    assert b.stages["diagnostics_capacity"].status is StageStatus.UNOBSERVED
    assert b.stages["holdout"].status is StageStatus.UNOBSERVED


def test_audit_scope_refused_is_a_refusal(tmp_path):
    b = _load(tmp_path, _repo(tmp_path, audit_scope="REFUSED"))
    assert b.stages["audit"].status is StageStatus.REFUSED
    assert b.stages["audit"].refusal_code == "AUDIT_SCOPE_REFUSED"


def test_config_refusal_makes_compilation_refused(tmp_path):
    b = _load(tmp_path, _repo(tmp_path, with_refusal=True))
    assert b.stages["compilation"].status is StageStatus.REFUSED
    assert b.stages["compilation"].refusal_code == "MISSING_BINDING"


def test_reportability_phase_d(tmp_path):
    # Recorded free-dev stack + declared phase D -> non-reportable.
    b = _load(tmp_path, _repo(tmp_path, model_ids=_PD_STR), phase="D")
    assert b.reportability is ReportabilityStatus.NON_REPORTABLE_PHASE_D


def test_reportability_phase_f(tmp_path):
    # Recorded reportable stack + declared phase F -> reportable.
    b = _load(tmp_path, _repo(tmp_path, model_ids=_PF_STR), phase="F")
    assert b.reportability is ReportabilityStatus.REPORTABLE


def test_reportability_mislabel_cannot_upgrade_a_dev_run(tmp_path):
    # A pointer declaring F over a run whose trace records the free-dev stack must NOT become
    # reportable: reportability follows what actually ran, not the requester's label.
    b = _load(tmp_path, _repo(tmp_path, model_ids=_PD_STR), phase="F")
    assert b.reportability is ReportabilityStatus.NON_REPORTABLE_PHASE_D


def test_reportability_absent_trace_fails_closed(tmp_path):
    # No recorded model ids -> nothing to verify what ran -> fail closed, even under phase F.
    b = _load(tmp_path, _repo(tmp_path), phase="F")
    assert b.reportability is ReportabilityStatus.NON_REPORTABLE_OTHER


def test_reportability_unrecognised_stack_fails_closed():
    assert (
        derive_reportability(
            "F", "some-unpinned-model,another", phase_f_ids=_PHASE_F_IDS, phase_d_ids=_PHASE_D_IDS
        )
        is ReportabilityStatus.NON_REPORTABLE_OTHER
    )


def test_git_stamp_normalisation_never_fabricates_full(tmp_path):
    b = _load(tmp_path, _repo(tmp_path))
    # Quant carried full + short.
    assert b.upstream_stamps.quant.full == "a" * 40
    assert b.upstream_stamps.quant.short == "aaaaaaa"
    assert b.upstream_stamps.quant.source_form == "full+short"
    # Auditor carried a short-only SHA under git_commit — full must stay None (never invented).
    assert b.upstream_stamps.audit.full is None
    assert b.upstream_stamps.audit.short == "1111111"
    assert b.upstream_stamps.audit.source_form == "short"
    assert b.upstream_stamps.auditor_prereg_tag == "auditor-prereg"


def test_declared_stage_overrides_derived(tmp_path):
    stages_block = """stages:
  scientist:
    status: not_applicable
    evidence: entry rule produced no failing toggle (asserted)
"""
    b = _load(tmp_path, _repo(tmp_path), stages_block=stages_block)
    assert b.stages["scientist"].status is StageStatus.NOT_APPLICABLE
    assert b.stages["scientist"].evidence.kind == "declared_in_pointer"


def test_docs_hold_parsed_serialised_dicts(tmp_path):
    b = _load(tmp_path, _repo(tmp_path))
    assert b.doc(ArtefactType.AUDIT_REPORT)["audit_scope"] == "COMPLETE"
    assert b.has(ArtefactType.STRATEGY_RESULT)
    assert b.code_version.short == "reporter"


def test_missing_audit_scope_fails_closed_to_unobserved(tmp_path):
    # Review fix 6: an audit core present but missing audit_scope must not read as success.
    b = _load(tmp_path, _repo(tmp_path, audit_scope=None))
    assert b.stages["audit"].status is StageStatus.UNOBSERVED
    assert b.stages["audit"].evidence.kind == "malformed"


def test_unknown_declared_stage_raises(tmp_path):
    # Review fix 5: a typo'd stage name must not be silently discarded.
    from agents.reporter.manifest import ManifestError

    stages_block = """stages:
  sciontist:
    status: not_applicable
    evidence: typo
"""
    m = parse_manifest_text(
        _pointer(tmp_path, _repo(tmp_path), stages_block=stages_block),
        repo_root=tmp_path,
        verify_hashes=True,
    )
    with pytest.raises(ManifestError):
        load_bundle(m, repo_root=tmp_path, code_version=_CV)
