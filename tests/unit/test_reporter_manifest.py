"""R1 run index (docs/reporter/reporter_spec_v0.2.md §4): pointer parsing, path-traversal defence, hash check."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from agents.reporter.manifest import (
    ManifestError,
    parse_manifest_text,
    scaffold_pointer,
)
from shared.reporting.claims import ArtefactType


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _repo(tmp_path: Path):
    d = tmp_path / "runs" / "g3" / "bbw"
    d.mkdir(parents=True)
    spec = d / "spec_0.json"
    spec.write_text('{"header": {}}')
    qd = tmp_path / "results" / "quant" / "run_x"
    qd.mkdir(parents=True)
    res = qd / "drf.json"
    res.write_text('{"summary": {}}')
    return tmp_path, spec, res


def _valid_text(tmp_path: Path, spec: Path, res: Path, **ov) -> str:
    run_id = ov.get("run_id", "bbw_drf")
    join = ov.get("join", "asserted manually; no mechanical join exists")
    phase = ov.get("phase", "D")
    spec_path = ov.get("spec_path", str(spec.relative_to(tmp_path)))
    return f"""
reporter_run_id: {run_id}
schema_version: 1
paper_id: BBW
strategy_id: drf
phase: {phase}
join_rationale: {join}
artefacts:
  spec:
    path: {spec_path}
    sha256: "{_sha(spec)}"
    schema_version: "1.1"
  quant_result:
    path: {res.relative_to(tmp_path)}
    sha256: "{_sha(res)}"
"""


def test_valid_pointer_parses_and_resolves_types(tmp_path):
    root, spec, res = _repo(tmp_path)
    m = parse_manifest_text(
        _valid_text(root, spec, res), repo_root=root, verify_hashes=True
    )
    assert m.reporter_run_id == "bbw_drf"
    assert m.artefacts["spec"].artefact_type is ArtefactType.STRATEGY_SPEC
    assert m.artefacts["spec"].producing_component == "librarian"
    assert m.artefacts["quant_result"].artefact_type is ArtefactType.STRATEGY_RESULT
    assert m.artefacts["spec"].schema_version == "1.1"


def test_bad_run_id_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    with pytest.raises(ManifestError):
        parse_manifest_text(
            _valid_text(root, spec, res, run_id="BBW/../etc"),
            repo_root=root,
            verify_hashes=True,
        )


def test_parent_traversal_path_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res, spec_path="../escape.json")
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_absolute_path_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res, spec_path="/etc/passwd")
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_symlink_escape_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    outside = tmp_path.parent / "outside_escape.json"
    outside.write_text('{"x": 1}')
    link = root / "link.json"
    os.symlink(outside, link)
    text = _valid_text(root, spec, res, spec_path="link.json")
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_duplicate_artefact_key_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res) + f"""  spec:
    path: {spec.relative_to(root)}
    sha256: "{_sha(spec)}"
"""
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_unknown_artefact_key_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res) + f"""  bogus:
    path: {res.relative_to(root)}
    sha256: "{_sha(res)}"
"""
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_missing_join_rationale_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res).replace(
        "join_rationale: asserted manually; no mechanical join exists\n", ""
    )
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_missing_spec_artefact_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = f"""
reporter_run_id: bbw_drf
schema_version: 1
paper_id: BBW
strategy_id: drf
phase: D
join_rationale: asserted
artefacts:
  quant_result:
    path: {res.relative_to(root)}
    sha256: "{_sha(res)}"
"""
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_hash_mismatch_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    good = _sha(spec)
    bad = ("0" if good[0] != "0" else "1") + good[1:]
    text = _valid_text(root, spec, res).replace(good, bad)
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_invalid_phase_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    with pytest.raises(ManifestError):
        parse_manifest_text(
            _valid_text(root, spec, res, phase="X"), repo_root=root, verify_hashes=True
        )


def test_declared_stages_parse(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res) + """stages:
  scientist:
    status: not_applicable
    evidence: entry_rule_empty
"""
    m = parse_manifest_text(text, repo_root=root, verify_hashes=True)
    assert m.declared_stages["scientist"].status == "not_applicable"


def test_declared_stage_bad_status_rejected(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = _valid_text(root, spec, res) + """stages:
  scientist:
    status: maybe
    evidence: x
"""
    with pytest.raises(ManifestError):
        parse_manifest_text(text, repo_root=root, verify_hashes=True)


def test_scaffold_roundtrips_through_parse(tmp_path):
    root, spec, res = _repo(tmp_path)
    text = scaffold_pointer(
        reporter_run_id="bbw_drf",
        paper_id="BBW",
        strategy_id="drf",
        phase="D",
        artefacts={
            "spec": str(spec.relative_to(root)),
            "quant_result": str(res.relative_to(root)),
        },
        repo_root=root,
    )
    m = parse_manifest_text(text, repo_root=root, verify_hashes=True)
    assert m.artefacts["spec"].sha256 == _sha(spec)
    assert m.artefacts["quant_result"].sha256 == _sha(res)
