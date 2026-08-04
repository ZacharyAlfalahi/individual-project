"""R7 registry + atomic publication (docs/reporter/reporter_spec_v0.2.md §10, §11)."""

from __future__ import annotations


import pytest

from _reporter_fixtures import make_bundle
from agents.reporter.bundle import StageStatus
from agents.reporter.registry import build_registry, publish
from agents.reporter.renderer import render_note
from shared.reporting.claims import ArtefactType

_TOGGLES = ("meas_err", "stale_price", "survivorship", "lib_gap", "lab_trim")


def _audit_doc(*, meas_err_nan=False):
    doe = {t: 0.0 for t in _TOGGLES}
    if meas_err_nan:
        doe["meas_err"] = float("nan")
    return {
        "audit_scope": "COMPLETE",
        "runnable_toggles": list(_TOGGLES),
        "shapley": {
            "shapley_share_of_registered_endpoint_gap": {t: 0.0 for t in _TOGGLES}
        },
        "saturated_bases": {"doe_effects": doe},
    }


def _synthetic_bundle(*, run_id="bbw_drf", meas_err_nan=False):
    docs = {
        ArtefactType.STRATEGY_SPEC: {"paper_facts": None},
        ArtefactType.STRATEGY_RESULT: {
            "summary": {"sharpe": 0.5, "t_stat": 2.0, "average": 0.001}
        },
        ArtefactType.AUDIT_REPORT: _audit_doc(meas_err_nan=meas_err_nan),
    }
    stage_status = {
        "extraction": StageStatus.SUCCEEDED,
        "compilation": StageStatus.SUCCEEDED,
        "execution": StageStatus.SUCCEEDED,
        "audit": StageStatus.SUCCEEDED,
    }
    return make_bundle(docs=docs, stage_status=stage_status, run_id=run_id)


def test_publish_writes_all_files(tmp_path):
    out = tmp_path / "reporter"
    result = publish(
        [_synthetic_bundle(run_id="a"), _synthetic_bundle(run_id="b")],
        out_root=out,
        staging_root=tmp_path / "staging",
    )
    assert result.n_runs == 2
    for name in ("runs", "stages", "audit_toggles", "artefacts", "claims_index"):
        assert (out / "registry" / f"{name}.jsonl").exists()
    assert (out / "registry" / "manifest.json").exists()
    assert (out / "runs" / "a" / "note.md").exists()
    assert (out / "runs" / "a" / "claims.json").exists()
    # runs.jsonl has exactly two rows.
    lines = (out / "registry" / "runs.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_rebuild_is_byte_identical(tmp_path):
    # INV-8: same inputs + same code -> byte-identical registry.
    b = [_synthetic_bundle(run_id="a"), _synthetic_bundle(run_id="b")]
    publish(b, out_root=tmp_path / "r1", staging_root=tmp_path / "s1")
    publish(b, out_root=tmp_path / "r2", staging_root=tmp_path / "s2")
    for name in ("runs", "stages", "audit_toggles", "artefacts", "claims_index"):
        f1 = (tmp_path / "r1" / "registry" / f"{name}.jsonl").read_bytes()
        f2 = (tmp_path / "r2" / "registry" / f"{name}.jsonl").read_bytes()
        assert f1 == f2, name
    m1 = (tmp_path / "r1" / "registry" / "manifest.json").read_bytes()
    m2 = (tmp_path / "r2" / "registry" / "manifest.json").read_bytes()
    assert m1 == m2


def test_republish_over_existing_swaps_atomically(tmp_path):
    out = tmp_path / "reporter"
    publish([_synthetic_bundle(run_id="a")], out_root=out, staging_root=tmp_path / "s1")
    assert len((out / "registry" / "runs.jsonl").read_text().splitlines()) == 1
    # Republish with two runs; the swap replaces the previous publication cleanly.
    publish(
        [_synthetic_bundle(run_id="a"), _synthetic_bundle(run_id="b")],
        out_root=out,
        staging_root=tmp_path / "s2",
    )
    assert len((out / "registry" / "runs.jsonl").read_text().splitlines()) == 2
    assert not (out.with_name(out.name + ".prev")).exists()


def test_nan_is_encoded_not_rejected(tmp_path):
    out = tmp_path / "reporter"
    publish(
        [_synthetic_bundle(run_id="a", meas_err_nan=True)],
        out_root=out,
        staging_root=tmp_path / "s",
    )
    toggles = (out / "registry" / "audit_toggles.jsonl").read_text()
    assert '"__nan__":true' in toggles
    assert "NaN" not in toggles  # never a bare (invalid) NaN literal


def test_audit_toggle_rows_carry_bias_class_and_dummy_reason():
    # ADR §5.1/§5.4: every audit_toggles row records the global bias_class and the
    # per-strategy dummy_reason. On this all-runnable synthetic audit, meas_err is the
    # sole data-quality correction and no toggle is a dummy.
    bundle = _synthetic_bundle(run_id="a")
    reg = build_registry([(bundle, render_note(bundle))])
    rows = {r["toggle"]: r for r in reg.rows["audit_toggles"]}
    assert rows["meas_err"]["bias_class"] == "data_quality_correction"
    for t in ("stale_price", "survivorship", "lib_gap", "lab_trim"):
        assert rows[t]["bias_class"] == "methodological_construction"
    # all runnable, no measured no-op declared in the synthetic invariance => no dummies
    assert all(r["dummy_reason"] is None for r in reg.rows["audit_toggles"])


def test_referential_integrity_enforced():
    bundle = _synthetic_bundle(run_id="a")
    doc = render_note(bundle)
    reg = build_registry([(bundle, doc)])
    # Every stage/claim/artefact row references the one declared run.
    known = {r["reporter_run_id"] for r in reg.rows["runs"]}
    for name in ("stages", "audit_toggles", "artefacts", "claims_index"):
        for row in reg.rows[name]:
            assert row["reporter_run_id"] in known


def test_duplicate_run_id_rejected():
    b = _synthetic_bundle(run_id="dup")
    d = render_note(b)
    with pytest.raises(ValueError):
        build_registry([(b, d), (b, d)])


def test_jsonl_lines_end_with_single_lf(tmp_path):
    out = tmp_path / "reporter"
    publish([_synthetic_bundle(run_id="a")], out_root=out, staging_root=tmp_path / "s")
    raw = (out / "registry" / "runs.jsonl").read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r" not in raw
