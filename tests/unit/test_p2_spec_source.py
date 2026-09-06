"""WS-C (P2) — real Phase-F spec provenance loader + matcher. Offline, no LLM.

Covers the phase=report gate (no silent downgrade), byte-exact member matching with
full provenance, and the fail-loud guards: an unmatched spec, a slug-only near-miss
(no fuzzy matching), a duplicate (paper,label) key, and a missing manifest.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from evaluation.codegen.p2_spec_source import (
    P2SpecSourceError,
    load_real_specs,
    match_specs_to_members,
    slug,
)


def _write_run(run_dir, phase="report", commit="abc1234", specs=(), declare_outputs=True):
    """Write a fixture run dir. By default the manifest DECLARES its spec outputs (with sha256),
    matching the frozen runs, so the provenance-integrity guard is exercised. Pass
    ``declare_outputs=False`` to model an older manifest with no ``outputs`` block."""
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for i, (paper_id, label) in enumerate(specs):
        content = json.dumps({"header": {"paper_id": paper_id, "strategy_label": {"value": label}}})
        p = run_dir / f"spec_{i}.json"
        p.write_text(content, encoding="utf-8")
        outputs[str(p)] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest = {"operational_profile": {"phase": phase}, "code": {"commit": commit}}
    if declare_outputs:
        manifest["outputs"] = outputs
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _rows(*pairs):
    # pairs of (paper, name) -> gold member rows
    return [{"paper_id": f"{p}::{slug(n)}", "paper": p, "name": n} for p, n in pairs]


# --- phase gate: a non-report run can NEVER seed a reportable census -----------------

def test_non_report_phase_is_a_build_error(tmp_path):
    run = _write_run(tmp_path / "dev_run", phase="dev",
                     specs=[("BBW_2021", "Alpha strategy")])
    with pytest.raises(P2SpecSourceError, match="not 'report'"):
        load_real_specs([run])


def test_missing_manifest_is_a_build_error(tmp_path):
    empty = tmp_path / "no_manifest"
    empty.mkdir()
    with pytest.raises(P2SpecSourceError, match="no run_manifest"):
        load_real_specs([empty])


# --- byte-exact load + provenance ---------------------------------------------------

def test_load_keys_by_paper_and_label_with_provenance(tmp_path):
    run = _write_run(tmp_path / "bbw", commit="b68cc6c",
                     specs=[("BBW_2021", "Alpha strategy"), ("BBW_2021", "Beta strategy")])
    specs, prov = load_real_specs([run])
    assert set(specs) == {("bbw_2021", "Alpha strategy"), ("bbw_2021", "Beta strategy")}
    p = prov[("bbw_2021", "Alpha strategy")]
    assert set(p) == {"run_dir", "spec_file", "spec_sha256", "manifest_phase", "code_commit"}
    assert p["manifest_phase"] == "report" and p["code_commit"] == "b68cc6c"
    assert len(p["spec_sha256"]) == 64


def test_a_review_exit_dir_contributes_nothing_but_is_not_an_error(tmp_path):
    review = _write_run(tmp_path / "dfps", specs=[])       # phase=report, zero specs
    specs, prov = load_real_specs([review])
    assert specs == {} and prov == {}


def test_match_byte_exact_maps_specs_to_member_ids(tmp_path):
    run = _write_run(tmp_path / "bbw",
                     specs=[("BBW_2021", "Alpha strategy"), ("BBW_2021", "Beta strategy")])
    specs, prov = load_real_specs([run])
    rows = _rows(("bbw_2021", "Alpha strategy"), ("bbw_2021", "Beta strategy"),
                 ("bbw_2021", "Gamma auxiliary"))
    by_id, prov_by_id = match_specs_to_members(specs, prov, rows)
    assert set(by_id) == {"bbw_2021::alpha_strategy", "bbw_2021::beta_strategy"}
    assert prov_by_id["bbw_2021::alpha_strategy"]["manifest_phase"] == "report"


# --- fail-loud matching guards ------------------------------------------------------

def test_unmatched_spec_is_a_build_error(tmp_path):
    run = _write_run(tmp_path / "bbw", specs=[("BBW_2021", "Unlisted strategy")])
    specs, prov = load_real_specs([run])
    rows = _rows(("bbw_2021", "Alpha strategy"))
    with pytest.raises(P2SpecSourceError, match="spec_unmatched_to_member"):
        match_specs_to_members(specs, prov, rows)


def test_slug_near_miss_is_a_build_error_no_fuzzy_matching(tmp_path):
    # label slugs to the same id as the gold name but is not byte-exact -> hard error
    run = _write_run(tmp_path / "bbw", specs=[("BBW_2021", "Alpha   strategy!")])
    specs, prov = load_real_specs([run])
    rows = _rows(("bbw_2021", "Alpha strategy"))
    assert slug("Alpha   strategy!") == slug("Alpha strategy")     # same slug
    with pytest.raises(P2SpecSourceError, match="by slug but NOT byte-exactly"):
        match_specs_to_members(specs, prov, rows)


def test_two_specs_one_member_is_a_build_error(tmp_path):
    # same (paper,label) across two dirs -> duplicate key caught at load time
    r1 = _write_run(tmp_path / "a", specs=[("BBW_2021", "Alpha strategy")])
    r2 = _write_run(tmp_path / "b", specs=[("BBW_2021", "Alpha strategy")])
    with pytest.raises(P2SpecSourceError, match="one spec per construction"):
        load_real_specs([r1, r2])


# --- provenance integrity vs the manifest-declared outputs --------------------------

def test_stray_spec_not_in_manifest_outputs_is_a_build_error(tmp_path):
    # the guard against a spurious spec silently flipping a member to routed
    run = _write_run(tmp_path / "bbw", specs=[("BBW_2021", "Alpha strategy")])
    (run / "spec_9.json").write_text(                       # a stray file, not declared
        json.dumps({"header": {"paper_id": "BBW_2021", "strategy_label": {"value": "Sneaky"}}}),
        encoding="utf-8")
    with pytest.raises(P2SpecSourceError, match="provenance integrity"):
        load_real_specs([run])


def test_a_rewritten_spec_fails_the_sha_check(tmp_path):
    run = _write_run(tmp_path / "bbw", specs=[("BBW_2021", "Alpha strategy")])
    # rewrite the spec AFTER the manifest declared its sha -> integrity break
    (run / "spec_0.json").write_text(
        json.dumps({"header": {"paper_id": "BBW_2021", "strategy_label": {"value": "Alpha strategy"},
                               "tampered": True}}),
        encoding="utf-8")
    with pytest.raises(P2SpecSourceError, match="rewritten after the run"):
        load_real_specs([run])


def test_a_manifest_without_outputs_skips_the_integrity_check(tmp_path):
    # older schema: no `outputs` block -> cannot integrity-check, but still loads
    run = _write_run(tmp_path / "bbw", specs=[("BBW_2021", "Alpha strategy")],
                     declare_outputs=False)
    specs, prov = load_real_specs([run])
    assert set(specs) == {("bbw_2021", "Alpha strategy")}
