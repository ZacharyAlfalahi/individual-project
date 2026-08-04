"""Reporter thresholds loader (docs/reporter/reporter_spec_v0.2.md §7): resolves the real block, fails loud."""

from __future__ import annotations

import textwrap

import pytest

from agents.reporter.thresholds import (
    ReporterThresholdError,
    load_reporter_params,
)


def test_loads_real_reporter_block_and_resolves_refs():
    params = load_reporter_params()
    assert params.schema_version == 1
    assert params.rel_tol == pytest.approx(1e-9)
    sc = params.structural_constants
    # Resolved from the referenced sibling blocks (R3), never restated in reporter:.
    assert sc["fdr_q"] == pytest.approx(0.10)
    assert sc["momentum_horizon"] == pytest.approx(6.0)
    assert sc["bootstrap_replicates"] == pytest.approx(1000.0)
    assert sc["materiality_threshold"] == pytest.approx(0.001)


def _write(tmp_path, body: str):
    p = tmp_path / "thresholds.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_missing_reporter_block_raises(tmp_path):
    p = _write(tmp_path, "auditor:\n  fdr:\n    q: 0.10\n")
    with pytest.raises(ReporterThresholdError):
        load_reporter_params(p)


def test_missing_structural_ref_raises(tmp_path):
    p = _write(
        tmp_path,
        """
        auditor:
          fdr:
            q: 0.10
        reporter:
          schema_version: 1
          verify:
            rel_tol: 1.0e-9
          structural_constants:
            fdr_q_ref: auditor.fdr.q
        """,
    )
    with pytest.raises(ReporterThresholdError):
        load_reporter_params(p)


def test_dangling_reference_fails_loud(tmp_path):
    p = _write(
        tmp_path,
        """
        reporter:
          schema_version: 1
          verify:
            rel_tol: 1.0e-9
          structural_constants:
            momentum_horizon_ref: mom6.formation_months
            fdr_q_ref: auditor.fdr.q
            bootstrap_replicates_ref: auditor.bootstrap.n_replicates
            materiality_threshold_ref: auditor.practical_significance.vartheta
        """,
    )
    # None of the referenced sibling blocks exist in this tmp file.
    with pytest.raises(ReporterThresholdError):
        load_reporter_params(p)


def test_missing_rel_tol_raises(tmp_path):
    p = _write(
        tmp_path,
        """
        reporter:
          schema_version: 1
          structural_constants:
            momentum_horizon_ref: mom6.formation_months
        """,
    )
    with pytest.raises(ReporterThresholdError):
        load_reporter_params(p)
