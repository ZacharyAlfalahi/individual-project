"""WS-C — the codegen reference arm over the oracle set: series read-back tied to the hash P1
archived (a re-used sandbox must never be scored), the two comparators, and a report that
states it is NOT the registered Arm B. Offline: no models, no engine."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.codegen.reference_arm import (  # noqa: E402
    ORACLE_DIVERGENCE_LABEL,
    ReferenceArmError,
    archived_output_sha,
    build_reference_arm,
    load_generated_series,
    render_reference_report,
)

_TH = {
    "arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5},
    "agreement": {"correlation_min": 0.99, "sign_agreement_min": 0.95},
    "divergence_strata": {"high_divergence_corr_lt": 0.90, "medium_divergence_corr_lt": 0.99},
    "taxonomy_sampling": {"strata": ["arm", "divergence_magnitude"]},
    "zoo_list": {"frozen_sha256": "TO_SET"},
    "min_overlap_months": 3,
}
_MODELS = ("model_a", "model_b")


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="portfolio_return")


def _write_csv(path: Path, values) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = _series(values)
    frame = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "portfolio_return": s.values})
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(root: Path, strategy: str, model_id: str, sha: str | None) -> None:
    d = root / strategy / model_id / "code"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({"output_sha256": sha}), encoding="utf-8")


def _run(strategy, model_id, status="ok"):
    return {"strategy": strategy, "model_id": model_id, "sandbox_status": status}


# --- read-back + integrity -------------------------------------------------------

def test_reads_ok_runs_and_ties_each_to_the_archived_hash(tmp_path):
    sandbox, archive = tmp_path / "sandbox", tmp_path / "archive"
    sha = _write_csv(sandbox / "reported" / "drf" / "model_a" / "out" / "portfolio_returns.csv",
                     [0.01, -0.02, 0.03, 0.01])
    _archive(archive, "drf", "model_a", sha)

    series, sources = load_generated_series(
        [_run("drf", "model_a"), _run("drf", "model_b", "wont_run")], sandbox, archive)

    assert len(series["drf"]["model_a"]) == 4
    assert series["drf"]["model_b"] is None
    assert [s.status for s in sources] == ["read", "no_output"]
    assert sources[0].sha256 == sha


def test_a_reused_sandbox_is_refused_not_scored(tmp_path):
    """The fabrication path: a CSV left by an earlier run must never be scored as this one."""
    sandbox, archive = tmp_path / "sandbox", tmp_path / "archive"
    _write_csv(sandbox / "reported" / "drf" / "model_a" / "out" / "portfolio_returns.csv",
               [0.01, 0.02, 0.03])
    _archive(archive, "drf", "model_a", "a" * 64)          # archived a DIFFERENT output

    with pytest.raises(ReferenceArmError, match="stale"):
        load_generated_series([_run("drf", "model_a")], sandbox, archive)


def test_an_ok_run_with_no_csv_is_a_loud_disagreement(tmp_path):
    with pytest.raises(ReferenceArmError, match="disagree"):
        load_generated_series([_run("drf", "model_a")], tmp_path / "s", tmp_path / "a")


def test_two_archived_outputs_for_one_cell_fail_loud(tmp_path):
    archive = tmp_path / "archive"
    for i, sha in enumerate(("a" * 64, "b" * 64)):
        d = archive / "drf" / "model_a" / f"code{i}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"output_sha256": sha}), encoding="utf-8")
    with pytest.raises(ReferenceArmError, match="distinct output hashes"):
        archived_output_sha(archive, "drf", "model_a")


def test_an_unarchived_ok_run_is_refused_not_scored(tmp_path):
    """Without the archived hash nothing ties the CSV to the run that wrote it, and sandbox
    paths are re-used by construction — so an unverifiable series must not be scored."""
    sandbox = tmp_path / "sandbox"
    _write_csv(sandbox / "reported" / "drf" / "model_a" / "out" / "portfolio_returns.csv",
               [0.01, 0.02, 0.03])
    with pytest.raises(ReferenceArmError, match="unverifiable"):
        load_generated_series([_run("drf", "model_a")], sandbox, tmp_path / "archive")


def test_no_archive_entry_returns_none(tmp_path):
    assert archived_output_sha(tmp_path, "drf", "model_a") is None


def test_the_archive_lookup_is_scoped_to_the_phase(tmp_path):
    """The sandbox is phase-scoped but the archive is not: without filtering, a dev run and a
    reported run of the same cell look like one ambiguous cell."""
    archive = tmp_path / "archive"
    for phase, sha in (("dev", "a" * 64), ("reported", "b" * 64)):
        d = archive / "drf" / "model_a" / f"code_{phase}"
        d.mkdir(parents=True)
        (d / "meta.json").write_text(json.dumps({"phase": phase, "output_sha256": sha}),
                                     encoding="utf-8")

    assert archived_output_sha(archive, "drf", "model_a", phase="reported") == "b" * 64
    assert archived_output_sha(archive, "drf", "model_a", phase="dev") == "a" * 64
    with pytest.raises(ReferenceArmError):            # unscoped: genuinely ambiguous
        archived_output_sha(archive, "drf", "model_a")


def test_two_run_records_for_one_cell_fail_loud(tmp_path):
    with pytest.raises(ReferenceArmError, match="two run records"):
        load_generated_series([_run("drf", "model_a", "wont_run"),
                               _run("drf", "model_a", "wont_run")],
                              tmp_path / "s", tmp_path / "a")


# --- the two comparators ---------------------------------------------------------

def test_both_comparators_are_computed_where_both_models_ran():
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series = {
        "drf": {"model_a": base, "model_b": base.copy()},          # a pair -> agreement
        "crf": {"model_a": base, "model_b": None},                 # no pair -> divergence only
    }
    result = build_reference_arm(series, lambda s: base + 0.001, _TH, model_ids=_MODELS)

    assert [a.paper_id for a in result.agreements] == ["drf"]      # crf has no pair
    assert result.agreements[0].arm == "reference"
    assert result.distribution["agreement_rate"] == 1.0
    # divergence rows: drf both models + crf's single model
    assert [(d.paper_id, d.model_id) for d in result.divergences] == [
        ("crf", "model_a"), ("drf", "model_a"), ("drf", "model_b")]
    assert all(d.label == ORACLE_DIVERGENCE_LABEL for d in result.divergences)


def test_a_strategy_with_only_one_series_is_counted_not_dropped():
    """It has no pair to score, so it is absent from the agreement denominator — and must
    therefore be visible in the headline instead of vanishing."""
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    series = {"drf": {"model_a": base, "model_b": base.copy()},
              "crf": {"model_a": base, "model_b": None}}
    result = build_reference_arm(series, lambda s: base + 0.001, _TH, model_ids=_MODELS)

    assert result.unpaired == ("crf",)
    assert result.distribution["n_unpaired_strategies"] == 1
    assert result.distribution["n_scored"] == 1
    report = render_reference_report(result)
    assert "only ONE model's series" in report and "crf" in report


def test_the_oracle_side_is_not_labelled_neither_side_is_truth():
    """On this population the oracle IS the deterministic reference, so the boundary arms'
    label would be false here."""
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    result = build_reference_arm({"drf": {"model_a": base, "model_b": base.copy()}},
                                 lambda s: base + 0.001, _TH, model_ids=_MODELS)
    assert all("neither side is truth" not in d.label for d in result.divergences)
    assert all("oracle is the deterministic reference" in d.label for d in result.divergences)


def test_no_oracle_is_loaded_for_a_strategy_that_generated_nothing():
    def _oracle(strategy):
        raise AssertionError("must not load an oracle for a strategy with no series")

    result = build_reference_arm({"mom6": {"model_a": None, "model_b": None}},
                                 _oracle, _TH, model_ids=_MODELS)
    assert result.divergences == () and result.unpaired == ("mom6",)


def test_a_disagreeing_pair_is_scored_not_dropped():
    a = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    b = _series([-0.01, 0.02, -0.03, -0.02, -0.015])            # mirrored
    result = build_reference_arm({"lrf": {"model_a": a, "model_b": b}},
                                 lambda s: a, _TH, model_ids=_MODELS)
    row = result.agreements[0]
    assert row.agrees is False and row.stratum == "high"
    assert result.distribution["n_scored"] == 1


def test_report_states_it_is_not_the_registered_arm_b_and_carries_the_caveat():
    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    result = build_reference_arm({"drf": {"model_a": base, "model_b": base.copy()}},
                                 lambda s: base, _TH, model_ids=_MODELS)
    report = render_reference_report(result, {"p1 run": "x"})

    assert "not** the registered Arm B" in report
    assert "Inter-model agreement is not correctness." in report     # §6 caveat verbatim
    assert "sharpe" not in report.lower()
    assert ORACLE_DIVERGENCE_LABEL in report
    assert "neither side is truth" not in report      # false on this population
