"""The RQ2 coverage driver (2026-09-04): glue over tested pieces, pinned at the
seams -- refusal-code collection, the not_run expansion over registered names,
and (machine-local) the recorded headline of the first real scoring run."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_t3_coverage import (  # noqa: E402
    _NOT_RUN_PAPERS,
    observed_rows,
    refusal_codes_of,
)


def _refusal(value):
    return SimpleNamespace(code=SimpleNamespace(value=value))


def test_refusal_codes_collects_strategy_and_leg_codes():
    result = SimpleNamespace(
        refusals=(_refusal("REVIEW_REQUIRED"),),
        leg_calls=(SimpleNamespace(refused=True, result=_refusal("UNSUPPORTED_FAMILY")),
                   SimpleNamespace(refused=False, result=None)),
    )
    assert refusal_codes_of(result) == ["REVIEW_REQUIRED", "UNSUPPORTED_FAMILY"]


def test_not_run_papers_is_the_dfps_review_exit():
    assert set(_NOT_RUN_PAPERS) == {"DFPS_2026"}       # the 2026-09-03 typed outcome


# --- CI-7 (2026-09-06): per-construction assembly events in the coverage ----

def _events_dir(tmp_path, events):
    d = tmp_path / "run_dir"
    d.mkdir(parents=True)
    (d / "events.json").write_text(json.dumps(events), encoding="utf-8")
    return d


def test_assembly_events_yield_typed_not_run_rows_without_double_count(tmp_path):
    # VaR carries a typed per-construction event; the blanket expansion must
    # cover the REMAINING DFPS registered names only (26), never re-add VaR.
    d = _events_dir(tmp_path, [
        {"kind": "assembly_incomplete", "paper_id": "DFPS_2026",
         "construction_name": "VaR", "detail": "sort_signal did not resolve",
         "routing": "review"},
    ])
    rows, detail = observed_rows(
        [d], None, None,
        not_run_papers={"DFPS_2026": "review exit", "BBW_2021": "test blanket"},
    )
    var_rows = [r for r in rows if r["name"] == "VaR"]
    assert len(var_rows) == 1
    assert var_rows[0]["outcome"] == "not_run"
    assert var_rows[0]["extraction_event"] == "assembly_incomplete"
    assert detail["VaR"]["reason"] == "sort_signal did not resolve"
    # every registered non-excluded construction exactly once (the frozen 32)
    assert len(rows) == 32
    assert len({(r["paper_id"], r["name"]) for r in rows}) == 32


def test_unregistered_event_constructions_never_get_denominator_rows(tmp_path):
    # CMKT is a DFPS auxiliary row -- no coverage label, so no row; the event
    # is retained in the detail block instead of crashing the scorer.
    d = _events_dir(tmp_path, [
        {"kind": "assembly_incomplete", "paper_id": "DFPS_2026",
         "construction_name": "CMKT", "detail": "aux", "routing": "review"},
    ])
    rows, detail = observed_rows(
        [d], None, None,
        not_run_papers={"DFPS_2026": "review exit", "BBW_2021": "test blanket"},
    )
    assert not [r for r in rows if r["name"] == "CMKT"]
    assert any(ev["construction_name"] == "CMKT" for ev in detail["__events__"])
    assert len(rows) == 32


def test_duplicate_rows_fail_loud(tmp_path):
    ev = {"kind": "assembly_incomplete", "paper_id": "DFPS_2026",
          "construction_name": "VaR", "detail": "d", "routing": "review"}
    d1 = _events_dir(tmp_path / "a", [ev])
    d2 = _events_dir(tmp_path / "b", [ev])
    with pytest.raises(RuntimeError, match="duplicate outcome row"):
        observed_rows([d1, d2], None, None,
                      not_run_papers={"DFPS_2026": "x", "BBW_2021": "y"})


def test_incomplete_denominator_fails_loud(tmp_path):
    # BBW_2021 contributes nothing (no dir, no blanket) -> the frozen-32
    # headline is not computable and the driver must say so.
    with pytest.raises(RuntimeError, match="no outcome row"):
        observed_rows([], None, None, not_run_papers={"DFPS_2026": "x"})


@pytest.mark.skipif(
    not (_REPO_ROOT / "results" / "rq2_coverage.json").exists(),
    reason="first real coverage run artefact absent",
)
def test_first_real_run_headline_pins():
    d = json.loads((_REPO_ROOT / "results" / "rq2_coverage.json").read_text())
    s = d["score"]
    assert s["denominator"] == 32                       # frozen (CI-5)
    assert s["C_end_to_end_full"] == 0.0
    assert s["n_unobserved"] == 27                      # all DFPS rows
    assert len(s["false_refusals"]) == 3                # the three bond rows
    assert not s.get("fir_events")                      # zero false implementations
