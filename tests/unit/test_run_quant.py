"""
Validation for `scripts/run_quant.py` — the RQ2 "B1" StrategySpec -> run driver.

Two load-bearing behaviours, both exercised end-to-end:

  (a) the three supported anchors {str, drf, mom6} COMPILE (adapt) and RUN on the real
      development panel with ZERO refusals and finite headline means — the B1 "done"
      criterion. This reads only data/development/ (the autouse holdout guard in
      tests/conftest.py would fail the test if any holdout path were touched), and the
      run window landing inside 2002-2021 is asserted as a second, data-level proof that
      the holdout is untouched.

  (b) a deliberately UNSUPPORTED spec (a registry-version drift the adapter refuses upfront)
      produces an `AdaptResult.refused` that `record_anchor` records as a TYPED refusal, and
      `build_coverage` tallies it into the RQ2 coverage denominator — the coverage branch,
      exercised non-trivially (the refused record must never touch the panel, so an EMPTY
      panel is passed and must not raise).
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

import run_quant
from agents.librarian.adapter.adapt import adapt_spec
from agents.quant.library.run_config import corrected
from evaluation.gold_specs.gold_loader import load_gold_spec

ANCHORS = ("str", "drf", "mom6")


# One real dev-panel materialisation shared across the (a) tests (the view() is the
# expensive step). Module-scoped so it is paid once. Reads data/development/ only.
@pytest.fixture(scope="module")
def dev_inputs():
    panel, subs = run_quant.load_inputs(corrected())
    return panel, subs


# --------------------------------------------------------------------------
# (a) the three anchors compile + run, 0 refusals, finite means
# --------------------------------------------------------------------------

@pytest.mark.parametrize("anchor", ANCHORS)
def test_anchor_compiles_and_runs_finite(anchor, dev_inputs):
    panel, subs = dev_inputs
    result = run_quant.adapt_anchor(anchor, subs)
    assert not result.refused, f"{anchor} unexpectedly refused: {result.to_dict()['refusals']}"

    record = run_quant.record_anchor(anchor, result, panel)
    assert record["status"] == "run"
    assert record["anchor"] == anchor

    summary = record["summary"]
    # Finite headline mean + a real sample; a non-finite mean is coerced to None by the driver.
    assert summary["mean_pct_per_month"] is not None
    assert summary["t_stat"] is not None
    assert summary["n_months"] > 0
    # mean_pct_per_month is exactly average * 100 (the anchor-gold unit).
    assert summary["mean_pct_per_month"] == pytest.approx(summary["average"] * 100)

    # Data-level holdout proof: the run window lands inside the development split (2002-2021).
    assert pd.Timestamp(summary["first_date"]).year >= 2002
    assert pd.Timestamp(summary["last_date"]).year <= 2021


def test_all_three_anchors_zero_refusals_full_coverage(dev_inputs):
    panel, subs = dev_inputs
    records = [
        run_quant.record_anchor(a, run_quant.adapt_anchor(a, subs), panel) for a in ANCHORS
    ]
    coverage = run_quant.build_coverage(records)

    assert coverage["n_candidates"] == 3
    assert coverage["refused"] == 0
    assert coverage["compiled"] == 3
    assert coverage["run"] == 3
    for a in ANCHORS:
        assert coverage["per_anchor"][a]["status"] == "run"
        assert coverage["per_anchor"][a]["refusal_codes"] == []


# --------------------------------------------------------------------------
# (b) an unsupported spec -> typed refusal, recorded into coverage
# --------------------------------------------------------------------------

def _registry_drifted_result():
    """A refused `AdaptResult`: take a real gold spec and stamp a nonexistent
    registry_version, which `adapt_spec` refuses upfront (REVIEW_REQUIRED) because the
    concept->column table can no longer be trusted to mirror the spec's concepts. A
    structural, deterministic stand-in for an unsupported family."""
    spec = load_gold_spec("str")
    drifted = dataclasses.replace(
        spec, header=dataclasses.replace(spec.header, registry_version="v0.0-nonexistent")
    )
    result = adapt_spec(drifted)
    assert result.refused  # precondition for the branch under test
    return result


def test_unsupported_spec_recorded_as_typed_refusal():
    result = _registry_drifted_result()

    # An EMPTY panel: a refused compile must short-circuit before any run, so the panel is
    # never read. If record_anchor tried to run, an empty panel would raise — proving the
    # branch does not silently execute a refused strategy.
    record = run_quant.record_anchor("drifted_str", result, pd.DataFrame())

    assert record["status"] == "refused"
    assert record["anchor"] == "drifted_str"
    assert len(record["refusals"]) >= 1
    codes = [r["code"] for r in record["refusals"]]
    assert "REVIEW_REQUIRED" in codes
    assert record["refusals"][0]["field"] == "registry_version"


def test_refusal_counts_into_coverage_denominator():
    refused = run_quant.record_anchor("drifted_str", _registry_drifted_result(), pd.DataFrame())
    coverage = run_quant.build_coverage([refused])

    assert coverage["n_candidates"] == 1
    assert coverage["refused"] == 1
    assert coverage["compiled"] == 0
    assert coverage["run"] == 0
    assert coverage["per_anchor"]["drifted_str"]["status"] == "refused"
    assert "REVIEW_REQUIRED" in coverage["per_anchor"]["drifted_str"]["refusal_codes"]
