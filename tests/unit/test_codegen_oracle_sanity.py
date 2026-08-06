"""WS-C (P1) — oracle-vs-oracle sanity: every oracle round-tripped through the
sandbox output contract must score RUNS_RIGHT at the exact tier, proving the
comparator's alignment and NaN handling end-to-end (acceptance item)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.codegen.oracles import ORACLES, export_oracle_csv, load_oracle_series
from evaluation.codegen.sandbox import parse_output_csv
from evaluation.codegen.scoring import Verdict, load_scoring_thresholds, score_run

_FACTORS = Path(__file__).resolve().parents[2] / "data" / "development" / "factors"

pytestmark = pytest.mark.skipif(
    not (_FACTORS / "bbw_factors.parquet").exists(),
    reason="factor parquets absent (data/ not built in this checkout)",
)


@pytest.mark.parametrize("strategy", sorted(ORACLES))
def test_oracle_round_trip_scores_runs_right_exact(strategy, tmp_path):
    csv_path = export_oracle_csv(strategy, tmp_path / f"{strategy}.csv")
    series = parse_output_csv(csv_path)
    result = score_run(
        strategy, "ok", None, load_oracle_series(strategy), series,
        load_scoring_thresholds(),
    )
    assert result.verdict is Verdict.RUNS_RIGHT, result.to_dict()
    assert result.exact_tier
    assert result.rung3["n_overlap"] >= 200      # the dev window is ~19 years
