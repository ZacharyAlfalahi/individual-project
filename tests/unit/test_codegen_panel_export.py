"""WS-C (P1) — the corr-family engine-panel export. The strong check: correct construction on the
exported panel must reproduce EVERY oracle at the exact tier (proves the panel is byte-faithful to
what the frozen oracle builders consumed). Skips cleanly when the dev panel/oracles are absent."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.codegen.panel_export import SCHEMA_COLUMNS, build_codegen_panel  # noqa: E402


def _panel_or_skip():
    try:
        return build_codegen_panel()
    except FileNotFoundError as exc:
        pytest.skip(f"dev panel/signals absent: {exc}")


def test_panel_has_the_frozen_schema_and_no_holdout_dates():
    panel = _panel_or_skip()
    assert list(panel.columns) == list(SCHEMA_COLUMNS)
    assert not panel.duplicated(subset=["cusip", "date"]).any()
    # dev window only — the holdout starts 2022-01.
    assert pd.Timestamp(panel["date"].max()) < pd.Timestamp("2022-01-01")


def test_panel_reproduces_every_oracle_at_exact_tier():
    panel = _panel_or_skip()
    from evaluation.codegen.oracles import load_oracle_series
    from evaluation.codegen.scoring import Verdict, load_scoring_thresholds, score_run
    from agents.quant.library.characteristic_sort import run_characteristic_sort
    from agents.quant.library.overlap import run_with_holding_period
    from agents.quant.library.bbw_factors import run_bbw_factor, compose_crf, CRF_COMPONENTS
    import build_str
    import build_mom6

    th = load_scoring_thresholds()

    def ser(df, col="strategy_ret"):
        return pd.Series(df[col].values, index=pd.DatetimeIndex(df["date"]),
                         name="portfolio_return").dropna().sort_index()

    def assert_exact(strategy, cand):
        r = score_run(strategy, "ok", None, load_oracle_series(strategy), cand, th)
        assert r.verdict is Verdict.RUNS_RIGHT and r.exact_tier, (
            f"{strategy}: {r.verdict} exact={r.exact_tier} corr={r.rung3.get('correlation')}")

    try:
        load_oracle_series("drf")
    except FileNotFoundError as exc:
        pytest.skip(f"oracle factors absent: {exc}")

    assert_exact("str", ser(run_characteristic_sort(panel, build_str.str_rulebook(signal_lag=0))["monthly_returns"]))
    cfg = build_mom6.load_config()
    assert_exact("mom6", ser(run_with_holding_period(panel, build_mom6.mom6_rulebook(cfg),
                                                     holding_period=int(cfg["holding_months"]))))
    assert_exact("drf", ser(run_bbw_factor(panel, "drf")["monthly_returns"]))
    assert_exact("lrf", ser(run_bbw_factor(panel, "lrf")["monthly_returns"]))
    components = {c: run_bbw_factor(panel, c)["monthly_returns"] for c in CRF_COMPONENTS}
    assert_exact("crf", ser(compose_crf(components), col="crf"))
