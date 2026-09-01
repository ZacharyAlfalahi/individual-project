"""WS-C (P2) — report + end-to-end fixture path: router -> selector -> metrics ->
taxonomy -> report, with ZERO live model calls. Asserts the §6 caveat verbatim,
no Sharpe-as-performance, below-floor suppression, and the driver's cache-first
dry-run makes zero generate() calls."""

from __future__ import annotations

import pandas as pd
import yaml

from evaluation.codegen.census import (
    CensusInput,
    RoutingDecision,
    run_census,
)
from evaluation.codegen.p2_driver import build_prompt_from_spec, run_p2_driver
from evaluation.codegen.p2_metrics import compute_p2_metrics
from evaluation.codegen.p2_report import (
    AGREEMENT_CAVEAT,
    _blockquote,
    render_report,
)
from evaluation.codegen.p2_selector import select_arms
from evaluation.codegen.p2_taxonomy import (
    eligibility_exclusion_accounting,
    stratified_failure_sample,
)

_TH = {
    "arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5},
    "agreement": {"correlation_min": 0.99, "sign_agreement_min": 0.95},
    "divergence_strata": {"high_divergence_corr_lt": 0.90, "medium_divergence_corr_lt": 0.99},
    "taxonomy_sampling": {"strata": ["arm", "divergence_magnitude"]},
    "zoo_list": {"frozen_sha256": "TO_SET"},
    "min_overlap_months": 3,
}


class CountingStub:
    """Counts calls — the zero-generation proof on the fixture path."""

    def __init__(self, response="```python\nprint('x')\n```"):
        self.name = "stub"
        self.response = response
        self.calls = 0

    def generate(self, prompt, *, seed):
        self.calls += 1
        return self.response


def _spec(pid):
    # carries claimed_headline_metric so redaction is provable
    return {
        "header": {"paper_id": pid, "strategy_label": "synthetic sort"},
        "part1": {"formation_structure": "quintile sort on a characteristic"},
        "claimed_headline_metric": {"mean": 0.012, "t_stat": 3.4, "unit": "monthly"},
    }


def _census_with(n_refusals, n_compilable):
    inputs = (
        [CensusInput(f"r{i}", _spec(f"r{i}"), True) for i in range(n_refusals)]
        + [CensusInput(f"c{i}", _spec(f"c{i}"), True) for i in range(n_compilable)]
    )

    def route(inp):
        return (RoutingDecision(False, True, "refuse_no_strategy")
                if inp.paper_id.startswith("r") else RoutingDecision(True, False))

    return run_census(inputs, route)


def _series(values, start="2010-01-31"):
    idx = pd.date_range(start, periods=len(values), freq="ME")
    return pd.Series(values, index=idx, name="portfolio_return")


# --- end-to-end render (above floor) --------------------------------------------

def test_end_to_end_render_carries_caveat_and_no_sharpe():
    census = _census_with(5, 5)
    zoo = [f"r{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
    sel = select_arms(census, zoo, _TH)

    base = _series([0.01, -0.02, 0.03, 0.02, 0.015])
    pairs = {pid: (base, base.copy()) for pid in sel.members()}
    compiler = {pid: base + 0.001 for pid in sel.arm_b}
    metrics = compute_p2_metrics(census, sel, pairs, compiler, _TH)
    taxonomy = stratified_failure_sample(metrics.agreements, per_stratum=2)
    eligibility = eligibility_exclusion_accounting(census)

    report = render_report(metrics, sel, census, taxonomy=taxonomy, eligibility=eligibility)

    # (1) the §6 caveat appears VERBATIM (as the rendered blockquote)
    assert _blockquote(AGREEMENT_CAVEAT) in report
    assert "Inter-model agreement is not correctness." in report
    assert "license no" in report and "validity claim." in report
    # (2) never a Sharpe-as-performance number
    assert "sharpe" not in report.lower()
    # (3) the primary estimand + census fate table are rendered
    assert "correlation distribution (PRIMARY)" in report
    assert "Census fate table" in report
    assert "neither side is truth" in report
    assert "MDE by arm size" in report


def test_below_floor_report_suppresses_numbers_but_keeps_caveat():
    census = _census_with(3, 6)
    zoo = [f"r{i}" for i in range(3)] + [f"c{i}" for i in range(6)]
    sel = select_arms(census, zoo, _TH)
    metrics = compute_p2_metrics(census, sel, {}, None, _TH)
    assert metrics.suppressed is True

    taxonomy = stratified_failure_sample(metrics.agreements, per_stratum=2)  # empty
    report = render_report(metrics, sel, census, taxonomy=taxonomy)

    assert _blockquote(AGREEMENT_CAVEAT) in report          # caveat still carried
    assert "SUPPRESSED" in report
    assert "correlation distribution (PRIMARY)" not in report   # no agreement numbers
    assert "sharpe" not in report.lower()
    assert "Census fate table" in report                    # fate table still emitted


def test_empty_report_renders_without_a_run():
    report = render_report(None, None, None)
    assert _blockquote(AGREEMENT_CAVEAT) in report
    assert "sharpe" not in report.lower()


# --- driver fixture path: ZERO live generation ----------------------------------

def test_driver_dry_run_makes_zero_generate_calls():
    census = _census_with(5, 5)
    zoo = [f"r{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
    sel = select_arms(census, zoo, _TH)
    stub = CountingStub()

    out = run_p2_driver(census, sel, dry_run=True, client_factory=lambda m: stub)

    assert stub.calls == 0                                   # the zero-generation proof
    assert out["emitted_numbers"] is False
    assert out["gated_emit"] is False
    # a prompt hash for every selected member (Arm A ∪ Arm B)
    assert set(out["prompt_sha256"]) == set(sel.members())


def test_driver_not_gated_when_status_not_frozen(tmp_path):
    # even with census_available=True and dry_run=False, a draft status keeps it
    # fixture-only. Injected via thresholds_path: the live corpus.selection.status
    # froze on 2026-09-01 (T2-SEL-4), so the draft state must be a fixture.
    draft = tmp_path / "thresholds_draft.yaml"
    draft.write_text(
        yaml.safe_dump({
            "librarian": {"model_stack": {"phase_f": {
                "model_a": {"model_id": "stub-a", "vendor": "stub", "api_key_env": "X"},
                "model_b": {"model_id": "stub-b", "vendor": "stub", "api_key_env": "Y"},
            }}},
            "corpus": {"selection": {"status": "draft_pending_review"}},
        }),
        encoding="utf-8",
    )
    census = _census_with(5, 5)
    zoo = [f"r{i}" for i in range(5)] + [f"c{i}" for i in range(5)]
    sel = select_arms(census, zoo, _TH)
    stub = CountingStub()

    out = run_p2_driver(census, sel, dry_run=False, census_available=True,
                        client_factory=lambda m: stub, thresholds_path=draft)

    assert out["corpus_selection_status"] != "frozen"
    assert stub.calls == 0 and out["emitted_numbers"] is False


def test_build_prompt_from_spec_redacts_headline_metric():
    prompt = build_prompt_from_spec(_spec("p"))
    assert "claimed_headline_metric" not in prompt
    assert "quintile sort on a characteristic" in prompt   # construction survives
    # a StrategySpec-like object with .to_dict() also works
    class _Obj:
        def to_dict(self):
            return _spec("q")
    assert "claimed_headline_metric" not in build_prompt_from_spec(_Obj())
