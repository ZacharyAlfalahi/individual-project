"""
G3 §3.3 aggregation altitudes (the G = 3 rule).

The risks this file guards are all about a summary claiming more than it has:
a mean of two numbers presented as an estimate, an aggregate that quietly covers
2 of 3 anchors, a Phase-D paper pooled into a reportable total, or a two-paper
leave-one-out presented as a robustness analysis.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.harness.aggregation import (  # noqa: E402
    ANCHOR_SET,
    aggregate,
    render_aggregate,
)
from evaluation.harness.calibration_report import compute_metrics  # noqa: E402
from evaluation.harness.gold_calibration import score_anchor  # noqa: E402
from evaluation.harness.reportability import Reportability, ReportabilityError  # noqa: E402
from evaluation.harness.run_artefacts import load_run  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
_RUNS = {"drf": _ROOT / "runs" / "g3_2026-07-22_v3" / "bbw",
         "mom6": _ROOT / "runs" / "g3_2026-07-22_v3" / "jnps",
         "str": _ROOT / "runs" / "g3_2026-07-22_v3" / "drr"}

_needs_runs = pytest.mark.skipif(
    not all((p / "trace_0.json").exists() for p in _RUNS.values()),
    reason="post-fix dev runs absent (runs/ is gitignored)",
)


def _bundles():
    out = {}
    for anchor, run in _RUNS.items():
        art = load_run(run)
        out[anchor] = compute_metrics(score_anchor(anchor, run, artefacts=art), art)
    return out


# --- the altitudes -----------------------------------------------------------

@_needs_runs
def test_micro_pools_raw_counts_not_rates():
    """Micro is a pooled count, so its denominator is the sum of the papers'."""
    b = _bundles()
    agg = aggregate(b)
    expected_n = sum(x.coverage.denominator for x in b.values())
    expected_k = sum(x.coverage.numerator for x in b.values())
    assert (agg.micro["coverage"].numerator, agg.micro["coverage"].denominator) \
        == (expected_k, expected_n)


@_needs_runs
def test_macro_is_an_unweighted_mean_that_shows_its_constituents():
    """Contract §5.3 house style: never the summary alone. On G<=3 the mean is a
    summary of too few points to stand by itself."""
    b = _bundles()
    agg = aggregate(b)
    m = agg.macro["selective_accuracy"]
    assert m.n_papers == 3
    assert dict(m.values) == {"drf": pytest.approx(3 / 4), "mom6": pytest.approx(6 / 7),
                              "str": pytest.approx(6 / 8)}
    assert m.mean == pytest.approx((3 / 4 + 6 / 7 + 6 / 8) / 3)
    assert "constituents" in m.render()


@_needs_runs
def test_macro_and_micro_genuinely_differ():
    """They answer different questions and the anchors differ enormously. If they
    ever coincide across the board, one of them is not being computed."""
    agg = aggregate(_bundles())
    assert agg.macro["selective_accuracy"].mean != pytest.approx(
        agg.micro["selective_accuracy"].value)


@_needs_runs
def test_no_dispersion_statistic_is_emitted_on_so_few_papers():
    """A standard deviation over 2-3 papers estimates nothing. MacroStat carries
    the constituents instead, and must not grow a std field by accident."""
    import dataclasses

    agg = aggregate(_bundles())
    names = {f.name for f in dataclasses.fields(type(agg.macro["coverage"]))}
    assert not (names & {"std", "stdev", "sd", "variance", "sem"})


@_needs_runs
def test_leave_one_out_drops_exactly_one_paper():
    agg = aggregate(_bundles())
    assert set(agg.leave_one_out) == {"drf", "mom6", "str"}
    # at G=3, dropping one leaves a genuine POOL of two, not a single paper
    rest = agg.leave_one_out["drf"]["selective_accuracy"]
    assert (rest.numerator, rest.denominator) == (6 + 6, 7 + 8)   # mom6 + str pooled


@_needs_runs
def test_leave_one_out_is_not_declared_degenerate_at_three_papers():
    """At G=3 dropping one leaves a real pool, so the degeneracy caveat must NOT
    fire -- carrying it would understate what the analysis supports."""
    out = render_aggregate(aggregate(_bundles()), allow_non_reportable=True)
    assert "degenerates to the per-paper row" not in out


@_needs_runs
def test_leave_one_out_is_still_declared_degenerate_below_three_papers():
    """The caveat must fire when it applies. Synthetic G=2 subset."""
    b = _bundles()
    two = {k: v for k, v in list(b.items())[:2]}
    out = render_aggregate(aggregate(two), allow_non_reportable=True)
    assert "degenerates to the per-paper row" in out
    assert "NOT as a robustness claim" in out


@_needs_runs
def test_the_third_anchor_stabilised_selective_accuracy():
    """At G=2 selective accuracy spanned 21 points across drops -- which read as
    pipeline instability. At G=3 it spans under 10, so the G=2 swing was an
    artefact of having two papers, not a property of the pipeline. This is the
    single clearest argument for why the third anchor was worth recovering."""
    agg = aggregate(_bundles())
    vals = [agg.leave_one_out[a]["selective_accuracy"].value for a in agg.anchors_scored]
    assert max(vals) - min(vals) < 0.10


# --- completeness ------------------------------------------------------------

@_needs_runs
def test_the_full_anchor_set_reports_complete():
    """All three anchors now have artefacts -- the first aggregate that genuinely
    covers the gold corpus."""
    agg = aggregate(_bundles())
    assert agg.anchors_expected == ANCHOR_SET
    assert agg.complete is True and agg.missing == ()
    out = render_aggregate(agg, allow_non_reportable=True)
    assert "3 of 3 anchors" in out
    assert "does NOT cover the gold corpus" not in out


@_needs_runs
def test_incomplete_coverage_is_declared_not_implied():
    """The guard still has to fire when an anchor IS missing -- an aggregate over
    a subset must never read as if it covered the corpus."""
    b = _bundles()
    subset = {k: v for k, v in b.items() if k != "str"}
    agg = aggregate(subset)
    assert agg.complete is False and agg.missing == ("str",)
    out = render_aggregate(agg, allow_non_reportable=True)
    assert "2 of 3 anchors" in out
    assert "does NOT cover the gold corpus" in out


# --- reportability -----------------------------------------------------------

@_needs_runs
def test_aggregate_reportability_is_the_and_over_constituents():
    """One Phase-D paper contaminates the aggregate. §1 governs which numbers may
    be reported, not how many are averaged together."""
    agg = aggregate(_bundles())
    assert agg.reportability.reportable is False
    assert "non-reportable anchor" in agg.reportability.reason
    with pytest.raises(ReportabilityError):
        render_aggregate(agg)


@_needs_runs
def test_all_reportable_constituents_give_a_reportable_aggregate():
    b = _bundles()
    live = Reportability(phase="phase_f", reportable=True, reason="ok",
                         model_a_id="claude-sonnet-4-6", model_b_id="gemini-3.5-flash")
    b = {k: replace(v, reportability=live) for k, v in b.items()}
    agg = aggregate(b)
    assert agg.reportability.reportable is True
    render_aggregate(agg)                    # no raise


@_needs_runs
def test_mixed_phases_never_pool():
    """Calibration is pair-specific (§1); pooling across phases would produce a
    number that describes no pair."""
    b = _bundles()
    keys = list(b)
    live = Reportability(phase="phase_f", reportable=True, reason="ok",
                         model_a_id="claude-sonnet-4-6", model_b_id="gemini-3.5-flash")
    dev = Reportability(phase="phase_d", reportable=True, reason="ok (synthetic)",
                        model_a_id="gemini-3.1-flash-lite", model_b_id="mistral-small-latest")
    # ALL constituents reportable, so the non-reportable guard cannot fire first --
    # this isolates the mixed-phase rule specifically.
    b = {k: replace(v, reportability=live) for k, v in b.items()}
    b[keys[-1]] = replace(b[keys[-1]], reportability=dev)
    agg = aggregate(b)
    assert agg.reportability.reportable is False
    assert "different phases" in agg.reportability.reason


def test_empty_aggregate_is_not_reportable_and_does_not_divide_by_zero():
    agg = aggregate({})
    assert agg.reportability.reportable is False
    assert agg.micro == {} and agg.leave_one_out == {}
    assert agg.macro == {} or all(m.mean is None for m in agg.macro.values())


# --- the §3.3 disclaimer -----------------------------------------------------

@_needs_runs
def test_no_population_level_claim_is_rendered():
    out = render_aggregate(aggregate(_bundles()), allow_non_reportable=True)
    assert "no population-level claim is made" in out
    assert "observed gold corpus" in out
