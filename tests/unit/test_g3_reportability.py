"""
G3 phase gate (evaluation contract §1, D33, D37).

This is the FIRST mechanical enforcement of the phase discipline anywhere in the
stack: until now `--phase report` differed from `--phase dev` by a YAML key, the
D33 authorization was a process step with no code behind it, and the only marker was a
hand-typed banner in a baseline document.

The tests below pin the four properties that make it a gate rather than a label:
it derives the phase from what ANSWERED (the header) not what was requested, it
compares the pair UNORDERED, it FAILS CLOSED on anything unpinned, and the
resulting stamp cannot be dropped by a caller that forgets it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from evaluation.harness.reportability import (  # noqa: E402
    ReportabilityError,
    classify_phase,
    load_g3_thresholds,
    load_model_stack,
    require_reportable,
)

_ROOT = Path(__file__).resolve().parents[2]
_BBW = _ROOT / "runs" / "g3_2026-07-22_v3" / "bbw"

_STACK = {
    "phase_d": {"model_a": {"model_id": "gemini-3.1-flash-lite"},
                "model_b": {"model_id": "mistral-small-latest"}},
    "phase_f": {"model_a": {"model_id": "claude-sonnet-4-6"},
                "model_b": {"model_id": "gemini-3.5-flash"}},
}

_FULL_PROVENANCE = {
    "paper_id": "BBW_2019", "registry_hash": "abc", "canonical_text_hash": "def",
    "silence_table_version": "v1.1", "prompt_template_hashes": "ghi",
    "timestamp": "2026-07-21T00:00:00+00:00", "run_id": "bbw-report-x",
}


def _header(a, b, **over):
    return {**_FULL_PROVENANCE, "model_a_id": a, "model_b_id": b, **over}


# --- classification ----------------------------------------------------------

def test_phase_d_pair_is_never_reportable():
    rep = classify_phase(_header("gemini-3.1-flash-lite", "mistral-small-latest"), _STACK)
    assert rep.phase == "phase_d"
    assert rep.reportable is False
    assert "no Phase-D number enters the project" in rep.reason
    assert rep.banner.startswith("*** NON-REPORTABLE (phase_d")


def test_phase_f_pair_with_full_provenance_is_reportable():
    rep = classify_phase(_header("claude-sonnet-4-6", "gemini-3.5-flash"), _STACK)
    assert rep.phase == "phase_f"
    assert rep.reportable is True
    assert rep.banner == ""


def test_pair_is_matched_unordered():
    """Which vendor is model_a is a configuration detail, not an identity."""
    rep = classify_phase(_header("gemini-3.5-flash", "claude-sonnet-4-6"), _STACK)
    assert rep.phase == "phase_f" and rep.reportable is True


def test_provider_snapshot_suffix_still_matches():
    """Contract §1 anticipates alias drift: a provider may return
    claude-sonnet-4-6-20260215 for a configured claude-sonnet-4-6."""
    rep = classify_phase(_header("claude-sonnet-4-6-20260215", "gemini-3.5-flash"), _STACK)
    assert rep.phase == "phase_f" and rep.reportable is True


def test_only_a_dated_snapshot_suffix_counts_as_the_same_sku():
    """The dangerous direction: `gemini-3.5-flash-lite` STARTS WITH
    `gemini-3.5-flash`, so a bare prefix rule would accept a different SKU -- with
    a different free-tier budget -- as a snapshot of the pinned one, which is
    exactly the drift the header check exists to catch. Only a version stamp
    (separator + digits) counts."""
    from evaluation.harness.reportability import _sku_matches

    assert _sku_matches("claude-sonnet-4-6", "claude-sonnet-4-6") is True
    assert _sku_matches("claude-sonnet-4-6-20260215", "claude-sonnet-4-6") is True
    assert _sku_matches("gemini-3.5-flash-lite", "gemini-3.5-flash") is False
    assert _sku_matches("gemini-3.5-flash-preview", "gemini-3.5-flash") is False
    assert _sku_matches("gemini-3.5-flash", "gemini-3.5-flash-lite") is False
    assert _sku_matches("", "gemini-3.5-flash") is False


def test_a_lite_variant_does_not_satisfy_a_pinned_full_sku():
    """End-to-end form of the same guard, through classify_phase."""
    rep = classify_phase(_header("gemini-3.5-flash-lite", "claude-sonnet-4-6"), _STACK)
    assert rep.phase is None and rep.reportable is False


def test_unrecognised_pair_fails_closed():
    rep = classify_phase(_header("gpt-4o", "llama-3"), _STACK)
    assert rep.phase is None
    assert rep.reportable is False
    assert "matches neither" in rep.reason


def test_phase_f_with_missing_provenance_is_not_reportable():
    """A run that cannot be reproduced is not reportable whatever pair made it."""
    h = _header("claude-sonnet-4-6", "gemini-3.5-flash")
    h["canonical_text_hash"] = ""
    rep = classify_phase(h, _STACK)
    assert rep.phase == "phase_f"
    assert rep.reportable is False
    assert "canonical_text_hash" in rep.reason


def test_git_dirty_is_reported_unverifiable_not_assumed_clean():
    """Contract §1 requires git_dirty = false for Phase F, but neither the spec
    nor the trace header records it. G3 surfaces the gap rather than assuming."""
    rep = classify_phase(_header("claude-sonnet-4-6", "gemini-3.5-flash"), _STACK)
    assert rep.git_dirty is None


# --- the gate ----------------------------------------------------------------

def test_require_reportable_refuses_by_default_and_opts_in_explicitly():
    dev = classify_phase(_header("gemini-3.1-flash-lite", "mistral-small-latest"), _STACK)
    with pytest.raises(ReportabilityError) as exc:
        require_reportable(dev)
    assert "NON-REPORTABLE" in str(exc.value)
    require_reportable(dev, allow_non_reportable=True)      # explicit opt-in is fine

    live = classify_phase(_header("claude-sonnet-4-6", "gemini-3.5-flash"), _STACK)
    require_reportable(live)                                 # no raise


def test_the_stamp_cannot_be_dropped():
    """Reportability is a mandatory field on RunArtefacts and AnchorScore -- no
    default, so neither can be constructed without deciding the phase."""
    import dataclasses

    from evaluation.harness.gold_calibration import AnchorScore
    from evaluation.harness.run_artefacts import RunArtefacts

    for cls in (RunArtefacts, AnchorScore):
        fld = next(f for f in dataclasses.fields(cls) if f.name == "reportability")
        assert fld.default is dataclasses.MISSING
        assert fld.default_factory is dataclasses.MISSING


# --- config ------------------------------------------------------------------

def test_g3_thresholds_load_from_the_committed_file():
    t = load_g3_thresholds()
    assert t.min_cell_n == 20                    # D34's uncalibrated bar
    assert 1.95 < t.wilson_z < 1.97
    assert t.interval_label == "descriptive"     # contract §3.3


def test_missing_g3_block_raises_rather_than_defaulting(tmp_path):
    """Mirrors load_verified_standing_subs on the G2 side: a calibration run
    against silently-defaulted thresholds is a number nobody chose."""
    p = tmp_path / "thresholds.yaml"
    p.write_text("librarian:\n  model_stack: {}\n", encoding="utf-8")
    with pytest.raises(ReportabilityError) as exc:
        load_g3_thresholds(p)
    assert "librarian.g3" in str(exc.value)


def test_the_committed_stack_still_carries_both_phases():
    stack = load_model_stack()
    assert "phase_d" in stack and "phase_f" in stack


# --- real artefacts ----------------------------------------------------------

@pytest.mark.skipif(not (_BBW / "trace_0.json").exists(),
                    reason="post-fix BBW dev run absent (runs/ is gitignored)")
def test_the_real_dev_run_classifies_as_phase_d_and_is_refused():
    from evaluation.harness.gold_calibration import score_anchor

    s = score_anchor("drf", _BBW)
    assert s.reportability.phase == "phase_d"
    assert s.reportability.reportable is False
    with pytest.raises(ReportabilityError):
        require_reportable(s.reportability)
