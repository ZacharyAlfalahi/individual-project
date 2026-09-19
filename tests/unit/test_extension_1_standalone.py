"""extension_1 — the pre-committed human baseline arm. What must hold regardless of its result:
it refuses to run while the registration is a draft, it carries true provenance rather than a
ladder rung, its threshold comes from the registration rather than from code, and it is never a
member of a proposal family."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agents.scientist.schemas.proposal import ProposalSource  # noqa: E402

import run_extension_1 as X  # noqa: E402


def test_a_draft_registration_is_refused(tmp_path):
    """A threshold that is not pre-registered is not a pre-commitment."""
    draft = tmp_path / "extension_1_config.yaml"
    draft.write_text("# STATUS: DRAFT — not final\nmedian_split: {}\n", encoding="utf-8")
    ok, status = X.pre_registered(draft)
    assert ok is False and "DRAFT" in status


def test_a_pre_registered_config_is_accepted(tmp_path):
    registered = tmp_path / "extension_1_config.yaml"
    registered.write_text("# STATUS: PRE-REGISTERED\n", encoding="utf-8")
    ok, status = X.pre_registered(registered)
    assert ok is True and "PRE-REGISTERED" in status


def test_a_registration_without_a_status_line_is_refused(tmp_path):
    silent = tmp_path / "extension_1_config.yaml"
    silent.write_text("median_split: {}\n", encoding="utf-8")
    ok, _ = X.pre_registered(silent)
    assert ok is False


def test_the_shipped_registration_is_pre_registered_and_carries_the_frozen_median():
    ok, _ = X.pre_registered()
    assert ok is True
    cfg = X.load_config()
    assert cfg["median_split"]["median_baa_aaa_spread_pp"] == 0.935
    # the alternative window stays a recorded sensitivity, not a second threshold
    assert (cfg["median_split"]["sensitivity_primary_inference_window"]
            ["median_baa_aaa_spread_pp"]) == 0.920


def test_the_proposal_takes_its_threshold_from_the_registration_not_from_code():
    class _Case:
        case_id, strategy_id = "rq4_str", "str"

    cfg = {"median_split": {"median_baa_aaa_spread_pp": 1.234}}
    proposal = X.build_extension_1(_Case(), cfg)
    assert "1.234" in proposal.rationale


def test_the_arm_carries_true_provenance_not_a_ladder_rung():
    class _Case:
        case_id, strategy_id = "rq4_str", "str"

    proposal = X.build_extension_1(_Case(), X.load_config())
    assert proposal.generation.source is ProposalSource.PRE_COMMITTED_HUMAN
    assert proposal.generation.source.value not in {
        "llm_researcher", "retrieval_only", "random_eligible"}
    # the registration date, not the run date — it was pre-committed
    assert proposal.generation.generated_at.startswith("2026-08-07")


def test_the_conditioning_is_lagged_and_binary_as_registered():
    class _Case:
        case_id, strategy_id = "rq4_str", "str"

    delta = X.build_extension_1(_Case(), X.load_config()).config_delta
    assert delta.conditioning_variable == "baa_aaa_spread"
    assert delta.conditioning_lag_months == 1          # never used at lag 0
    assert delta.interaction_form == "binary_above_historical_median"
