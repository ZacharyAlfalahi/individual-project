"""L2-gate ablation tests (scripts/run_l2_gate_ablation.py) — synthetic replays plus
a machine-local pin on the recorded artifact."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from evaluation.harness.reportability import Reportability                  # noqa: E402
from evaluation.harness.run_artefacts import RunArtefacts, RunField         # noqa: E402
from scripts.run_l2_gate_ablation import replay_at_level                    # noqa: E402
from scripts.run_relocator_rescore import assert_shipped_set_matches        # noqa: E402

# The page hyphenates "quintiles" across a line break: L1 preserves the hyphen
# ("quin- tiles" after whitespace collapse), L2 de-hyphenates and joins it.
_PAGE = ("We sort bonds into quin-\ntiles based on their downside risk and form "
         "value-weighted portfolios each month. Filler sentence follows here.")
_JOINED_QUOTE = ("We sort bonds into quintiles based on their downside risk and form "
                 "value-weighted portfolios each month.")


def _ct() -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "0"},
        normalisation={"ladder_level": "L1", "rules": []},
        pages=(_PAGE,), status="stub",
    )


def _raw(value: str, quote: str) -> dict:
    return {"answered": True, "value": value, "quote": quote, "kind": "part1_enum",
            "model_id": "m", "returned": "m"}


def _recorded(fields: dict[str, RunField], tmp_path: Path) -> RunArtefacts:
    return RunArtefacts(
        run_dir=tmp_path, paper_id="STUB", header={},
        fields=fields,
        reportability=Reportability(phase=None, reportable=False, reason="test",
                                    model_a_id="m", model_b_id="m", run_id=None),
    )


def _field(name: str, *, shipped: bool) -> RunField:
    return RunField(
        field=name, a_answered=True, b_answered=True, a_quote="q", b_quote="q",
        a_located=shipped, b_located=shipped, a_model_id="m", b_model_id="m",
        normalised_a="sorted_portfolios", normalised_b="sorted_portfolios",
        final_tag="STATED" if shipped else "UNKNOWN",
        shipped_reason="r", ship_choice=None, not_extracted=False,
    )


def test_l2_arm_recovers_hyphenation_only_quotes(tmp_path):
    recorded = _recorded({"formation_structure": _field("formation_structure",
                                                        shipped=False)}, tmp_path)
    raw = {"formation_structure": _raw("sorted_portfolios", _JOINED_QUOTE)}

    off = replay_at_level(recorded, raw, dict(raw), _ct(), None)
    assert not off.fields["formation_structure"].shipped
    assert_shipped_set_matches("stub", off, recorded, tmp_path)

    l2 = replay_at_level(recorded, raw, dict(raw), _ct(), "L2")
    assert l2.fields["formation_structure"].shipped
    assert l2.fields["formation_structure"].a_located


def test_recorded_arm_reproduces_a_located_field(tmp_path):
    # A quote verbatim at L1 ships identically in both arms (L2 is a superset).
    l1_quote = "Filler sentence follows here."
    recorded = _recorded({"asset_class": _field("asset_class", shipped=True)}, tmp_path)
    raw = {"asset_class": _raw("corporate_bonds", l1_quote)}
    off = replay_at_level(recorded, raw, dict(raw), _ct(), None)
    l2 = replay_at_level(recorded, raw, dict(raw), _ct(), "L2")
    assert off.fields["asset_class"].shipped and l2.fields["asset_class"].shipped
    assert_shipped_set_matches("stub", off, recorded, tmp_path)


_ARTIFACT = _REPO_ROOT / "results" / "l2_gate_ablation.json"
_HEADLINE = {"coverage": [47, 132], "selective_accuracy": [36, 47], "over_claim": [8, 47]}


@pytest.mark.skipif(not _ARTIFACT.exists(),
                    reason="recorded L2-ablation artifact not shipped with the repository")
def test_recorded_l2_ablation_artifact_pins():
    result = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert result["diagnostic"].startswith("L2-gate ablation")
    primary = result["primary"]
    assert primary["pooled_off"] == _HEADLINE
    n_recovered = sum(len(a["newly_shipped"]) for a in primary["per_anchor"].values())
    assert 0 <= n_recovered <= 24
