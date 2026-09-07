"""Re-scoring tests (scripts/run_relocator_rescore.py) — synthetic
recorded runs through the replay + cross-pin machinery, plus a machine-local
pin on the committed artifact once the real pass has run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from evaluation.gold_specs.gold_loader import load_gold_spec                # noqa: E402
from evaluation.harness.relocate_thresholds import RelocatorConfig          # noqa: E402
from evaluation.harness.reportability import Reportability                  # noqa: E402
from evaluation.harness.run_artefacts import (                              # noqa: E402
    RunArtefacts,
    RunField,
    canonical_trace_json,
)
from scripts import run_relocator_rescore as rescore_mod                    # noqa: E402
from scripts.run_relocator_rescore import (                                 # noqa: E402
    assert_newly_shipped_within_qgf,
    assert_shipped_set_matches,
    canonical_text_for,
    replay_relocated,
    resolve_strategy,
)

CFG = RelocatorConfig(metric="sequencematcher_ratio_l1", accept_bar=0.8,
                      min_quote_chars=30, min_anchor_chars=10)

_S1 = ("We sort bonds into quintiles based on their downside risk and form "
       "value-weighted portfolios each month.")
_S2 = ("The distress factor earns a significant premium after controlling for "
       "duration and rating in every specification.")


def _ct() -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "0"},
        normalisation={"ladder_level": "L1", "rules": []},
        pages=("Intro filler text. " + _S1, _S2 + " Closing filler text."),
        status="stub",
    )


def _raw(value: str, quote: str | None, *, answered: bool = True) -> dict:
    return {"answered": answered, "value": value if answered else None,
            "quote": quote if answered else None, "kind": "part1_enum",
            "model_id": "m", "returned": "m"}


def _recorded_field(name: str, *, shipped: bool) -> RunField:
    return RunField(
        field=name, a_answered=True, b_answered=True,
        a_quote="q", b_quote="q", a_located=shipped, b_located=shipped,
        a_model_id="m", b_model_id="m",
        normalised_a="sorted_portfolios", normalised_b="sorted_portfolios",
        final_tag="STATED" if shipped else "UNKNOWN",
        shipped_reason="r", ship_choice=None, not_extracted=False,
    )


def _recorded(fields: dict[str, RunField], tmp_path: Path) -> RunArtefacts:
    return RunArtefacts(
        run_dir=tmp_path, paper_id="STUB", header={},
        fields=fields,
        reportability=Reportability(phase=None, reportable=False, reason="test",
                                    model_a_id="m", model_b_id="m", run_id=None),
    )


def test_cross_pin_reproduces_synthetic_recorded_run(tmp_path):
    recorded = _recorded({
        "formation_structure": _recorded_field("formation_structure", shipped=True),
        "asset_class": _recorded_field("asset_class", shipped=False),
    }, tmp_path)
    raw_a = {"formation_structure": _raw("sorted_portfolios", _S1),
             "asset_class": _raw("corporate_bonds", "this quote is nowhere in the text")}
    raw_b = {"formation_structure": _raw("sorted_portfolios", _S1),
             "asset_class": _raw("corporate_bonds", "another quote also absent entirely")}
    variant_off, log = replay_relocated(recorded, raw_a, raw_b, _ct(), None)
    assert log == []
    assert variant_off.fields["formation_structure"].shipped
    assert not variant_off.fields["asset_class"].shipped
    assert_shipped_set_matches("stub", variant_off, recorded, tmp_path)


def test_cross_pin_raises_on_drift(tmp_path):
    # The recorded run claims asset_class shipped; the exact replay disagrees.
    recorded = _recorded({
        "asset_class": _recorded_field("asset_class", shipped=True),
    }, tmp_path)
    raw = {"asset_class": _raw("corporate_bonds", "this quote is nowhere in the text")}
    variant_off, _ = replay_relocated(recorded, raw, dict(raw), _ct(), None)
    with pytest.raises(RuntimeError, match="do not report"):
        assert_shipped_set_matches("stub", variant_off, recorded, tmp_path)


def test_relocation_ships_only_agreeing_fields(tmp_path):
    # Both quotes relocate, but the values DISAGREE -> the dual gate holds.
    recorded = _recorded({
        "formation_structure": _recorded_field("formation_structure", shipped=False),
    }, tmp_path)
    drifted = _S1.replace("quintiles", "deciles")
    raw_a = {"formation_structure": _raw("sorted_portfolios", drifted)}
    raw_b = {"formation_structure": _raw("characteristic_regression", drifted)}
    variant_on, log = replay_relocated(recorded, raw_a, raw_b, _ct(), CFG)
    assert not variant_on.fields["formation_structure"].shipped
    assert log and log[0]["a"]["method"] == "relocated"


def test_both_quotes_must_pass_exact_or_relocate(tmp_path):
    recorded = _recorded({
        "formation_structure": _recorded_field("formation_structure", shipped=False),
    }, tmp_path)
    drifted = _S1.replace("quintiles", "deciles")

    raw_a = {"formation_structure": _raw("sorted_portfolios", _S1)}          # exact
    raw_b = {"formation_structure": _raw("sorted_portfolios", drifted)}      # relocates
    variant_on, log = replay_relocated(recorded, raw_a, raw_b, _ct(), CFG)
    assert variant_on.fields["formation_structure"].shipped
    assert log[0]["a"]["method"] == "exact" and log[0]["b"]["method"] == "relocated"
    assert log[0]["shipped"]

    garbage = "completely unrelated words about equity index options and dealers"
    raw_b_bad = {"formation_structure": _raw("sorted_portfolios", garbage)}
    variant_bad, log_bad = replay_relocated(recorded, raw_a, raw_b_bad, _ct(), CFG)
    assert not variant_bad.fields["formation_structure"].shipped
    assert log_bad[0]["b"]["method"] == "rejected"


def test_newly_shipped_subset_invariant(tmp_path):
    def field(name, *, located, norm_b="v", shipped=False):
        return RunField(
            field=name, a_answered=True, b_answered=True, a_quote="q", b_quote="q",
            a_located=located, b_located=True, a_model_id="m", b_model_id="m",
            normalised_a="v", normalised_b=norm_b,
            final_tag="STATED" if shipped else "UNKNOWN",
            shipped_reason="r", ship_choice=None, not_extracted=False)

    # Only "qgf" is agree-but-gate-failed; "other" DISAGREES, so it is not.
    off = _recorded({
        "qgf": field("qgf", located=False),
        "other": field("other", located=False, norm_b="w"),
    }, tmp_path)
    on_ok = _recorded({
        "qgf": _recorded_field("qgf", shipped=True),
        "other": _recorded_field("other", shipped=False),
    }, tmp_path)
    newly, qgf = assert_newly_shipped_within_qgf("stub", on_ok, off)
    assert newly == ["qgf"] and qgf == {"qgf"}

    on_bad = _recorded({
        "qgf": _recorded_field("qgf", shipped=False),
        "other": _recorded_field("other", shipped=True),
    }, tmp_path)
    with pytest.raises(RuntimeError, match="do not report"):
        assert_newly_shipped_within_qgf("stub", on_bad, off)


def _write_strategy(run_dir: Path, index: int, label: str) -> None:
    trace = {"header": {"paper_id": "STUB", "strategy_label": label}, "records": []}
    trace_json = canonical_trace_json(trace)
    (run_dir / f"trace_{index}.json").write_text(trace_json, encoding="utf-8")
    import hashlib
    spec = {"header": {"trace_sha256": hashlib.sha256(trace_json.encode()).hexdigest()}}
    (run_dir / f"spec_{index}.json").write_text(json.dumps(spec), encoding="utf-8")


def test_resolve_strategy_single_multi_and_no_match(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    _write_strategy(single, 0, "anything")
    art, idx = resolve_strategy("str", single)
    assert idx == 0 and art.paper_id == "STUB"

    raw_label = load_gold_spec("str").header.strategy_label
    gold_label = getattr(raw_label, "value", raw_label)
    multi = tmp_path / "multi"
    multi.mkdir()
    _write_strategy(multi, 0, "decoy strategy")
    _write_strategy(multi, 1, gold_label)
    art, idx = resolve_strategy("str", multi)
    assert idx == 1 and art.header["strategy_label"] == gold_label

    # The real bbw archive stamps a LONGER label than the gold value
    # ("Downside Risk Factor (DRF)" vs "DRF") — unique containment must match.
    contain = tmp_path / "contain"
    contain.mkdir()
    _write_strategy(contain, 0, "decoy strategy")
    _write_strategy(contain, 1, f"Long form of the {gold_label} strategy")
    art, idx = resolve_strategy("str", contain)
    assert idx == 1

    nomatch = tmp_path / "nomatch"
    nomatch.mkdir()
    _write_strategy(nomatch, 0, "decoy one")
    _write_strategy(nomatch, 1, "decoy two")
    with pytest.raises(RuntimeError, match="none matches gold"):
        resolve_strategy("str", nomatch)


def test_canonical_text_for_guards(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(rescore_mod, "_REPO_ROOT", tmp_path)
    ct_rel = "evaluation/canonical_texts/stub.frozen.yaml"
    ct_path = tmp_path / ct_rel
    ct_path.parent.mkdir(parents=True)
    ct_path.write_text(
        "source_pdf: stub.pdf\nsource_sha256: x\n"
        "parser: {name: stub, version: '0'}\n"
        "normalisation: {ladder_level: L1, rules: []}\n"
        "pages: ['page one text here']\nstatus: stub\n", encoding="utf-8")
    good_sha = hashlib.sha256(ct_path.read_bytes()).hexdigest()

    def manifest(run_name: str, inputs: dict) -> Path:
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(
            json.dumps({"inputs": inputs}), encoding="utf-8")
        return run_dir

    ct = canonical_text_for(manifest("good", {ct_rel: good_sha, "other/input.yaml": "z"}))
    assert ct.pages == ("page one text here",)

    with pytest.raises(RuntimeError, match="exactly one canonical-text"):
        canonical_text_for(manifest("none", {"other/input.yaml": "z"}))
    with pytest.raises(RuntimeError, match="exactly one canonical-text"):
        canonical_text_for(manifest("two", {ct_rel: good_sha,
                                            "evaluation/canonical_texts/b.yaml": "z"}))
    with pytest.raises(RuntimeError, match="drifted"):
        canonical_text_for(manifest("drift", {ct_rel: "0" * 64}))


_ARTIFACT = _REPO_ROOT / "results" / "relocator_rescore.json"
_HEADLINE = {"coverage": [47, 132], "selective_accuracy": [36, 47], "over_claim": [8, 47]}


@pytest.mark.skipif(not _ARTIFACT.exists(),
                    reason="committed qr1 rescore artifact absent on this machine "
                           "(the strict-universe calibration admits no bar, so it never exists)")
def test_committed_rescore_artifact_pins():
    result = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert result["diagnostic"].startswith("Relocate-then-certify")
    primary = result["primary"]
    # The relocation-OFF arm IS the registered headline, byte-for-byte.
    assert primary["pooled_off"] == _HEADLINE
    n_recovered = sum(len(a["newly_shipped"]) for a in primary["per_anchor"].values())
    n_qgf = sum(a["n_agree_quote_gate_failed"] for a in primary["per_anchor"].values())
    assert n_qgf == 24
    assert 0 <= n_recovered <= 24


_QR2_ARTIFACT = _REPO_ROOT / "results" / "relocator_rescore_qr2.json"


@pytest.mark.skipif(not _QR2_ARTIFACT.exists(),
                    reason="committed qr2 rescore artifact absent on this machine")
def test_committed_qr2_rescore_artifact_pins():
    result = json.loads(_QR2_ARTIFACT.read_text(encoding="utf-8"))
    assert result["config"]["protocol"] == "qr2"
    assert result["config"]["accept_bar"] == 0.90
    assert result["config"]["calibration_status"] == "recall_floor_unmet"
    primary = result["primary"]
    assert primary["pooled_off"] == _HEADLINE
    assert primary["pooled_on"]["coverage"] == [65, 132]
    n_recovered = sum(len(a["newly_shipped"]) for a in primary["per_anchor"].values())
    assert n_recovered == 18
