"""C-lever (2026-09-02): targeted-field extraction + the T5 Arm-A runner.

Pins: the --fields allowlist (target asked, others not_extracted, structural
always, default None byte-identical -- the existing driver suite covers default),
fail-loud validation, the T5 variant overrides + manifest provenance, and the
runner's sheet handling incl. grader-compatible sheet ids."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts import run_librarian                                        # noqa: E402
from scripts import run_t5_extraction as t5x                             # noqa: E402
from scripts.run_t5_adversarial import sheet_id as grader_sheet_id       # noqa: E402


# ---------------------------------------------------------------------------
# The --fields allowlist through the real fake-phase driver.
# ---------------------------------------------------------------------------

def _spec(out: Path) -> dict:
    return json.loads((out / "spec_0.json").read_text(encoding="utf-8"))


def test_targeted_field_asked_others_not_extracted(tmp_path):
    out = tmp_path / "targeted"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake",
                             "--fields", "weighting_scheme", "--out", str(out)])
    assert rc == 0
    spec = _spec(out)
    p2 = spec["part2"]
    # The target was asked (the fake pair scripts weighting_scheme -> STATED).
    assert p2["weighting_scheme"]["tag"] == "STATED"
    # An untargeted capped field carries the registered not_extracted degrade.
    hp = p2["holding_period"]
    assert hp["tag"] == "UNKNOWN"
    assert hp["evidence"]["unknown_reason"] == "not_extracted"
    # Structural fields still ran (Part 1 + combiner present in the spec).
    assert spec["part1"]["asset_class"]["tag"] in ("STATED", "UNKNOWN")
    assert "combiner" in p2


def test_empty_allowlist_is_structural_only(tmp_path):
    out = tmp_path / "structural"
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake",
                             "--fields", "", "--out", str(out)])
    assert rc == 0
    p2 = _spec(out)["part2"]
    assert p2["weighting_scheme"]["evidence"]["unknown_reason"] == "not_extracted"
    assert p2["holding_period"]["evidence"]["unknown_reason"] == "not_extracted"


def test_unknown_field_name_fails_loud(tmp_path):
    with pytest.raises(SystemExit):
        run_librarian.main(["--paper", "bbw", "--phase", "fake",
                            "--fields", "weighting_schem", "--out", str(tmp_path / "x")])


def test_canonical_text_override_recorded_in_manifest(tmp_path):
    """The T5 variant mechanism: an override path drives the run AND lands hashed
    in the manifest inputs (attribution of exactly which text was extracted)."""
    out = tmp_path / "override"
    override = "evaluation/canonical_texts/bbw_2019.frozen.yaml"   # self-override: stays green
    rc = run_librarian.main(["--paper", "bbw", "--phase", "fake",
                             "--canonical-text", override, "--out", str(out)])
    assert rc == 0
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    assert override in manifest["inputs"]
    assert manifest["inputs"][override] is not None                # hashed, file exists


# ---------------------------------------------------------------------------
# The T5 runner.
# ---------------------------------------------------------------------------

def test_arm_a_sheets_exclude_rejects_and_carry_grader_ids():
    sheets = t5x.arm_a_sheets()
    assert sheets, "Arm-A sheets should exist in the repo"
    assert all(not s["path"].name.startswith("reject_") for s in sheets)
    # sheet_id must match the GRADER's derivation exactly (grading-compat spine).
    import yaml
    probe = sheets[0]
    raw = yaml.safe_load(probe["path"].read_text(encoding="utf-8"))
    assert probe["sheet_id"] == grader_sheet_id(raw)
    assert all(s["anchor"] in t5x.PAPER_KEY_OF_ANCHOR for s in sheets)


def test_fields_arg_for_extractable_compound_and_failloud():
    assert t5x.fields_arg_for("weighting_scheme") == "weighting_scheme"
    assert t5x.fields_arg_for("control_axis") == "control_axis"
    # Compound target: EVERY component asked (the drf long_leg+weighting_scheme
    # paraphrase sheet -- silently dropping to structural-only graded NOT_ASKED ->
    # spurious MISS; MAJOR review finding 2026-09-02).
    assert t5x.fields_arg_for("long_leg+weighting_scheme") == "long_leg,weighting_scheme"
    # An unextractable component fails LOUD, never silent structural-only.
    with pytest.raises(ValueError, match="unextractable"):
        t5x.fields_arg_for("strategy_label")
    with pytest.raises(ValueError, match="unextractable"):
        t5x.fields_arg_for("long_leg+strategy_label")


def test_every_scoreable_sheet_target_is_fully_extractable():
    """Census pin: fields_arg_for must succeed for every scoreable Arm-A sheet.
    A new sheet whose target the runner cannot ask should fail HERE, at build
    time, not as a paid-run CRASH row."""
    for s in t5x.arm_a_sheets():
        if s["scoreable"]:
            assert t5x.fields_arg_for(s["field"])                  # no raise, non-empty


@pytest.mark.skipif(not t5x.FROZEN_DIR.exists(),
                    reason="machine-local frozen variants (gitignored) not present")
def test_t5_runner_e2e_one_variant_fake_phase(tmp_path):
    """End-to-end on one real perturbed variant at fake phase: the runner produces
    a run dir the grader's conventions can consume, plus the run map + log."""
    run_map_path = tmp_path / "map.json"
    rc = t5x.main(["--limit-variants", "1", "--anchor", "drf", "--phase", "fake",
                   "--out-root", str(tmp_path / "t5"), "--run-map", str(run_map_path)])
    assert rc == 0
    run_map = json.loads(run_map_path.read_text(encoding="utf-8"))
    assert len(run_map) == 1
    sid, run_dir = next(iter(run_map.items()))
    assert (Path(run_dir).name == sid)                             # grader runs_dir convention
    assert (Path(run_dir) / "spec_0.json").exists()
    log = json.loads((tmp_path / "t5" / "t5_extraction_log.json").read_text(encoding="utf-8"))
    assert log["variants"][sid]["outcome"] in ("ok", "review")


def test_run_variants_skips_and_continues(monkeypatch, tmp_path):
    frozen_ok = tmp_path / "v1.frozen.yaml"
    frozen_ok.write_text("status: frozen\n", encoding="utf-8")
    sheets = [
        {"sheet_id": "drf__weighting_scheme__c1", "anchor": "drf",
         "field": "weighting_scheme", "scoreable": True, "frozen": frozen_ok},
        {"sheet_id": "drf__combiner__c3", "anchor": "drf",
         "field": "combiner", "scoreable": False, "frozen": frozen_ok},
        {"sheet_id": "mom6__n_groups__c2", "anchor": "mom6",
         "field": "n_groups", "scoreable": True,
         "frozen": tmp_path / "missing.frozen.yaml"},
        {"sheet_id": "str__long_leg__c4", "anchor": "str",
         "field": "long_leg", "scoreable": True, "frozen": frozen_ok},
        {"sheet_id": "drf__long_leg+weighting_scheme__c3", "anchor": "drf",
         "field": "long_leg+weighting_scheme", "scoreable": True, "frozen": frozen_ok},
    ]
    calls = []

    def stub_main(argv):
        calls.append(argv)
        if "--paper" in argv and argv[argv.index("--paper") + 1] == "drr":
            raise ValueError("boom")                               # str -> drr crashes
        return 0

    monkeypatch.setattr(run_librarian, "main", stub_main)
    log = t5x.run_variants(sheets, tmp_path / "out", ["--phase", "fake"])

    v = log["variants"]
    assert v["drf__weighting_scheme__c1"]["outcome"] == "ok"
    assert v["drf__combiner__c3"]["outcome"] == "skipped_non_scoreable"
    assert v["mom6__n_groups__c2"]["outcome"] == "missing_frozen_text"
    assert v["str__long_leg__c4"]["outcome"].startswith("CRASH")
    assert v["drf__long_leg+weighting_scheme__c3"]["outcome"] == "ok"
    assert log["crashed"] is True
    assert list(log["run_map"]) == ["drf__weighting_scheme__c1",
                                    "drf__long_leg+weighting_scheme__c3"]
    # The completed variants were invoked targeted, against their variant texts.
    argv = calls[0]
    assert argv[argv.index("--fields") + 1] == "weighting_scheme"
    assert argv[argv.index("--canonical-text") + 1] == str(frozen_ok)
    # Compound-target regression (MAJOR review finding 2026-09-02): both
    # components asked, never structural-only.
    argv = calls[2]
    assert argv[argv.index("--fields") + 1] == "long_leg,weighting_scheme"
    # Non-scoreable sheets must never be paid for.
    assert len(calls) == 3                                         # ok + crash + compound


def test_run_map_merges_on_partial_rerun(monkeypatch, tmp_path):
    """A --limit-variants/--anchor subset re-run extends the run map, never
    shrinks a previously complete one (review finding, 2026-09-02)."""
    frozen_dir = _REPO_ROOT / "evaluation" / "adversarial" / "frozen"
    if not list(frozen_dir.glob("drf__*.frozen.yaml")):
        pytest.skip("regenerable perturbed frozen texts (gitignored) not present")
    monkeypatch.setattr(run_librarian, "main", lambda argv: 0)
    mp = tmp_path / "map.json"
    mp.write_text(json.dumps({"prev__field__c9": "runs/t5/prev__field__c9"}),
                  encoding="utf-8")
    rc = t5x.main(["--limit-variants", "1", "--anchor", "drf", "--phase", "fake",
                   "--out-root", str(tmp_path / "t5"), "--run-map", str(mp)])
    assert rc == 0
    merged = json.loads(mp.read_text(encoding="utf-8"))
    assert "prev__field__c9" in merged                             # retained
    assert len(merged) == 2                                        # + this run's variant
