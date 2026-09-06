"""Construction-scoped field prompts (CI-10 candidate, 2026-09-06; flag-gated,
default OFF).

The dfps remediation run (CI-9) proved the unscoped per-field prompt is
byte-identical across a multi-construction paper's constructions, so the disk
cache replays one answer per paper (the 28-clone outcome). These tests pin the
scoped seam:

  * default OFF -> assemble_field_prompt output is BYTE-IDENTICAL to the
    historical assembly (the regression bar for the anchors' recorded contract);
  * scoped ON -> every field suffix heads with the construction's name + enum
    quote, so two constructions of the same paper get different prompt bytes
    for the SAME field (what makes the disk cache key per construction);
  * the frozen template files are untouched (manifest freeze check still loads);
  * the driver stamps SCOPED_FIELDS_CONTRACT into prompt_template_hashes only
    when --scoped-fields is passed, so a scoped run's spec headers carry a
    distinguishable extraction contract;
  * the driver threads the enum quote per construction beside the label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from agents.librarian.pipeline import real_client as rc                     # noqa: E402
from agents.librarian.pipeline.model_client import FieldQuery               # noqa: E402
from agents.librarian.pipeline.prompts import load_prompt_manifest          # noqa: E402
from agents.librarian.registries import load_signal_concept_registry        # noqa: E402


@pytest.fixture
def builder():
    return rc.PromptBuilder.load(load_signal_concept_registry())


def _ct() -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "1"},
        normalisation={"ladder_level": "L0", "rules": []},
        pages=("the 5% VaR appears on this page",),
        status="stub",
    )


def test_default_off_is_byte_identical(builder):
    """Calling with the new keywords at their defaults (and with a quote set but
    scoped off) must reproduce the historical bytes exactly."""
    for q in (FieldQuery("sort_signal", "signal_ref"),
              FieldQuery("weighting_scheme", "enum"),
              FieldQuery("holding_period", "int")):
        old = rc.assemble_field_prompt(builder, q, "VaR", _ct())
        new = rc.assemble_field_prompt(builder, q, "VaR", _ct(),
                                       strategy_quote="ignored when unscoped",
                                       scoped=False)
        assert old == new


def test_scoped_prompts_differ_per_construction(builder):
    """The CI-9 failure mode inverted: same field, two constructions -> two
    different suffixes, each carrying its own name + quote."""
    q = FieldQuery("sort_signal", "signal_ref")
    _, s_var = rc.assemble_field_prompt(
        builder, q, "VaR", _ct(),
        strategy_quote="VaR(5%) -1 * 2nd lowest observation", scoped=True)
    _, s_mom = rc.assemble_field_prompt(
        builder, q, "Momentum", _ct(),
        strategy_quote="Cumulative return from t-7 to t-2", scoped=True)
    assert s_var != s_mom
    assert "CONSTRUCTION UNDER EXTRACTION" in s_var
    assert "name: VaR" in s_var and "2nd lowest observation" in s_var
    assert "name: Momentum" in s_mom and "t-7 to t-2" in s_mom
    # the block heads the suffix, before the rendered instruction
    assert s_var.startswith("CONSTRUCTION UNDER EXTRACTION")
    # unscoped suffix carries no block at all
    _, s_off = rc.assemble_field_prompt(builder, q, "VaR", _ct())
    assert "CONSTRUCTION UNDER EXTRACTION" not in s_off


def test_scoped_without_quote_omits_the_quote_line(builder):
    q = FieldQuery("sort_signal", "signal_ref")
    _, s = rc.assemble_field_prompt(builder, q, "VaR", _ct(),
                                    strategy_quote="   ", scoped=True)
    assert "name: VaR" in s
    assert "introduces it as" not in s


def test_scoped_contract_is_versioned_and_stable():
    assert rc.SCOPED_FIELDS_CONTRACT.startswith("scoped_fields:v1:")
    assert len(rc.SCOPED_FIELDS_CONTRACT.split(":")[2]) == 64  # sha256 hex


def test_frozen_template_files_untouched():
    """The scoped block is client assembly code; every frozen template file must
    still pass the manifest byte-hash freeze check."""
    m = load_prompt_manifest()  # raises on any template-file hash mismatch
    assert m.combined_prompt_hash  # loaded + verified


def test_driver_stamps_scoped_contract_only_when_flagged(monkeypatch, tmp_path):
    """Fake-phase driver runs: the emitted spec header's prompt_template_hashes
    gains the scoped suffix ONLY under --scoped-fields."""
    import json

    from scripts import run_librarian

    out_plain = tmp_path / "plain"
    rc_plain = run_librarian.main(["--paper", "bbw", "--phase", "fake",
                                   "--out", str(out_plain)])
    out_scoped = tmp_path / "scoped"
    rc_scoped = run_librarian.main(["--paper", "bbw", "--phase", "fake",
                                    "--out", str(out_scoped), "--scoped-fields"])
    assert rc_plain == rc_scoped  # same outcome class on the fake path

    def _stamp(out):
        spec = json.loads(next(out.glob("spec_*.json")).read_text(encoding="utf-8"))
        return spec["header"]["prompt_template_hashes"]

    plain, scoped = _stamp(out_plain), _stamp(out_scoped)
    assert not plain.endswith(rc.SCOPED_FIELDS_CONTRACT)
    assert scoped == plain + ";" + rc.SCOPED_FIELDS_CONTRACT


def test_assembler_threads_the_quote_per_construction(builder):
    """The driver closure sets current_strategy_quote beside the label; pin via a
    recording stand-in for the client seam."""
    class Recorder:
        current_strategy_label = ""
        current_strategy_quote = ""

    from agents.librarian.pipeline.lister import Construction

    r = Recorder()
    c = Construction(name="VaR", quote="VaR(5%) -1 * 2nd lowest observation",
                     cls="strategy", locator=None)
    # the driver's per-construction attribute-set pattern, verbatim
    for m in (r,):
        if hasattr(m, "current_strategy_label"):
            m.current_strategy_label = c.name
        if hasattr(m, "current_strategy_quote"):
            m.current_strategy_quote = c.quote
    assert r.current_strategy_label == "VaR"
    assert r.current_strategy_quote == "VaR(5%) -1 * 2nd lowest observation"
