"""T4(b) seam-3 emitted-spec deserialization and adaptation tests.

With an emitted spec in the run dir, seam 3 adapts it typed and byte-compares
its G2 rulebook to the key-derived rulebook; a spec-less dir stays NOT_GRADED."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_t4b_kat import _key_derived_spec, grade_seam_3  # noqa: E402

_KEY = _REPO_ROOT / "evaluation" / "synthetic" / "planted_key_synth_2026.yaml"


def _key():
    return yaml.safe_load(_KEY.read_text(encoding="utf-8"))


def _status(seam, name):
    for c in seam.checks:
        if c.name == name:
            return c.status
    raise AssertionError(f"check {name!r} absent: {[c.name for c in seam.checks]}")


def test_emitted_spec_grades_match_when_identical_to_key(tmp_path):
    """The strongest self-consistency case: the key-derived spec written AS the
    emitted spec must adapt clean and byte-match its own rulebook."""
    key = _key()
    spec = _key_derived_spec(key)
    (tmp_path / "spec_0.json").write_text(
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    s = grade_seam_3(tmp_path, key)
    assert _status(s, "emitted_spec.adapt.refused") == "MATCH"
    assert _status(s, "rulebook_byte_equal") == "MATCH"


def test_spec_less_run_dir_stays_not_graded(tmp_path):
    s = grade_seam_3(tmp_path, _key())
    assert _status(s, "emitted_spec.adapt") == "NOT_GRADED"
    assert _status(s, "rulebook_byte_equal") == "NOT_GRADED"


@pytest.mark.skipif(
    not (_REPO_ROOT / "runs" / "corpus_synth_report" / "synth").exists(),
    reason="live synth run dir absent (machine-local)",
)
def test_live_synth_run_unchanged_by_the_wiring():
    """The 2026-09-03 live run emitted no spec (Guard-1 refusal): the new wiring
    must leave its seam-3 verdict exactly as recorded."""
    s = grade_seam_3(_REPO_ROOT / "runs" / "corpus_synth_report" / "synth", _key())
    assert _status(s, "emitted_spec.adapt") == "NOT_GRADED"
    assert _status(s, "rulebook_byte_equal") == "NOT_GRADED"
