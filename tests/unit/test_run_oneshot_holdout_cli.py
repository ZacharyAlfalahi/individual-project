"""Pre-open refusals of the one-shot holdout CLI (`scripts/run_oneshot_holdout.py`).

The real run is allowed EXACTLY ONE holdout access. Stage-2 derivation reads the corrected
AuditReport and the RQ4 funnel artefact only after that access is spent, so an input that is
missing must be refused BEFORE the open — otherwise a typo costs the single read.

Data-free: every case here returns before any gate is opened, and `data/holdout/` is never touched.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLI = importlib.import_module("run_oneshot_holdout")

_REFUSED = 2


def _dev_prep(tmp_path: Path) -> Path:
    """The development preparation artefacts the real run requires before the open."""
    out = tmp_path / "oneshot"
    out.mkdir()
    (out / "p3_moderate_prior.json").write_text(
        json.dumps({"sigma_wide": 1.0, "sigma_moderate": 0.5, "sigma_sceptical": 0.25}), encoding="utf-8")
    (out / "oneshot_dev_pins.json").write_text(json.dumps({}), encoding="utf-8")
    return out


@pytest.mark.parametrize("missing", ["audit", "funnel"])
def test_real_refuses_before_the_open_when_a_stage2_input_is_missing(tmp_path, monkeypatch, capsys, missing):
    """A missing stage-2 input is refused up front, not after the holdout has been read."""
    monkeypatch.delenv("SCIENTIST_HOLDOUT_UNLOCK", raising=False)
    out = _dev_prep(tmp_path)
    present = tmp_path / "present.json"
    present.write_text(json.dumps({}), encoding="utf-8")
    absent = tmp_path / "absent.json"

    rc = CLI._run_real(
        out_dir=out,
        audit_report=absent if missing == "audit" else present,
        funnel_artefact=present if missing == "audit" else absent,
    )

    assert rc == _REFUSED
    err = capsys.readouterr().err
    assert "stage-2 derivation input missing" in err and "absent.json" in err


def test_the_stage2_check_precedes_the_unlock_check(tmp_path, monkeypatch, capsys):
    """Ordering is the point: with the unlock env also unset, the stage-2 input is what is named.

    If this ever reports the unlock refusal instead, the existence check has drifted below the
    gate and a missing input can again be discovered only after the single access is spent.
    """
    monkeypatch.delenv("SCIENTIST_HOLDOUT_UNLOCK", raising=False)
    out = _dev_prep(tmp_path)

    rc = CLI._run_real(out_dir=out, audit_report=tmp_path / "no_audit.json",
                       funnel_artefact=tmp_path / "no_funnel.json")

    assert rc == _REFUSED
    err = capsys.readouterr().err
    assert "stage-2 derivation input missing" in err
    assert "SCIENTIST_HOLDOUT_UNLOCK" not in err


def test_missing_development_preparation_is_still_refused_first(tmp_path, monkeypatch, capsys):
    """The pre-existing dev-prep refusal keeps its precedence (p3 / dev pins before anything else)."""
    monkeypatch.delenv("SCIENTIST_HOLDOUT_UNLOCK", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()

    rc = CLI._run_real(out_dir=empty, audit_report=tmp_path / "a.json", funnel_artefact=tmp_path / "f.json")

    assert rc == _REFUSED
    assert "development preparation artefact missing" in capsys.readouterr().err
