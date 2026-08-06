"""E6 / DEV-G4-PBO-1 — the supplementary PBO wiring is provably INERT.

The three conditions, pinned: (i) additive, post-gate — the block is
computed FROM the finished records and returned separately; (ii) gate
decisions byte-identical with and without it — non-mutation proven on the
record structures, and no gate module imports pbo (source scan in the P3
pattern); (iii) nothing under `agents/scientist/experimentalist/` changed —
the wiring lives in the `reporting` sibling subpackage."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from agents.scientist.reporting.supplementary import (
    PBO_CAVEAT,
    PBO_SCOPE,
    family_pbo_supplementary,
)
from agents.scientist.schemas.evaluation import (
    BOOLEAN_FIELDS,
    Booleans,
    EvaluationRecord,
    Measurements,
)

_GATE_DIR = Path(__file__).resolve().parents[2] / "agents" / "scientist" / "experimentalist"


def _record(pid: str, *, audit_clean: bool = True) -> EvaluationRecord:
    flags = {f: True for f in BOOLEAN_FIELDS}
    flags["audit_clean"] = audit_clean
    if not audit_clean:
        flags["bh_survived"] = False
        flags["cpcv_qualified"] = False
    return EvaluationRecord(proposal_id=pid, booleans=Booleans(**flags),
                            refusal_code=None, measurements=Measurements())


def _returns(seed: int, t: int = 120, mean: float = 0.0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=t, freq="ME")
    return pd.Series(mean + 0.01 * rng.standard_normal(t), index=idx)


def test_no_gate_module_imports_pbo():
    for path in sorted(_GATE_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "shared.stats.pbo" not in src, f"{path.name} imports the pbo module"
        assert "pbo_cscv" not in src, f"{path.name} references pbo_cscv"


def test_records_are_byte_identical_with_and_without_the_supplementary():
    records = tuple(_record(pid) for pid in ("a", "b", "c", "d"))
    returns = {pid: _returns(i) for i, pid in enumerate(("a", "b", "c", "d"))}
    before = [dataclasses.asdict(r) for r in records]
    block = family_pbo_supplementary(records, returns)
    after = [dataclasses.asdict(r) for r in records]
    assert before == after                       # gate outputs untouched, byte-identical
    assert block["status"] == "computed"         # the block is a SEPARATE artefact
    assert "pbo" in block and block is not records


def test_computed_block_carries_scope_caveat_and_geometry():
    records = tuple(_record(pid) for pid in ("a", "b", "c", "d"))
    returns = {pid: _returns(i) for i, pid in enumerate(("a", "b", "c", "d"))}
    block = family_pbo_supplementary(records, returns, purge=3, embargo=2)
    assert block["scope"] == PBO_SCOPE and block["caveat"] == PBO_CAVEAT
    assert block["geometry"] == {"n_groups": 8, "test_groups": 2, "purge": 3, "embargo": 2}
    assert 0.0 <= block["pbo"] <= 1.0 and block["n_candidates"] == 4


def test_dominant_candidate_yields_zero_pbo():
    records = tuple(_record(pid) for pid in ("dom", "x", "y", "z"))
    returns = {pid: _returns(i) for i, pid in enumerate(("dom", "x", "y", "z"))}
    returns["dom"] = returns["dom"] * 0.01 + 0.05          # dominates every fold
    block = family_pbo_supplementary(records, returns)
    assert block["status"] == "computed" and block["pbo"] == 0.0


def test_membership_is_audit_clean_only():
    records = (_record("a"), _record("b"), _record("dirty", audit_clean=False))
    returns = {pid: _returns(i) for i, pid in enumerate(("a", "b", "dirty"))}
    block = family_pbo_supplementary(records, returns)
    assert block["n_candidates"] == 2
    assert "dirty" not in block["is_best_counts"]


def test_small_family_is_typed_not_raised():
    records = (_record("only"),)
    block = family_pbo_supplementary(records, {"only": _returns(0)})
    assert block["status"] == "insufficient_family" and block["n_candidates"] == 1
