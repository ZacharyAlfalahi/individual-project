"""lab_trim no-op gate driver (scripts/run_lab_trim_invariance_gate.py).

Pins the declaration (lab_trim declared beside the registered stale_price, nothing else; str's
not_applicable meas_err untouched), the fail-loud reproduction check, and the recorded gate
artefact (when present).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_lab_trim_invariance_gate as G


@pytest.mark.parametrize("anchor", ["str", "drf", "mom6"])
def test_gate_facts_declare_only_stale_price_and_lab_trim(anchor):
    facts = {f.toggle_id: f for f in G.gate_facts(anchor)}
    declared = {t for t, f in facts.items() if f.expect_no_op}
    assert declared == {"stale_price", "lab_trim"}
    if anchor == "str":
        assert facts["meas_err"].not_applicable and not facts["meas_err"].expect_no_op


def _core_and_report():
    bases = {b: {"∅": 0.01, "lab_trim": 0.0} for b in ("harsanyi_dividends", "walsh_coefficients", "doe_effects")}
    core = {"saturated_bases": bases, "shapley": {"shapley_values": {"lab_trim": 0.0}}}
    report = json.loads(json.dumps(core))
    return core, report


def test_reproduction_exact_passes_and_counts():
    core, report = _core_and_report()
    assert G.verify_reproduction(core, report) == {"exact": True, "n_coordinates_checked": 7}


def test_reproduction_refuses_any_drift():
    core, report = _core_and_report()
    core["saturated_bases"]["doe_effects"]["lab_trim"] = 1e-18
    with pytest.raises(G.LatticeMismatch, match="doe_effects"):
        G.verify_reproduction(core, report)


_ARTIFACT = Path(__file__).resolve().parents[2] / "results" / "auditor" / "lab_trim_invariance_gate.json"


@pytest.mark.skipif(not _ARTIFACT.is_file(),
                    reason="recorded lab_trim gate artefact not shipped with the repository")
def test_recorded_gate_artifact_tripwire():
    a = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert set(a["anchors"]) == {"str", "drf"}
    for anchor, block in a["anchors"].items():
        assert block["lattice_reproduction"]["exact"] is True
        verdicts = {i["toggle_id"]: i for i in block["invariance"]}
        assert set(verdicts) == {"stale_price", "lab_trim"}
        for inv in verdicts.values():
            assert inv["config_hashes_differ"] and inv["returns_identical"]
            assert inv["membership_verified"] and inv["metrics_identical"]
            assert inv["is_no_op"] is True
