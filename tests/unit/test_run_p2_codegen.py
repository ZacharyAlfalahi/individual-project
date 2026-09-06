"""WS-C (P2) — the P2 CLI execute gate, the census IO round-trip, and the census builder's
pre-registered partition. Offline (no LLM). Skips the on-disk-artefact checks if the census
has not been built yet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evaluation.codegen.census import CensusInput, RoutingDecision, run_census  # noqa: E402
from evaluation.codegen.p2_census_io import load_census, save_census  # noqa: E402
from evaluation.codegen.p2_selector import select_arms  # noqa: E402

import build_p2_census as B  # noqa: E402
import run_p2_codegen as CLI  # noqa: E402


# ---- census IO round-trip (spec-carrying) --------------------------------------------------

def test_census_io_round_trips_with_extracted_spec(tmp_path):
    inputs = [
        CensusInput("p::a", {"name": "A", "quote": "q"}, True),
        CensusInput("p::b", {"name": "B", "quote": "q"}, True),
    ]
    route = {"p::a": RoutingDecision(True, False), "p::b": RoutingDecision(False, True, "refuse_asset_class")}
    result = run_census(inputs, lambda i: route[i.paper_id])
    out = tmp_path / "c.json"
    save_census(result, out, metadata={"k": "v"})
    back = load_census(out)
    assert [m.paper_id for m in back.members] == ["p::a", "p::b"]
    assert back.member("p::a").extracted_spec == {"name": "A", "quote": "q"}   # spec survives
    assert back.member("p::b").refused and back.member("p::b").refusal_reason == "refuse_asset_class"


# ---- gate helpers --------------------------------------------------------------------------

def test_zoo_list_sha256_is_deterministic():
    a = CLI.zoo_list_sha256(("x", "y", "z"))
    b = CLI.zoo_list_sha256(("x", "y", "z"))
    assert a == b and a != CLI.zoo_list_sha256(("x", "y"))


def test_zoo_list_freeze_ok_rejects_owner_to_set(tmp_path, monkeypatch):
    ok, msg = CLI.zoo_list_freeze_ok({"zoo_list": {"frozen_sha256": "TO_SET"}})
    assert not ok and "TO_SET" in msg


# ---- on-disk artefacts (built by scripts/build_p2_census.py) -------------------------------

def _census_or_skip():
    if not CLI.census_exists():
        pytest.skip("p2_census.json not built (run scripts/build_p2_census.py)")
    return load_census(CLI._CENSUS_PATH)


def test_built_census_carries_prereg_denom_and_overlaid_dispositions():
    # After the real-spec overlay the OBSERVED dispositions are 3 compilable + 2 refused +
    # 27 eligibility exclusions (DFPS review exit), but the PRE-REGISTERED denominator is
    # still 27/5 — both are recorded; the id count and frozen order do not move.
    census = _census_or_skip()
    meta = CLI.load_metadata(CLI._CENSUS_PATH)
    assert len(census.members) == 32
    assert len(census.compilable_set()) == 3
    assert len(census.refusal_set()) == 2
    assert len(census.eligibility_exclusions()) == 27
    assert all(m.refusal_reason == "refuse_asset_class" for m in census.refusal_set())
    assert all(m.exclusion_reason == "extraction_review_exit_no_spec"
               for m in census.eligibility_exclusions())
    assert meta["prereg_denominator"]["implement"] == 27
    assert meta["prereg_denominator"]["refuse"] == 5
    assert meta["observed_dispositions"] == {
        "compilable": 3, "refused": 2, "eligibility_excluded": 27}


def test_built_execute_gate_passes_all_three():
    _census_or_skip()
    ok, lines = CLI.execute_gate()
    assert ok, "\n".join(lines)
    assert all("PASS" in ln for ln in lines)


def test_select_arms_on_built_census_is_below_floor():
    census = _census_or_skip()
    zoo = CLI.load_zoo_list(CLI._ZOO_LIST_PATH)
    thresholds = {"arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5}}
    sel = select_arms(census, zoo, thresholds)
    # |Arm A| = 2 (the two BBW equity refusals) < floor of 5 -> below floor binds
    assert sel.arm_a_size == 2 and len(sel.arm_b) == 2
    assert sel.below_floor is True


# ---- the real-spec overlay (build_p2_census, no disk write) --------------------------------

def test_overlay_types_specless_members_as_eligibility_exclusions():
    member_rows = B._member_rows(B._load_constructions())
    route = B.prereg_route(member_rows)
    # give real specs to only the first 5 members; the remaining 27 are spec-less
    specs = {r["paper_id"]: {"header": {}} for r in member_rows[:5]}
    inputs, route_overlaid = B.build_overlaid(member_rows, route, specs)
    census = run_census(inputs, lambda inp: route_overlaid[inp.paper_id])
    routed = len(census.compilable_set()) + len(census.refusal_set())
    excluded = census.eligibility_exclusions()
    assert routed == 5                                   # the 5 with specs keep their routing
    assert len(excluded) == len(member_rows) - 5         # the rest are typed exclusions
    assert all(m.exclusion_reason == "extraction_review_exit_no_spec" for m in excluded)
    assert len(census.members) == 32                     # the id count never moves


def test_overlay_preserves_frozen_order_and_sha():
    member_rows = B._member_rows(B._load_constructions())
    route = B.prereg_route(member_rows)
    full = {r["paper_id"]: {"header": {}} for r in member_rows}       # every member routed
    partial = {r["paper_id"]: {"header": {}} for r in member_rows[:5]}  # 5 routed, 27 excluded
    inp_f, ro_f = B.build_overlaid(member_rows, route, full)
    inp_p, ro_p = B.build_overlaid(member_rows, route, partial)
    zoo_f = tuple(m.paper_id for m in run_census(inp_f, lambda i: ro_f[i.paper_id]).members)
    zoo_p = tuple(m.paper_id for m in run_census(inp_p, lambda i: ro_p[i.paper_id]).members)
    assert zoo_f == zoo_p                                             # I1: id order unchanged
    assert CLI.zoo_list_sha256(zoo_f) == CLI.zoo_list_sha256(zoo_p)   # I2: sha unchanged


def test_prereg_denominator_drift_is_a_build_error():
    ok = {f"i{i}": "implement" for i in range(27)}
    ok.update({f"r{i}": "refuse" for i in range(5)})
    assert B.assert_prereg_denominator(ok) == (27, 5)                 # the frozen 27/5 passes
    drift = {f"i{i}": "implement" for i in range(26)}                 # 26/5 -> gold drift
    drift.update({f"r{i}": "refuse" for i in range(5)})
    with pytest.raises(SystemExit):
        B.assert_prereg_denominator(drift)


# ---- CLI below-floor short circuit: ZERO generation calls ----------------------------------

def test_execute_below_floor_makes_zero_generate_calls(tmp_path, monkeypatch):
    inputs = (
        [CensusInput(f"r{i}", {"header": {}}, True) for i in range(2)]
        + [CensusInput(f"c{i}", {"header": {}}, True) for i in range(3)]
        + [CensusInput(f"x{i}", None, False, exclusion_reason="extraction_review_exit_no_spec")
           for i in range(27)]
    )

    def route(inp):
        return (RoutingDecision(False, True, "refuse_asset_class")
                if inp.paper_id.startswith("r") else RoutingDecision(True, False))

    census = run_census(inputs, route)
    zoo = tuple(m.paper_id for m in census.members)

    # stub the gate + artefacts so the test is hermetic (no dependence on on-disk state)
    monkeypatch.setattr(CLI, "execute_gate", lambda: (True, ["[PASS] (stubbed gate)"]))
    monkeypatch.setattr(CLI, "load_census", lambda p: census)
    # provenance must cover every ROUTED member (2 refused + 3 compilable) — the CLI fails
    # loud otherwise (routed <=> has-spec by construction).
    routed = [f"r{i}" for i in range(2)] + [f"c{i}" for i in range(3)]
    monkeypatch.setattr(CLI, "load_metadata", lambda p: {
        "spec_provenance": {pid: {"manifest_phase": "report"} for pid in routed},
        "candidate_zoo_list_sha256": "deadbeef", "reportable_basis": "x"})
    monkeypatch.setattr(CLI, "load_zoo_list", lambda p: zoo)
    monkeypatch.setattr(CLI, "corpus_selection_status", lambda: "frozen")
    monkeypatch.setattr(CLI, "zoo_list_freeze_ok", lambda th: (True, "ok"))

    def _boom(*a, **k):
        raise AssertionError("build_codegen_factory must NOT be called below floor")

    monkeypatch.setattr(CLI, "build_codegen_factory", _boom)
    out_json, out_md = tmp_path / "b.json", tmp_path / "b.md"
    monkeypatch.setattr(CLI, "_BOUNDARY_JSON", out_json)
    monkeypatch.setattr(CLI, "_BOUNDARY_MD", out_md)

    rc = CLI.main(["--execute", "--phase", "reported"])     # would need paid creds if it generated
    assert rc == 0
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["below_floor"] is True and result["suppressed"] is True
    assert result["model_calls"] == 0 and result["spend_usd"] == 0.0
    assert out_md.exists()


def test_execute_refuses_when_a_routed_member_lacks_provenance(monkeypatch):
    # a routed member missing from spec_provenance => internally-inconsistent census => REFUSE
    inputs = [CensusInput("r0", {"header": {}}, True),
              CensusInput("c0", {"header": {}}, True)]

    def route(inp):
        return (RoutingDecision(False, True, "refuse_asset_class")
                if inp.paper_id == "r0" else RoutingDecision(True, False))

    census = run_census(inputs, route)
    monkeypatch.setattr(CLI, "execute_gate", lambda: (True, ["[PASS]"]))
    monkeypatch.setattr(CLI, "load_census", lambda p: census)
    monkeypatch.setattr(CLI, "load_metadata", lambda p: {          # only r0 has provenance; c0 missing
        "spec_provenance": {"r0": {"manifest_phase": "report"}}})
    monkeypatch.setattr(CLI, "load_zoo_list", lambda p: ("r0", "c0"))
    monkeypatch.setattr(CLI, "corpus_selection_status", lambda: "frozen")
    monkeypatch.setattr(CLI, "zoo_list_freeze_ok", lambda th: (True, "ok"))
    assert CLI.main(["--execute", "--phase", "reported"]) == 2
