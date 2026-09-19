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

from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from evaluation.codegen.census import CensusInput, RoutingDecision, run_census  # noqa: E402
from evaluation.codegen.p2_census_io import load_census, save_census  # noqa: E402
from evaluation.codegen.p2_driver import build_prompt_from_spec  # noqa: E402
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


def test_zoo_list_freeze_ok_rejects_to_set(tmp_path, monkeypatch):
    ok, msg = CLI.zoo_list_freeze_ok({"zoo_list": {"frozen_sha256": "TO_SET"}})
    assert not ok and "TO_SET" in msg


# ---- on-disk artefacts (built by scripts/build_p2_census.py) -------------------------------

def _census_or_skip():
    if not CLI.census_exists():
        pytest.skip("requires the built P2 census (data/development/codegen/p2_census.json) "
                    "— local pipeline output, not shipped with the repository")
    return load_census(CLI._CENSUS_PATH)


def test_built_census_carries_prereg_denom_and_router_dispositions():
    # The census partitions by the ROUTER's own decision (contract §11(c)). Pinned expectation
    # for the built census: 0 compilable + 5 refused + 27 eligibility exclusions (spec-less
    # members). The PRE-REGISTERED denominator is 27/5 and recorded; the id count and frozen
    # order do not move.
    census = _census_or_skip()
    meta = CLI.load_metadata(CLI._CENSUS_PATH)
    assert len(census.members) == 32
    assert meta["routing_mode"] == "router"
    assert len(census.compilable_set()) == 0
    assert len(census.refusal_set()) == 5
    assert len(census.eligibility_exclusions()) == 27
    # typed by the router's own refusal code, not by an adjudicated corpus fate
    assert all(m.refusal_reason == "REVIEW_REQUIRED" for m in census.refusal_set())
    assert all(m.exclusion_reason == "extraction_review_exit_no_spec"
               for m in census.eligibility_exclusions())
    assert meta["prereg_denominator"]["implement"] == 27
    assert meta["prereg_denominator"]["refuse"] == 5
    # the adjudicated labels survive as metadata, so both readings stay legible
    assert sum(1 for v in meta["prereg_routing"].values() if v == "refuse") == 5
    assert meta["observed_dispositions"] == {
        "compilable": 0, "refused": 5, "eligibility_excluded": 27}


def test_built_execute_gate_passes_all_three():
    _census_or_skip()
    ok, lines = CLI.execute_gate()
    assert ok, "\n".join(lines)
    assert all("PASS" in ln for ln in lines)


def test_select_arms_on_built_census_matches_the_pinned_arm_sizes():
    census = _census_or_skip()
    zoo = CLI.load_zoo_list(CLI._ZOO_LIST_PATH)
    thresholds = {"arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5}}
    sel = select_arms(census, zoo, thresholds)
    # Pinned expectation on the router's partition: |Arm A| = 5 meets the floor, and Arm B is
    # EMPTY (no member compiles).
    assert sel.arm_a_size == 5
    assert sel.arm_b == ()
    assert sel.below_floor is False


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


def _spec_for(pid):
    return {"header": {"paper_id": pid, "strategy_label": "synthetic sort"},
            "part1": {"formation_structure": "quintile sort"}}


def _below_floor_census():
    """2 refused + 3 compilable => |Arm A| = 2, below the registered floor of 5."""
    inputs = ([CensusInput(f"r{i}", _spec_for(f"r{i}"), True) for i in range(2)]
              + [CensusInput(f"c{i}", _spec_for(f"c{i}"), True) for i in range(3)])

    def route(inp):
        return (RoutingDecision(False, True, "refuse_asset_class")
                if inp.paper_id.startswith("r") else RoutingDecision(True, False))

    return run_census(inputs, route)


_STUB_CODE = (
    "```python\n"
    "import os\n"
    "rows = ['date,portfolio_return']\n"
    "vals = [0.01, -0.02, 0.03]\n"
    "i = 0\n"
    "for y in (2010, 2011, 2012):\n"
    "    for m in range(1, 13):\n"
    "        rows.append('%04d-%02d-28,%.6f' % (y, m, vals[i % 3]))\n"
    "        i += 1\n"
    "open(os.environ['OUTPUT_PATH'], 'w').write('\\n'.join(rows) + '\\n')\n"
    "```"
)


class _Code:
    def __init__(self, value):
        self.value = value


class _Refusal:
    def __init__(self, value):
        self.code = _Code(value)


class _RefusedAdapt:
    """A refusing compile: the deterministic compiler refuses the extracted spec."""
    refused = True
    refusals = (_Refusal("REVIEW_REQUIRED"),)
    leg_calls = ()


def _stub_cli(monkeypatch, census, tmp_path):
    """Stub the gate + artefacts for the CLI test (no engine)."""
    zoo = tuple(m.paper_id for m in census.members)
    routed = [m.paper_id for m in census.members if m.compilable or m.refused]
    monkeypatch.setattr(CLI, "execute_gate", lambda: (True, ["[PASS] (stubbed gate)"]))
    monkeypatch.setattr(CLI, "load_census", lambda p: census)
    monkeypatch.setattr(CLI, "load_metadata", lambda p: {
        "spec_provenance": {pid: {"manifest_phase": "report", "run_dir": "runs/x"}
                            for pid in routed},
        "candidate_zoo_list_sha256": "deadbeef", "reportable_basis": "x",
        "routing_mode": "router"})
    monkeypatch.setattr(CLI, "load_zoo_list", lambda p: zoo)
    monkeypatch.setattr(CLI, "corpus_selection_status", lambda: "frozen")
    monkeypatch.setattr(CLI, "zoo_list_freeze_ok", lambda th: (True, "ok"))
    # the registered close-out must never be touched by an override run
    monkeypatch.setattr(CLI, "_BOUNDARY_JSON", tmp_path / "registered.json")
    monkeypatch.setattr(CLI, "_BOUNDARY_MD", tmp_path / "registered.md")


def test_execute_below_floor_under_override_runs_both_arms_with_zero_model_calls(
        tmp_path, monkeypatch):
    if not CLI.census_exists():
        pytest.skip("requires the built P2 census (data/development/codegen/p2_census.json) "
                    "— local pipeline output, not shipped with the repository")
    census = _below_floor_census()
    _stub_cli(monkeypatch, census, tmp_path)

    # Pre-seed the generation cache so the pre-flight sees ZERO misses: the run is a pure
    # replay (no credentials, no live client), yet exercises the whole executed path.
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(CLI, "_CACHE_ROOT", cache_root)
    cache = ResponseCache(cache_root)
    models = CLI.load_models("reported")
    selection = select_arms(census, tuple(m.paper_id for m in census.members),
                            CLI.load_p2_scoring_thresholds())
    for pid in selection.members():
        prompt = build_prompt_from_spec(census.member(pid).extracted_spec)
        for model in models:
            cache.put(prompt, model["model_id"], 0, _STUB_CODE)

    def _boom(*a, **k):
        raise AssertionError("no live client may be built on a pure replay")

    monkeypatch.setattr(CLI, "build_codegen_factory", _boom)
    monkeypatch.setattr(CLI, "export_codegen_panel",
                        lambda out, source_panel=None: (tmp_path / "panel.parquet", "f" * 64))
    # the Arm-B third implementation: the stubbed compiler refuses these specs
    monkeypatch.setattr(CLI, "compiler_chain",
                        lambda: (lambda d: d, lambda s, **k: _RefusedAdapt(), lambda r, p: None))
    monkeypatch.setattr(CLI, "subs_provider", lambda: (lambda: "subs"))
    monkeypatch.setattr(CLI, "panel_provider", lambda *a, **k: (lambda: "panel"))

    rc = CLI.main(["--execute", "--phase", "reported", "--scratch", str(tmp_path),
                   "--below-floor-override", "DEP-1-test"])
    assert rc == 0

    result = json.loads((tmp_path / "p2_boundary_reported.json").read_text(encoding="utf-8"))
    assert result["mode"] == "executed"
    assert result["floor_override"] == "DEP-1-test"
    assert result["below_floor"] is True and result["suppressed"] is False
    assert result["live_model_calls"] == 0 and result["spend_usd"] == 0.0
    assert result["generations"] == 8
    assert result["llm_policy"]["preflight"]["cache"]["misses"] == 0
    # a replay cannot vouch for the SKUs, and a below-floor run is never reportable —
    # both are named rather than collapsed into a bare boolean
    assert result["sku_match"] is None
    assert result["evidence_reportable"] is True
    assert result["reportable"] is False
    assert any("below the registered power floor" in b for b in result["reportable_blockers"])
    assert any("unverified" in b for b in result["reportable_blockers"])
    # both arms scored: 2 Arm A + 2 Arm B members, 8 runs (4 members × 2 models)
    assert len(result["metrics"]["agreements"]) == 4
    assert len(result["runs"]) == 8
    # Arm B's compiler comparison does not exist — refusals are typed, no series invented
    assert [a["outcome"] for a in result["compiler_attempts"]] == ["refused", "refused"]
    assert result["metrics"]["divergences"] == []
    report = (tmp_path / "p2_boundary_reported.md").read_text(encoding="utf-8")
    assert "BELOW-FLOOR DEPARTURE — DEP-1-test" in report
    # the registered close-out artefacts were not written by this run
    assert not (tmp_path / "registered.json").exists()


def test_an_extraction_only_arm_a_with_no_arm_b_is_not_reportable(tmp_path, monkeypatch):
    """An extraction-only Arm A with no Arm B. Arm A of five clears the floor by ARITHMETIC,
    but every member refused for an extraction reason, Arm B is empty and no pair is
    scorable — so the artefact must name all three rather than report a number."""
    if not CLI.census_exists():
        pytest.skip("requires the built P2 census (data/development/codegen/p2_census.json) "
                    "— local pipeline output, not shipped with the repository")
    inputs = [CensusInput(f"p::{i}", _spec_for(f"p::{i}"), True) for i in range(5)]
    census = run_census(inputs, lambda inp: RoutingDecision(False, True, "REVIEW_REQUIRED"))
    _stub_cli(monkeypatch, census, tmp_path)

    cache_root = tmp_path / "cache"
    monkeypatch.setattr(CLI, "_CACHE_ROOT", cache_root)
    cache = ResponseCache(cache_root)
    models = CLI.load_models("reported")
    selection = select_arms(census, tuple(m.paper_id for m in census.members),
                            CLI.load_p2_scoring_thresholds())
    assert selection.arm_a_size == 5 and selection.arm_b == ()
    assert selection.below_floor is False                    # the floor is met by arithmetic
    for pid in selection.members():
        prompt = build_prompt_from_spec(census.member(pid).extracted_spec)
        # one model runs, the other returns prose: every pair then has an empty side, so no
        # pair is scorable
        cache.put(prompt, models[0]["model_id"], 0, _STUB_CODE)
        cache.put(prompt, models[1]["model_id"], 0, "I cannot implement this specification.")

    monkeypatch.setattr(CLI, "export_codegen_panel",
                        lambda out, source_panel=None: (tmp_path / "panel.parquet", "f" * 64))
    monkeypatch.setattr(CLI, "compiler_chain",
                        lambda: (lambda d: d, lambda s, **k: _RefusedAdapt(), lambda r, p: None))
    monkeypatch.setattr(CLI, "subs_provider", lambda: (lambda: "subs"))
    monkeypatch.setattr(CLI, "panel_provider", lambda *a, **k: (lambda: "panel"))

    rc = CLI.main(["--execute", "--phase", "reported", "--scratch", str(tmp_path)])
    assert rc == 0

    result = json.loads((tmp_path / "p2_boundary_reported.json").read_text(encoding="utf-8"))
    assert result["below_floor"] is False
    assert result["arm_a_by_refusal_kind"] == {"coverage": 0, "extraction": 5}
    assert result["reportable"] is False
    blockers = " | ".join(result["reportable_blockers"])
    assert "no coverage refusals" in blockers          # the boundary is not reached
    assert "Arm B) is empty" in blockers               # no contrast exists
    assert "nothing was measured" in blockers          # no pair was scorable
    # the census the arms came from is pinned on the artefact
    assert result["census"]["sha256"] and result["census"]["routing_mode"] is not None


def test_close_out_refuses_to_rewrite_the_registered_artefact(tmp_path, monkeypatch):
    """The registered close-out is a recorded result: regenerating it would silently move
    its content with any change to the metrics schema."""
    census = _below_floor_census()
    _stub_cli(monkeypatch, census, tmp_path)
    registered = tmp_path / "registered.json"
    registered.write_text('{"suppressed": true}', encoding="utf-8")

    assert CLI.main(["--execute", "--phase", "reported"]) == 2
    assert registered.read_text(encoding="utf-8") == '{"suppressed": true}'


def test_close_out_rewrites_the_registered_artefact_only_when_asked(tmp_path, monkeypatch):
    census = _below_floor_census()
    _stub_cli(monkeypatch, census, tmp_path)
    registered = tmp_path / "registered.json"
    registered.write_text('{"suppressed": true}', encoding="utf-8")

    assert CLI.main(["--execute", "--phase", "reported", "--refresh-registered"]) == 0
    assert json.loads(registered.read_text(encoding="utf-8"))["below_floor"] is True


def test_basis_without_an_override_refuses_below_floor(tmp_path, monkeypatch):
    census = _below_floor_census()
    _stub_cli(monkeypatch, census, tmp_path)
    assert CLI.main(["--execute", "--phase", "reported", "--basis", "clean"]) == 2
    assert not (tmp_path / "registered.json").exists()     # nothing written


def test_execute_refuses_to_overwrite_an_existing_result(tmp_path, monkeypatch):
    census = _below_floor_census()
    _stub_cli(monkeypatch, census, tmp_path)
    existing = tmp_path / "p2_boundary_reported.json"
    existing.write_text("{}", encoding="utf-8")
    rc = CLI.main(["--execute", "--phase", "reported", "--scratch", str(tmp_path),
                   "--below-floor-override", "DEP-1-test"])
    assert rc == 2
    assert existing.read_text(encoding="utf-8") == "{}"    # untouched


def test_router_routing_partitions_by_the_live_refusal_not_the_adjudicated_label():
    """The contract's arm rule defines the arms by the ROUTER's own
    decision; the adjudicated implement/refuse labels must not partition the census."""
    rows = [{"paper_id": "p::a", "paper": "p", "name": "A"},
            {"paper_id": "p::b", "paper": "p", "name": "B"},
            {"paper_id": "p::c", "paper": "p", "name": "C"}]      # c has no spec
    specs = {"p::a": {"pid": "p::a"}, "p::b": {"pid": "p::b"}}
    decisions = {
        "p::a": _RefusedAdapt(),                                   # refused: REVIEW_REQUIRED
        "p::b": type("_Ok", (), {"refused": False, "refusals": (), "leg_calls": ()})(),
    }

    route, observed = B.router_route(
        rows, specs, load_spec=lambda d: d["pid"], adapt=lambda s, **k: decisions[s],
        subs="subs")

    assert route["p::a"].refused is True
    assert route["p::a"].refusal_reason == "REVIEW_REQUIRED"       # the router's own code
    assert route["p::b"].compilable is True
    assert "p::c" not in route                                     # no spec => never routed
    assert observed["p::a"]["refusal_codes"] == {"REVIEW_REQUIRED": 1}


def test_refusal_kinds_partition_the_closed_vocabulary():
    """An arm size means different things depending on WHY its members refused, so the two
    kinds must partition the vocabulary exactly."""
    from evaluation.codegen.census import (
        COVERAGE_REFUSAL_REASONS,
        EXTRACTION_REFUSAL_REASONS,
        P2CensusError,
        REFUSAL_REASONS,
        refusal_kind,
    )

    assert COVERAGE_REFUSAL_REASONS | EXTRACTION_REFUSAL_REASONS == REFUSAL_REASONS
    assert not (COVERAGE_REFUSAL_REASONS & EXTRACTION_REFUSAL_REASONS)
    assert refusal_kind("REVIEW_REQUIRED") == "extraction"     # the spec never certified
    assert refusal_kind("REFUSED_ON_SILENCE") == "extraction"
    assert refusal_kind("refuse_asset_class") == "coverage"    # the engine cannot represent it
    assert refusal_kind("MISSING_BINDING") == "coverage"
    assert refusal_kind(None) is None
    with pytest.raises(P2CensusError):
        refusal_kind("not_a_refusal_code")


def test_refusals_by_kind_splits_the_arm():
    from evaluation.codegen.census import CensusMember, CensusResult

    census = CensusResult((
        CensusMember("p::a", False, True, "REVIEW_REQUIRED", True, {"header": {}}),
        CensusMember("p::b", False, True, "refuse_asset_class", True, {"header": {}}),
    ))
    kinds = census.refusals_by_kind()
    assert [m.paper_id for m in kinds["extraction"]] == ["p::a"]
    assert [m.paper_id for m in kinds["coverage"]] == ["p::b"]
    assert census.member("p::a").to_dict()["refusal_kind"] == "extraction"


def test_dominant_refusal_code_is_deterministic():
    assert B.dominant_refusal_code(["X", "Y", "X"]) == "X"         # most frequent
    assert B.dominant_refusal_code(["Y", "X"]) == "X"              # tie -> alphabetical


def test_census_accepts_a_live_router_code_and_still_rejects_an_invented_one():
    from evaluation.codegen.census import CensusMember, P2CensusError

    ok = CensusMember("p::a", False, True, "REVIEW_REQUIRED", True, {"header": {}})
    assert ok.disposition == "refused"
    with pytest.raises(P2CensusError):
        CensusMember("p::b", False, True, "not_a_refusal_code", True, {"header": {}})


def test_the_compiler_and_the_models_see_the_same_panel_view():
    """An Arm-B divergence is an implementation difference only if both sides share a view."""
    record = CLI.compiler_view_record()
    assert record["views_match"] is True
    assert record["compiler_view"] == record["codegen_panel_view"]


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
