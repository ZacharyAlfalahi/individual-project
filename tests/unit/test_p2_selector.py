"""WS-C (P2) — arm selector: Arm-A = whole refusal set, Arm-B = first
min(|A|,max) compilable in frozen zoo-list order, below-floor flag, fail-loud."""

from __future__ import annotations

import pytest

from evaluation.codegen.census import CensusMember, CensusResult
from evaluation.codegen.p2_selector import P2SelectionError, select_arms

_TH = {"arms": {"arm_b_max": 5, "below_floor_min_arm_a": 5}}


def _refusal(pid: str) -> CensusMember:
    return CensusMember(pid, compilable=False, refused=True,
                        refusal_reason="refuse_no_strategy", text_quality_ok=True,
                        extracted_spec={"header": {"paper_id": pid}})


def _compilable(pid: str) -> CensusMember:
    return CensusMember(pid, compilable=True, refused=False, refusal_reason=None,
                        text_quality_ok=True, extracted_spec={"header": {"paper_id": pid}})


def _census(refusals, compilables) -> CensusResult:
    return CensusResult(tuple([_refusal(p) for p in refusals]
                              + [_compilable(p) for p in compilables]))


def test_arm_a_is_the_whole_refusal_set_in_zoo_order():
    # census input order is deliberately scrambled vs the zoo order
    census = _census(["r3", "r1", "r2", "r4", "r5", "r6"],
                     ["c2", "c1", "c3", "c4", "c5", "c6", "c7"])
    zoo = ["r1", "c1", "r2", "c2", "r3", "c3", "r4", "c4",
           "r5", "c5", "r6", "c6", "c7"]
    sel = select_arms(census, zoo, _TH)
    # Arm A = every refusal, ordered by the frozen zoo-list (not census order)
    assert sel.arm_a == ("r1", "r2", "r3", "r4", "r5", "r6")
    assert sel.arm_a_size == 6
    # Arm B = first min(6, 5)=5 compilable in zoo order
    assert sel.arm_b == ("c1", "c2", "c3", "c4", "c5")
    assert sel.below_floor is False
    assert sel.members() == sel.arm_a + sel.arm_b


def test_selection_is_deterministic():
    census = _census(["r1", "r2", "r3", "r4", "r5"], ["c1", "c2", "c3"])
    zoo = ["r1", "r2", "r3", "r4", "r5", "c1", "c2", "c3"]
    a = select_arms(census, zoo, _TH)
    b = select_arms(census, zoo, _TH)
    assert a == b


def test_arm_b_capped_at_compilable_count_when_fewer_than_target():
    census = _census(["r1", "r2", "r3", "r4", "r5"], ["c1", "c2"])
    zoo = ["r1", "r2", "r3", "r4", "r5", "c1", "c2"]
    sel = select_arms(census, zoo, _TH)
    assert sel.arm_a_size == 5 and sel.below_floor is False
    assert sel.arm_b == ("c1", "c2")            # only 2 compilable, though target is 5


def test_below_floor_when_arm_a_under_min():
    census = _census(["r1", "r2", "r3"], ["c1", "c2", "c3", "c4", "c5", "c6"])
    zoo = ["r1", "r2", "r3", "c1", "c2", "c3", "c4", "c5", "c6"]
    sel = select_arms(census, zoo, _TH)
    assert sel.arm_a_size == 3 and sel.below_floor is True
    # Arm B is still min(3, 5)=3 compilable — suppression happens downstream in metrics
    assert sel.arm_b == ("c1", "c2", "c3")


def test_member_absent_from_zoo_list_fails_loud():
    census = _census(["r1", "r2", "r3", "r4", "r5"], ["c1"])
    zoo = ["r1", "r2", "r3", "r4", "c1"]          # r5 missing from the frozen order
    with pytest.raises(P2SelectionError):
        select_arms(census, zoo, _TH)


def test_duplicate_zoo_entry_fails_loud():
    census = _census(["r1"], ["c1"])
    with pytest.raises(P2SelectionError):
        select_arms(census, ["r1", "c1", "r1"], _TH)
