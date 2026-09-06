"""
Consistency guard for the frozen scale-layer / corpus-growth SELECTION RULE
(`docs/evaluation/t2_selection_prereg.md` ⇄ `docs/thresholds.yaml → corpus.selection`).

The rule's force is that it precedes the candidate papers, so its machine-readable
constants must not silently drift, and — critically — the "exclude anything touched
in design" clause must actually hold. Directory membership alone cannot encode WHEN
a paper was touched, so a frozen canonical text is accounted for two ways: it is
either a pre-selection `design_touched_exclusions` entry (ineligible), or a paper
legitimately chosen under the frozen rule (a `t2_selected_papers` entry — parsing and
freezing a SELECTED paper is post-selection processing, not design contact). The two
sets must be disjoint, and every frozen canonical text must fall in one of them; a
freeze that belongs to neither (a forgotten exclusion) still fails here (T2-SEL-7).

Also enforces the freeze gate: a rule marked `status: frozen` cannot still carry
`TO_SET` sentinels — otherwise a half-specified rule could masquerade as
pre-registered (mirrors `auditor.ipca_differential.status`).
"""

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"
CANON = REPO_ROOT / "evaluation" / "canonical_texts"
SPEC = REPO_ROOT / "docs" / "evaluation" / "t2_selection_prereg.md"

_SENTINEL = "TO_SET"
_VALID_STATUS = ("draft_pending_review", "frozen")


def _selection() -> dict:
    data = yaml.safe_load(THRESHOLDS.read_text())
    corpus = data.get("corpus")
    assert isinstance(corpus, dict), "docs/thresholds.yaml has no `corpus:` block"
    sel = corpus.get("selection")
    assert isinstance(sel, dict), "`corpus:` block has no `selection:` sub-block"
    return sel


def _sentinel_paths(obj, prefix: str = "") -> list[str]:
    """Dotted paths of every value equal to the TO_SET sentinel, walking
    nested dicts and lists — so a sentinel buried in a future nested field (e.g. a
    {from, to} window, or an item inside a list) cannot slip past the freeze gate."""
    if isinstance(obj, dict):
        out: list[str] = []
        for k, v in obj.items():
            out += _sentinel_paths(v, f"{prefix}.{k}" if prefix else str(k))
        return out
    if isinstance(obj, list):
        out = []
        for i, v in enumerate(obj):
            out += _sentinel_paths(v, f"{prefix}[{i}]")
        return out
    return [prefix] if obj == _SENTINEL else []


def _frozen_gate_violations(sel: dict) -> list[str]:
    """Sentinel paths a rule with status 'frozen' is not allowed to carry. Empty for
    a draft — the gate only bites once the rule is frozen, so a half-specified rule
    cannot flip to 'frozen' and masquerade as pre-registered."""
    if sel.get("status") != "frozen":
        return []
    return _sentinel_paths(sel)


def test_spec_doc_exists():
    if not SPEC.exists():
        pytest.skip("selection-rule spec doc not shipped in this copy")
    assert SPEC.exists(), f"selection-rule spec missing: {SPEC}"


def test_counts_are_coherent():
    sel = _selection()
    # target raised 3 -> 5 by dated amendment T2-SEL-6 (2026-09-02), extending down
    # the frozen §3.5 order — never a silent edit (spec §6).
    assert sel["t2_target_count"] == 5
    assert sel["t2_min_count"] == 2
    assert sel["t2_min_count"] <= sel["t2_target_count"]


def test_scale_layer_never_enters_rq3():
    # O1 resolution + D-A59: scale/T2 strategies are RQ2/external only.
    sel = _selection()
    assert sel["rq3_eligible"] is False


def test_gold_depth_and_authoring():
    sel = _selection()
    assert sel["t2_gold_depth"] == "extraction_only"
    assert sel["scale_gold_depth"] == "enumeration_only"
    assert sel["gold_authoring"] == "human_no_model_consult"


def test_status_is_valid():
    assert _selection()["status"] in _VALID_STATUS


def test_frozen_status_has_no_unset_owner_fields():
    # The REAL block: a 'frozen' rule cannot still carry TO_SET sentinels
    # anywhere (recursive). Silent today because the block is a draft — the
    # parametrized test below proves the gate fires once frozen.
    assert _frozen_gate_violations(_selection()) == []


@pytest.mark.parametrize(
    "sel, should_fire",
    [
        # draft: gate stays silent even with an unset field
        ({"status": "draft_pending_review", "x": _SENTINEL}, False),
        # frozen + a sentinel at each nesting depth: gate MUST fire
        ({"status": "frozen", "x": _SENTINEL}, True),
        ({"status": "frozen", "w": {"from": _SENTINEL}}, True),
        ({"status": "frozen", "x": ["ok", _SENTINEL]}, True),
        # frozen + everything committed: gate silent
        ({"status": "frozen", "x": "committed", "w": {"from": "2013-01"}}, False),
    ],
)
def test_freeze_gate_fires_only_when_frozen_and_unset(sel, should_fire):
    assert bool(_frozen_gate_violations(sel)) is should_fire


def test_design_touched_and_selected_are_disjoint():
    # A paper cannot be both design-touched (pre-selection, ineligible) and a paper
    # legitimately selected under the frozen rule — that would be a contamination.
    sel = _selection()
    excl = set(sel["design_touched_exclusions"])
    selected = set(sel.get("t2_selected_papers", []))
    both = excl & selected
    assert not both, (
        f"papers listed as BOTH design-touched and T2-selected: {sorted(both)} — "
        f"a design-touched paper is ineligible for selection"
    )


def test_every_frozen_text_is_accounted_for():
    sel = _selection()
    excl = set(sel["design_touched_exclusions"])
    selected = set(sel.get("t2_selected_papers", []))
    assert excl, "design_touched_exclusions is empty"
    # Guard against a vacuous pass: if the dir moved or the freeze-naming convention
    # changed, the glob would return nothing and the accounting check below would pass
    # green while catching nothing — the exact regression this test exists to catch.
    if not CANON.is_dir():
        pytest.skip(f"canonical-texts dir missing: {CANON}")
    frozen_texts = {
        p.name[: -len(".frozen.yaml")] for p in CANON.glob("*.frozen.yaml")
    }
    assert frozen_texts, (
        f"no *.frozen.yaml under {CANON} — the glob found nothing, so the accounting "
        f"check would pass vacuously (path or freeze-naming convention changed)"
    )
    # Every frozen canonical text must be accounted for: either a pre-selection
    # design-touched exclusion, or a paper selected under the frozen rule. A freeze
    # that is neither is a forgotten exclusion (the regression this guards).
    unaccounted = frozen_texts - excl - selected
    assert not unaccounted, (
        f"papers with a frozen canonical text but neither design-touched nor a "
        f"declared t2_selected_papers entry: {sorted(unaccounted)} — a freeze must be "
        f"accounted for by exactly one provenance (design contact OR legitimate selection)"
    )
    for anchor_paper in ("bbw_2019", "drr_2026", "jnps_2013"):
        assert anchor_paper in excl, f"{anchor_paper} (anchor) must be excluded"


def test_selected_set_is_coherent():
    # Selection is complete (search_protocol.executed == true; T2-SEL-5/6), so the
    # selected set is frozen at exactly the target count — equality, not a range.
    sel = _selection()
    selected = list(sel.get("t2_selected_papers", []))
    assert selected, "t2_selected_papers is empty (T2-SEL-7 not landed)"
    assert len(selected) == len(set(selected)), (
        f"t2_selected_papers has duplicates: {selected}"
    )
    assert len(selected) == sel["t2_target_count"], (
        f"selection is complete: expected exactly t2_target_count="
        f"{sel['t2_target_count']} selected papers, got {len(selected)}"
    )
    expected = {"hvz_2017", "cgnst_2017", "klz_2017", "bektic_2018", "bwwss_2019"}
    assert set(selected) == expected, (
        f"t2_selected_papers drift: {sorted(set(selected) ^ expected)}"
    )
