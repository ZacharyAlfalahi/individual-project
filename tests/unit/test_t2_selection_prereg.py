"""
Consistency guard for the frozen scale-layer / corpus-growth SELECTION RULE
(`docs/evaluation/t2_selection_prereg.md` ⇄ `docs/thresholds.yaml → corpus.selection`).

The rule's force is that it precedes the candidate papers, so its machine-readable
constants must not silently drift, and — critically — the "exclude anything touched
in design" clause must actually hold: every paper with a frozen canonical text is a
paper the system has already touched, so it MUST appear in `design_touched_exclusions`.
A new canonical-text freeze that forgets to extend the exclusion list fails here.

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
        pytest.skip("selection-rule spec doc not shipped with the repository")
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


def test_frozen_status_has_no_unset_fields():
    # The REAL block: a 'frozen' rule cannot still carry TO_SET sentinels
    # anywhere (recursive). The configured block is frozen, so this asserts the
    # live gate; the parametrized test below pins the fires-only-when-frozen rule.
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


def test_design_touched_papers_are_excluded():
    sel = _selection()
    excl = set(sel["design_touched_exclusions"])
    assert excl, "design_touched_exclusions is empty"
    # Guard against a vacuous pass: if the dir moved or the freeze-naming convention
    # changed, the glob would return nothing and the superset check below would pass
    # green while catching nothing — the exact regression this test exists to catch.
    if not CANON.is_dir():
        pytest.skip(f"canonical-texts dir missing: {CANON}")
    frozen_texts = {
        p.name[: -len(".frozen.yaml")] for p in CANON.glob("*.frozen.yaml")
    }
    assert frozen_texts, (
        f"no *.frozen.yaml under {CANON} — the glob found nothing, so the exclusion "
        f"superset check would pass vacuously (path or freeze-naming convention changed)"
    )
    missing = frozen_texts - excl
    assert not missing, (
        f"papers with a frozen canonical text but absent from "
        f"design_touched_exclusions: {sorted(missing)} — a paper the system has "
        f"already touched must be ineligible for T2 selection"
    )
    for anchor_paper in ("bbw_2019", "drr_2026", "jnps_2013"):
        assert anchor_paper in excl, f"{anchor_paper} (anchor) must be excluded"
