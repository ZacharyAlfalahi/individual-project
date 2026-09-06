"""Build the P2 scale-layer coverage census + frozen zoo-list (WS-C).

Reads the human enumeration golds (``evaluation/gold_specs/enum_bbw_2021.yaml`` +
``enum_dfps_2026.yaml``) for the PRE-REGISTERED coverage partition (corpus_inventory §5,
2026-08-19: 32 = 27 implement + 5 equity-refuse), then OVERLAYS the REAL Phase-F
Librarian emissions on disk:

  * The five ``BBW_2021`` members carry the real specs emitted at
    ``runs/corpus_corpus_report/bbw2021/`` (phase=report): three compilable + two
    equity refusals.
  * The ``DFPS_2026`` Phase-F run exited to review with ZERO specs (``AssemblyIncomplete``;
    the schema forbids partial emission, D31), so all 27 DFPS members carry NO spec and
    are typed COUNTED eligibility exclusions (``extraction_review_exit_no_spec``) — never
    silent drops. Eligibility exclusions are STILL census members, so the 32-id frozen
    order and the pre-registered denominator do not move.

Writes:
  * ``data/development/codegen/p2_census.json`` — the CensusResult (32 members incl. the
    real ``extracted_spec`` for the five BBW members) + metadata (prereg routing +
    denominator, real-spec provenance, observed dispositions).
  * ``data/development/codegen/p2_zoo_list.txt`` — the 32 member ids, one per line, in the
    frozen order (paper, then construction order within the gold).

Invariants (fail loud): the 27/5 drift check runs on the PRE-OVERLAY prereg label table
(a property of the golds, not of any run); the recomputed ``zoo_list_sha256`` must equal
the frozen value in ``docs/thresholds.yaml`` once set (the frozen order cannot move).

DEV ONLY: routing inputs are the committed golds; spec inputs are the committed run
artefacts under ``runs/``; output under ``data/development/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.codegen.census import CensusInput, RoutingDecision, run_census  # noqa: E402
from evaluation.codegen.p2_census_io import save_census  # noqa: E402
from evaluation.codegen.p2_spec_source import load_member_specs, slug as _slug  # noqa: E402

_GOLD_DIR = _REPO_ROOT / "evaluation" / "gold_specs"
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"
_OUT_DIR = _REPO_ROOT / "data" / "development" / "codegen"
_CENSUS_PATH = _OUT_DIR / "p2_census.json"
_ZOO_LIST_PATH = _OUT_DIR / "p2_zoo_list.txt"

_ENUM_GOLDS = ("enum_bbw_2021", "enum_dfps_2026")

# Pre-registered coverage denominator (corpus_inventory §5, 2026-08-19). Matched by exact
# construction `name` from the enumeration golds — a transcription, not a fresh judgement.
_EQUITY_REFUSE_NAMES = frozenset({
    "Stock systematic risk (SR) quintile portfolios",
    "Stock idiosyncratic risk (IR) quintile portfolios",
    "Firm-level value factor (VALfirm / bev_mev)",
    "Firm-level equity momentum factor (EQMOMfirm / seas_1_1na)",
    "Firm-level leverage factor (LEVfirm / at_me)",
})
_EXCLUDED_NAMES = frozenset({
    "Firm-level factors from 153 equity signals (Jensen et al. 2022)",  # 153-family, out of scope
})
_EXPECT_IMPLEMENT = 27
_EXPECT_REFUSE = 5

#: The single eligibility-exclusion reason this close-out uses: a member whose Phase-F
#: extraction run exited to review with zero specs (D31 forbids partial emission).
_EXCLUSION_REASON_NO_SPEC = "extraction_review_exit_no_spec"


def _load_constructions() -> list[dict]:
    """Every enumeration-gold construction, tagged with its paper_id + a stable member id."""
    rows: list[dict] = []
    for stem in _ENUM_GOLDS:
        doc = yaml.safe_load((_GOLD_DIR / f"{stem}.yaml").read_text(encoding="utf-8"))
        paper = str(doc["paper_id"]).lower()
        for con in doc.get("constructions", []):
            rows.append({
                "paper_id": f"{paper}::{_slug(con['name'])}",
                "paper": paper,
                "name": con["name"],
                "cls": con.get("class"),
                "spec": {k: con.get(k) for k in ("name", "quote", "class", "grid")
                         if con.get(k) is not None},
            })
    return rows


def _member_rows(rows: list[dict]) -> list[dict]:
    """The routed (denominator) member rows: every ``strategy`` construction that is not a
    pre-registered exclusion, in gold order."""
    return [r for r in rows if r["cls"] == "strategy" and r["name"] not in _EXCLUDED_NAMES]


def _excluded_non_members(rows: list[dict]) -> list[dict]:
    """The pre-registered NON-member exclusions (auxiliaries + the 153-equity family) —
    metadata, never census members. Fail loud on an unexpected construction class."""
    excluded: list[dict] = []
    for r in rows:
        name, cls, pid = r["name"], r["cls"], r["paper_id"]
        if cls == "auxiliary" or name in _EXCLUDED_NAMES:
            excluded.append({"paper_id": pid, "name": name,
                             "reason": "auxiliary_factor" if cls == "auxiliary"
                             else "out_of_scope_153_equity_family"})
        elif cls != "strategy":
            raise SystemExit(f"unexpected class {cls!r} for {name!r} (expected strategy/auxiliary)")
    return excluded


def prereg_route(member_rows: list[dict]) -> dict[str, RoutingDecision]:
    """The PRE-REGISTERED routing decision for every member (a faithful transcription of the
    CI-5 labels): the five named equity constructions refuse (refuse_asset_class); every
    other strategy compiles."""
    route: dict[str, RoutingDecision] = {}
    for r in member_rows:
        if r["name"] in _EQUITY_REFUSE_NAMES:
            route[r["paper_id"]] = RoutingDecision(compilable=False, refused=True,
                                                   refusal_reason="refuse_asset_class")
        else:
            route[r["paper_id"]] = RoutingDecision(compilable=True, refused=False)
    return route


def prereg_routing_labels(route: dict[str, RoutingDecision]) -> dict[str, str]:
    """member_id -> ``"implement"`` | ``"refuse"`` — the CI-5 label table (a property of the
    golds), retained so the fate table can show the registered label beside the observed
    disposition."""
    return {pid: ("refuse" if d.refused else "implement") for pid, d in route.items()}


def assert_prereg_denominator(prereg_routing: dict[str, str]) -> tuple[int, int]:
    """FAIL LOUD (I3) unless the PRE-OVERLAY label table is exactly 27 implement / 5 refuse.
    This is a property of the golds, not of any run — it must keep guarding gold drift even
    though the overlay later types 27 members as eligibility exclusions."""
    n_impl = sum(1 for v in prereg_routing.values() if v == "implement")
    n_ref = sum(1 for v in prereg_routing.values() if v == "refuse")
    if (n_impl, n_ref) != (_EXPECT_IMPLEMENT, _EXPECT_REFUSE):
        raise SystemExit(
            f"prereg routing {n_impl} implement / {n_ref} refuse != pre-registered "
            f"{_EXPECT_IMPLEMENT} / {_EXPECT_REFUSE} — the enumeration golds drifted from the "
            "pre-registered denominator; reconcile before freezing")
    return n_impl, n_ref


def build_overlaid(
    member_rows: list[dict],
    route: dict[str, RoutingDecision],
    specs_by_member: dict[str, dict],
) -> tuple[list[CensusInput], dict[str, RoutingDecision]]:
    """Overlay real specs onto the pre-registered members.

    A member WITH a real spec stays routed (``text_quality_ok=True``, real spec attached,
    its pre-registered ``RoutingDecision`` retained). A member WITHOUT one becomes a COUNTED
    eligibility exclusion (``text_quality_ok=False``, ``extraction_review_exit_no_spec``) and
    is DROPPED from the route map — ``run_census`` must not route an ineligible input. Member
    order is preserved (eligibility exclusions are still members), so the frozen order and
    the id list cannot move."""
    inputs: list[CensusInput] = []
    route_overlaid: dict[str, RoutingDecision] = {}
    for r in member_rows:
        pid = r["paper_id"]
        spec = specs_by_member.get(pid)
        if spec is not None:
            inputs.append(CensusInput(paper_id=pid, extracted_spec=spec, text_quality_ok=True))
            route_overlaid[pid] = route[pid]
        else:
            inputs.append(CensusInput(paper_id=pid, extracted_spec=None, text_quality_ok=False,
                                      exclusion_reason=_EXCLUSION_REASON_NO_SPEC))
    return inputs, route_overlaid


def _frozen_zoo_sha() -> str:
    doc = yaml.safe_load(_THRESHOLDS.read_text(encoding="utf-8"))
    return str(doc["p2_codegen"]["zoo_list"]["frozen_sha256"])


def main() -> int:
    from scripts.run_p2_codegen import zoo_list_sha256  # reuse the CLI's canonical hash

    rows = _load_constructions()
    member_rows = _member_rows(rows)
    excluded = _excluded_non_members(rows)

    # --- I3: the drift check runs on the PRE-OVERLAY prereg label table -----------------
    route = prereg_route(member_rows)
    prereg = prereg_routing_labels(route)
    n_impl, n_ref = assert_prereg_denominator(prereg)

    # --- overlay the REAL Phase-F emissions ---------------------------------------------
    specs_by_member, provenance = load_member_specs(member_rows)
    inputs, route_overlaid = build_overlaid(member_rows, route, specs_by_member)
    result = run_census(inputs, lambda inp: route_overlaid[inp.paper_id])

    n_compilable = len(result.compilable_set())
    n_refused = len(result.refusal_set())
    n_excluded_members = len(result.eligibility_exclusions())

    zoo_list = tuple(m.paper_id for m in result.members)      # frozen order = census input order
    candidate_sha = zoo_list_sha256(zoo_list)

    # --- I1/I2: the frozen order cannot move --------------------------------------------
    frozen_sha = _frozen_zoo_sha()
    if frozen_sha == "TO_SET":
        print("NOTE: p2_codegen.zoo_list.frozen_sha256 is TO_SET (zoo-list not yet frozen); "
              "printing the candidate below to set.")
    elif candidate_sha != frozen_sha:
        raise SystemExit(
            f"zoo_list_sha256 {candidate_sha} != frozen {frozen_sha} — the frozen 32-id order "
            "moved (I1/I2); this is no longer a reporting change. STOP and reconcile.")

    metadata = {
        "source": ("routing from enumeration golds (NO-MODEL-CONSULT): " + ", ".join(_ENUM_GOLDS)
                   + "; specs from REAL Phase-F Librarian emissions (runs/corpus_corpus_report/)"),
        "denominator_prereg": "corpus_inventory §5 (2026-08-19): 32 = 27 implement + 5 equity-refuse",
        "extracted_spec_note": (
            "REAL Phase-F Librarian emissions (runs/corpus_corpus_report/, 2026-09-03, "
            "phase=report) for the five BBW_2021 members; the 27 DFPS_2026 members carry no spec "
            "(review exit, D31 forbids partial emission) and are typed eligibility exclusions."),
        "prereg_routing": prereg,
        "prereg_denominator": {"implement": _EXPECT_IMPLEMENT, "refuse": _EXPECT_REFUSE,
                               "total": _EXPECT_IMPLEMENT + _EXPECT_REFUSE,
                               "source": "corpus_inventory §5 (2026-08-19)"},
        "spec_provenance": provenance,
        "observed_dispositions": {"compilable": n_compilable, "refused": n_refused,
                                  "eligibility_excluded": n_excluded_members},
        "reportable_basis": ("phase=report scale-layer extraction; no generation performed "
                             "(below-floor rule)"),
        "exclusions": excluded,
        "candidate_zoo_list_sha256": candidate_sha,
    }
    save_census(result, _CENSUS_PATH, metadata=metadata)
    _ZOO_LIST_PATH.write_text("\n".join(zoo_list) + "\n", encoding="utf-8")

    print(f"census: {n_compilable} compilable + {n_refused} refused + {n_excluded_members} "
          f"eligibility exclusions = {len(result.members)} members")
    print(f"  prereg denominator (pre-overlay): {n_impl} implement / {n_ref} refuse")
    print(f"  {len(excluded)} pre-registered non-member exclusions (auxiliaries + 153-family)")
    print(f"  {_CENSUS_PATH}")
    print(f"  {_ZOO_LIST_PATH}")
    print(f"zoo_list_sha256 = {candidate_sha}"
          + ("  (== frozen)" if frozen_sha not in ("TO_SET",) and candidate_sha == frozen_sha
             else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
