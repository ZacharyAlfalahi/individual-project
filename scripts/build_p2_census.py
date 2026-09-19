"""Build the P2 scale-layer coverage census + frozen zoo-list (WS-C).

Reads the human enumeration golds (``evaluation/gold_specs/enum_bbw_2021.yaml`` +
``enum_dfps_2026.yaml``) for the PRE-REGISTERED member list + denominator (corpus_inventory
§5, 2026-08-19: 32 = 27 implement + 5 equity-refuse), then OVERLAYS the REAL Phase-F
Librarian emissions on disk and routes every spec-carrying member.

ROUTING (``--routing``, default ``router``). The contract's arm rule (§11(c)) defines the
arms by *the router's own partition*, not by the adjudicated implement/refuse split:

  * ``router`` (DEFAULT) — each member that carries a spec goes
    through the LIVE deterministic chain (``spec_from_dict`` -> ``adapt_spec`` with the
    hash-verified standing subs). Refused members carry the router's own typed refusal code
    (e.g. ``REVIEW_REQUIRED``); compiled members are the Arm-B pool. The adjudicated labels
    are recorded in metadata, and their 27/5 drift check always runs.
  * ``prereg`` — the transcription of the pre-registered CI-5 implement/refuse labels, kept
    so a label-partitioned census can be regenerated.

  * Members whose ``phase=report`` spec is found under ``runs/corpus_corpus_report/`` (the
    ``p2_spec_source`` default run dirs) are routed.
  * A member with no Phase-F spec on disk (e.g. a run that exited to review,
    ``AssemblyIncomplete``; the schema forbids partial emission, D31) carries NO spec and
    is typed a COUNTED eligibility exclusion (``extraction_review_exit_no_spec``) — never
    a silent drop. Eligibility exclusions are STILL census members, so the 32-id frozen
    order and the pre-registered denominator do not move.

Writes:
  * ``data/development/codegen/p2_census.json`` — the CensusResult (32 members incl. the
    real ``extracted_spec`` for each routed member) + metadata (prereg routing +
    denominator, real-spec provenance, observed dispositions).
  * ``data/development/codegen/p2_zoo_list.txt`` — the 32 member ids, one per line, in the
    frozen order (paper, then construction order within the gold).

Invariants (fail loud): the 27/5 drift check runs on the PRE-OVERLAY prereg label table
(a property of the golds, not of any run); the recomputed ``zoo_list_sha256`` must equal
the frozen value in ``docs/thresholds.yaml`` once set (the frozen order cannot move).

DEV ONLY: routing inputs are the enumeration golds; spec inputs are the Phase-F run
artefacts under ``runs/`` (local pipeline output, not shipped with the repository); output
under ``data/development/``.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.codegen.census import CensusInput, RoutingDecision, run_census  # noqa: E402
from evaluation.codegen.p2_census_io import save_census  # noqa: E402
from evaluation.codegen.p2_execute import refusal_codes  # noqa: E402
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


def dominant_refusal_code(codes: list[str]) -> str:
    """The refusal code a member is typed by: the most frequent, ties broken alphabetically
    so the census is reproducible. The full multiset is recorded beside it in metadata —
    typing by one code must never hide the rest."""
    counts = Counter(codes)
    if not counts:
        raise SystemExit(
            "dominant_refusal_code called with no codes — a refused AdaptResult always "
            "carries at least one, so an empty multiset means the routing result is malformed"
        )
    return sorted(counts, key=lambda c: (-counts[c], c))[0]


def router_route(
    member_rows: list[dict],
    specs_by_member: dict[str, dict],
    *,
    load_spec=None,
    adapt=None,
    subs=None,
) -> tuple[dict[str, RoutingDecision], dict[str, dict]]:
    """The LIVE deterministic-router partition — the contract's definition.

    Every member that carries a spec is routed by the real adapter; NO adjudicated label is
    consulted. Returns ``(route, observed_codes)`` where ``observed_codes`` records each
    refused member's full refusal-code multiset. Members without a spec are absent from the
    route map (``build_overlaid`` types them as eligibility exclusions).

    The chain is injected for testability; by default it is imported lazily, so the
    ``prereg`` mode needs neither the adapter nor the standing-substitution table."""
    if load_spec is None or adapt is None:
        from agents.librarian.adapter.adapt import adapt_spec as _adapt
        from agents.librarian.pipeline.spec_loader import spec_from_dict as _load
        load_spec = load_spec or _load
        adapt = adapt or _adapt
    if subs is None:
        from scripts.run_quant import load_standing_subs_verified
        subs = load_standing_subs_verified()

    route: dict[str, RoutingDecision] = {}
    observed: dict[str, dict] = {}
    for r in member_rows:
        pid = r["paper_id"]
        spec_dict = specs_by_member.get(pid)
        if spec_dict is None:
            continue
        result = adapt(load_spec(spec_dict), standing_subs=subs)
        codes = list(refusal_codes(result))
        observed[pid] = {"refused": bool(result.refused),
                         "refusal_codes": dict(Counter(codes))}
        route[pid] = (
            RoutingDecision(compilable=False, refused=True,
                            refusal_reason=dominant_refusal_code(codes))
            if result.refused else RoutingDecision(compilable=True, refused=False)
        )
    return route, observed


def prereg_routing_labels(route: dict[str, RoutingDecision]) -> dict[str, str]:
    """member_id -> ``"implement"`` | ``"refuse"`` — the CI-5 label table (a property of the
    golds), retained so the fate table can show the registered label beside the observed
    disposition."""
    return {pid: ("refuse" if d.refused else "implement") for pid, d in route.items()}


def assert_prereg_denominator(prereg_routing: dict[str, str]) -> tuple[int, int]:
    """FAIL LOUD (I3) unless the PRE-OVERLAY label table is exactly 27 implement / 5 refuse.
    This is a property of the golds, not of any run — it must keep guarding gold drift even
    when the overlay types spec-less members as eligibility exclusions."""
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


def main(argv: list[str] | None = None) -> int:
    from scripts.run_p2_codegen import zoo_list_sha256  # reuse the CLI's canonical hash

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--routing", choices=("router", "prereg"), default="router",
                    help="router (default): the live deterministic "
                         "router's own partition, as the contract's arm rule defines it; "
                         "prereg: the CI-5 label transcription, kept to regenerate a "
                         "label-partitioned census")
    args = ap.parse_args(argv)

    rows = _load_constructions()
    member_rows = _member_rows(rows)
    excluded = _excluded_non_members(rows)

    # --- I3: the drift check ALWAYS runs on the PRE-OVERLAY prereg label table ----------
    # (a property of the golds, independent of how this census routes).
    prereg_decisions = prereg_route(member_rows)
    prereg = prereg_routing_labels(prereg_decisions)
    n_impl, n_ref = assert_prereg_denominator(prereg)

    # --- overlay the REAL Phase-F emissions, then route ---------------------------------
    specs_by_member, provenance = load_member_specs(member_rows)
    if args.routing == "router":
        route, router_observed = router_route(member_rows, specs_by_member)
    else:
        route, router_observed = prereg_decisions, {}
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
            "moved (I1/I2); this is not a reporting change. STOP and reconcile.")

    metadata = {
        "source": ("member list + denominator from the enumeration golds (NO-MODEL-CONSULT): "
                   + ", ".join(_ENUM_GOLDS) + "; specs from REAL Phase-F Librarian emissions "
                   "(runs/corpus_corpus_report/); dispositions per `routing_mode` below"),
        "routing_mode": args.routing,
        "routing_note": (
            "router: every spec-carrying member routed by the LIVE deterministic chain "
            "(spec_from_dict -> adapt_spec with hash-verified standing subs), which is what "
            "the contract's arm rule (§11(c)) defines — the "
            "adjudicated implement/refuse labels are recorded but NOT used to partition. "
            "prereg: the transcription of those labels."
            if args.routing == "router" else
            "prereg: dispositions transcribe the adjudicated CI-5 implement/refuse labels "
            "(the contract's arm rule defines the router's own "
            "partition instead — see --routing router)."),
        "router_observed": router_observed,
        "denominator_prereg": "corpus_inventory §5 (2026-08-19): 32 = 27 implement + 5 equity-refuse",
        "extracted_spec_note": (
            "REAL Phase-F Librarian emissions (runs/corpus_corpus_report/, phase=report); members "
            "with no spec (a review exit; D31 forbids partial emission) are typed "
            "eligibility exclusions."),
        "prereg_routing": prereg,
        "prereg_denominator": {"implement": _EXPECT_IMPLEMENT, "refuse": _EXPECT_REFUSE,
                               "total": _EXPECT_IMPLEMENT + _EXPECT_REFUSE,
                               "source": "corpus_inventory §5 (2026-08-19)"},
        "spec_provenance": provenance,
        "observed_dispositions": {"compilable": n_compilable, "refused": n_refused,
                                  "eligibility_excluded": n_excluded_members},
        "reportable_basis": (f"phase=report scale-layer extraction; dispositions from the "
                             f"{args.routing} partition"),
        "exclusions": excluded,
        "candidate_zoo_list_sha256": candidate_sha,
    }
    save_census(result, _CENSUS_PATH, metadata=metadata)
    _ZOO_LIST_PATH.write_text("\n".join(zoo_list) + "\n", encoding="utf-8")

    print(f"routing: {args.routing}")
    print(f"census: {n_compilable} compilable + {n_refused} refused + {n_excluded_members} "
          f"eligibility exclusions = {len(result.members)} members")
    for pid, obs in sorted(router_observed.items()):
        print(f"  router {pid[:56]:56s} refused={obs['refused']} {obs['refusal_codes']}")
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
