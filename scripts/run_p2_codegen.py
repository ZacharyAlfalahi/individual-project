"""P2 coverage-boundary codegen — CLI (mirrors scripts/run_p1_codegen.py).

`--dry-run` (the ONLY runnable mode until the scale census exists AND
`corpus.selection.status == 'frozen'` AND the frozen zoo-list hash matches):
reports the Phase-F pair, the shared prompt assets, and the P2 execute-gate
state. Zero generation calls, zero spend, no holdout contact.

`--execute`: REFUSES unless (1) the scale census artefact exists, (2)
`corpus.selection.status == 'frozen'`, and (3) the frozen zoo-list sha256 in
`p2_codegen.zoo_list.frozen_sha256` matches the live zoo-list — the three P2
gates. Gate (2) PASSES since T2-SEL-4 (2026-09-01: `corpus.selection.status:
frozen`); (1) and (3) still fail (no census; sha is TO_SET), so
`--execute` fails closed on those two.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.live_client import build_codegen_factory, load_budget  # noqa: E402
from evaluation.codegen.p2_boundary import assemble_below_floor, derive_reportable  # noqa: E402
from evaluation.codegen.p2_census_io import load_census, load_metadata  # noqa: E402
from evaluation.codegen.p2_driver import corpus_selection_status, run_p2_driver  # noqa: E402
from evaluation.codegen.p2_metrics import load_p2_scoring_thresholds  # noqa: E402
from evaluation.codegen.p2_selector import select_arms  # noqa: E402
from evaluation.codegen.panel_export import export_codegen_panel  # noqa: E402
from evaluation.codegen.runner import load_models, prompt_asset_hashes  # noqa: E402
from evaluation.codegen.sandbox import detect_mechanism  # noqa: E402


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (mirrors run_p1_codegen / run_librarian): KEY=VALUE -> os.environ."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# Build-only artefact paths (the scale census + frozen zoo-list).
_CENSUS_PATH = _REPO_ROOT / "data" / "development" / "codegen" / "p2_census.json"
_ZOO_LIST_PATH = _REPO_ROOT / "data" / "development" / "codegen" / "p2_zoo_list.txt"

_FROZEN_STATUS = "frozen"

# The below-floor coverage-boundary close-out artefacts.
_RESULTS_DIR = _REPO_ROOT / "results"
_BOUNDARY_JSON = _RESULTS_DIR / "rq2_p2_boundary.json"
_BOUNDARY_MD = _RESULTS_DIR / "rq2_p2_boundary.md"


def load_zoo_list(path: Path) -> tuple[str, ...]:
    """One paper name per line; blanks and ``#`` comments skipped."""
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            names.append(s)
    return tuple(names)


def zoo_list_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(names)).encode("utf-8")).hexdigest()


def census_exists() -> bool:
    return _CENSUS_PATH.is_file()


def zoo_list_freeze_ok(thresholds: dict) -> tuple[bool, str]:
    """The frozen zoo-list sha256 must be set (not TO_SET) and match the
    live zoo-list byte-for-byte."""
    frozen = str(thresholds.get("zoo_list", {}).get("frozen_sha256", "TO_SET"))
    if frozen == "TO_SET":
        return False, "p2_codegen.zoo_list.frozen_sha256 is TO_SET (zoo-list not frozen)"
    if not _ZOO_LIST_PATH.is_file():
        return False, f"zoo-list file missing: {_ZOO_LIST_PATH}"
    live = zoo_list_sha256(load_zoo_list(_ZOO_LIST_PATH))
    if live != frozen:
        return False, (f"zoo-list sha256 {live[:16]}… != frozen {frozen[:16]}… "
                       "— re-freeze before any generation call")
    return True, "zoo-list freeze verified"


def execute_gate() -> tuple[bool, list[str]]:
    """The three P2 gates. Returns (ok, per-gate status lines)."""
    thresholds = load_p2_scoring_thresholds()
    status = corpus_selection_status()
    lines: list[str] = []

    g1 = census_exists()
    lines.append(f"  [{'PASS' if g1 else 'REFUSE'}] scale census exists ({_CENSUS_PATH})")

    g2 = status == _FROZEN_STATUS
    lines.append(f"  [{'PASS' if g2 else 'REFUSE'}] corpus.selection.status == 'frozen' "
                 f"(is '{status}')")

    g3, g3_msg = zoo_list_freeze_ok(thresholds)
    lines.append(f"  [{'PASS' if g3 else 'REFUSE'}] zoo-list freeze — {g3_msg}")

    return (g1 and g2 and g3), lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--execute", dest="dry_run", action="store_false")
    ap.add_argument("--phase", choices=("dev", "reported"), default="dev",
                    help="dev = free non-reportable pair (default); reported = paid Phase-F pair")
    ap.add_argument("--scratch", default=str(_REPO_ROOT / "data" / "development" / "codegen"))
    args = ap.parse_args(argv)

    print("P2 coverage-boundary codegen — " + ("DRY RUN (zero generation calls)"
                                               if args.dry_run else f"EXECUTE (phase={args.phase})"))
    print(f"sandbox mechanism: {detect_mechanism()}")
    print("shared prompt assets (P1/P2):")
    for name, digest in prompt_asset_hashes().items():
        print(f"  {name}: {digest}")

    ok, gate_lines = execute_gate()
    print("P2 execute gate:")
    for line in gate_lines:
        print(line)

    if args.dry_run:
        print(f"phase_f pair (from thresholds): {[m['model_id'] for m in load_models('reported')]}")
        print("dry run complete — no generation performed.")
        return 0

    if not ok:
        print("REFUSED: the P2 execute gate is not satisfied (see the gate lines above); "
              "P2 emits numbers only once the census exists, the selection is frozen, and "
              "the zoo-list hash matches.", file=sys.stderr)
        return 2

    # Gate passed: load the census and select arms BEFORE any credential / panel /
    # generation work — the arm sizes decide whether ANY generation is warranted.
    thresholds = load_p2_scoring_thresholds()
    census = load_census(_CENSUS_PATH)
    meta = load_metadata(_CENSUS_PATH)
    selection = select_arms(census, load_zoo_list(_ZOO_LIST_PATH), thresholds)
    provenance = meta.get("spec_provenance", {})
    # Every routed (compilable/refused) member MUST carry spec provenance — build_overlaid
    # guarantees routed <=> has-spec, so a routed member missing from spec_provenance means the
    # census metadata is internally inconsistent; fail loud rather than over-claim reportability.
    routed_ids = {m.paper_id for m in census.members if m.compilable or m.refused}
    missing_prov = routed_ids - set(provenance)
    if missing_prov:
        print(f"REFUSED: census metadata.spec_provenance is missing routed members "
              f"{sorted(missing_prov)} — the census is internally inconsistent (rebuild it via "
              "scripts/build_p2_census.py).", file=sys.stderr)
        return 2
    corpus_status = corpus_selection_status()
    zoo_ok, _ = zoo_list_freeze_ok(thresholds)
    # The EVIDENCE is reportable iff every routed member's spec is a phase=report emission,
    # the selection is frozen, and the zoo-list hash matches — but --phase dev is the free
    # non-reportable pair by convention, so it is never reportable.
    reportable = derive_reportable(provenance, corpus_status, zoo_ok) and (args.phase != "dev")
    print(f"arms: |Arm A|={selection.arm_a_size} (refusal set), |Arm B|={len(selection.arm_b)} "
          f"(compilable), below_floor={selection.below_floor}")

    if selection.below_floor:
        # BELOW FLOOR: the registered under-power rule suppresses agreement/divergence, so
        # NO generation is performed — zero model calls, $0.00, no credential requirement,
        # no panel export, no driver. The evidence (the census + typed exclusions) is the
        # whole of the result.
        report_meta = {
            "reportable_basis": meta.get("reportable_basis"),
            "corpus_selection_status": corpus_status,
            "zoo_list_sha256": meta.get("candidate_zoo_list_sha256"),
            "spec_run_dirs": sorted({p.get("run_dir") for p in provenance.values() if p.get("run_dir")}),
            "code_commit": sorted({(p.get("code_commit") or "")[:7]
                                   for p in provenance.values() if p.get("code_commit")}),
            "generation": "none (below-floor rule): 0 model calls, $0.00",
        }
        result, report_md = assemble_below_floor(
            census, selection, thresholds, provenance=provenance,
            corpus_status=corpus_status, zoo_sha_ok=zoo_ok, meta=report_meta)
        result["reportable"] = reportable      # honours the --phase dev override
        result["phase"] = args.phase
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _BOUNDARY_JSON.write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        _BOUNDARY_MD.write_text(report_md, encoding="utf-8")
        print(f"BELOW FLOOR — SUPPRESSED: {result['below_floor_reason']}")
        print(f"cost: $0.00, {result['model_calls']} model calls. reportable={reportable}.")
        print(f"results written: {_BOUNDARY_JSON}")
        print(f"report written:  {_BOUNDARY_MD}")
        return 0

    # --- ABOVE-FLOOR paid path (requires credentials + generation) --------------------
    _load_dotenv(_REPO_ROOT / ".env")
    models = load_models(args.phase)
    print(f"{args.phase} pair (from thresholds): {[m['model_id'] for m in models]}")
    missing = [m["api_key_env"] for m in models if not os.environ.get(m.get("api_key_env", ""))]
    if missing:
        print(f"REFUSED: missing {args.phase} credentials {missing} (populate .env / export them)",
              file=sys.stderr)
        return 2

    if args.phase == "dev":
        print("DEV (free) run — NON-reportable by convention. The extracted_spec values are the "
              "REAL Phase-F Librarian emissions, but the free dev model pair is not reportable; "
              "the reported figures use the Phase-F pair (procurement-gated).")

    budget = load_budget()
    factory = build_codegen_factory(budget=budget, temperature=0.0)

    print("exporting corr-family engine panel …")
    panel_path, panel_sha = export_codegen_panel()
    print(f"  {panel_path}  sha256={panel_sha[:16]}…")

    result = run_p2_driver(
        census, selection, dry_run=False, census_available=True, phase=args.phase,
        client_factory=factory, panel_path=panel_path,
        sandbox_root=Path(args.scratch) / "p2_sandbox")
    result["budget"] = budget.summary()
    result["reportable"] = reportable

    n_runs = len(result["runs"])
    n_gen_err = len(result.get("generation_errors", []))
    print(f"runs: {n_runs} ({n_gen_err} generation errors); "
          f"budget spent ${budget.spent_usd:.4f} of ${budget.usd_cap:.0f}")

    out_path = Path(args.scratch) / f"p2_results_{args.phase}.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"results written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
