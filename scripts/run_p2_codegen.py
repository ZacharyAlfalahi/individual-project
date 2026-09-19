"""P2 coverage-boundary codegen — CLI (mirrors scripts/run_p1_codegen.py).

`--dry-run` (the ONLY runnable mode until the scale census exists AND
`corpus.selection.status == 'frozen'` AND the frozen zoo-list hash matches):
reports the Phase-F pair, the shared prompt assets, and the P2 execute-gate
state. Zero generation calls, zero spend, no holdout contact.

`--execute`: REFUSES unless (1) the scale census artefact exists, (2)
`corpus.selection.status == 'frozen'`, and (3) the frozen zoo-list sha256 in
`p2_codegen.zoo_list.frozen_sha256` matches the live zoo-list — the three P2
gates. The execute path proceeds only when all three pass.

Past the gates the arm sizes decide the path. BELOW the registered floor
(`p2_codegen.arms.below_floor_min_arm_a`) the default is the zero-call close-out: the
census fate table + typed exclusions, no generation, `$0`. That close-out is a recorded
result, so re-running it is REFUSED unless `--refresh-registered` is passed.
`--below-floor-override <departure id>` instead runs both arms in full under a named
authorised departure; every number it emits is labelled descriptive-only, it never
touches the registered close-out, and it refuses to overwrite its own prior output.

LLM policy (design decision, 2026-09-11): cached responses replay first and a pre-flight refuses
before any live call whose estimated vendor spend exceeds its cap. A replayed run cannot
verify returned model SKUs, so `sku_match` is tri-state (`null` = unverified).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.live_client import (  # noqa: E402
    BudgetExceededError,
    build_codegen_factory,
    load_budget,
)
from evaluation.codegen.p2_boundary import assemble_below_floor, derive_reportable  # noqa: E402
from evaluation.codegen.p2_census_io import load_census, load_metadata  # noqa: E402
from evaluation.codegen.p2_driver import (  # noqa: E402
    build_prompt_from_spec,
    corpus_selection_status,
    run_p2_driver,
)
from evaluation.codegen.jsonio import dump_json  # noqa: E402
from evaluation.codegen.p2_execute import (  # noqa: E402
    arm_b_compiler,
    assemble_executed,
    generated_series,
)
from evaluation.codegen.p2_metrics import load_p2_scoring_thresholds  # noqa: E402
from evaluation.codegen.p2_selector import select_arms  # noqa: E402
from evaluation.codegen.panel_export import (  # noqa: E402
    CODEGEN_PANEL,
    codegen_run_config,
    export_codegen_panel,
    source_panel_path,
)
from evaluation.codegen.preflight import (  # noqa: E402
    PER_VENDOR_USD_CEILING,
    metered_by_model,
    replay_first_factory,
    run_preflight,
)
from evaluation.codegen.runner import load_models, prompt_asset_hashes  # noqa: E402
from evaluation.codegen.sandbox import detect_mechanism  # noqa: E402
from scripts import basis_inputs  # noqa: E402


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

#: Content-addressed generation cache (model + seed + prompt) — the replay-first policy's
#: store, so a re-run (or a second basis) never pays twice for the same prompt.
_CACHE_ROOT = _REPO_ROOT / "runs" / "p2_codegen" / "cache"
#: Pre-flight per-vendor refusal cap on ESTIMATED spend (design decision, 2026-09-11).
DEFAULT_MAX_USD = PER_VENDOR_USD_CEILING


@dataclass(frozen=True)
class P2Paths:
    """Where one executed invocation writes. Without a basis: the scratch dir (gitignored
    dev partition). With one: the consistent-basis tree, so a basis run never overwrites a
    recorded artefact — including P1's own panel export, which keeps its own path."""

    results_json: Path
    report_md: Path
    sandbox_root: Path
    panel_out: Path
    source_panel: Path | None


def resolve_outputs(basis: str | None, phase: str, scratch: str) -> P2Paths:
    if basis is None:
        root = Path(scratch)
        # P2 exports its OWN panel file: writing CODEGEN_PANEL would rewrite the panel whose
        # sha256 P1's recorded result pins.
        return P2Paths(root / f"p2_boundary_{phase}.json", root / f"p2_boundary_{phase}.md",
                       root / "p2_sandbox", root / f"p2_{CODEGEN_PANEL.name}", None)
    root = basis_inputs.basis_dir(basis, "codegen", "p2_boundary")
    return P2Paths(root / f"p2_boundary_{phase}.json", root / f"p2_boundary_{phase}.md",
                   root / "sandbox", root / "engine_panel_corr.parquet",
                   basis_inputs.PANELS[basis_inputs.check_basis(basis)][0])


def compiler_chain():
    """The deterministic chain for the Arm-B third implementation, imported lazily so the
    zero-call below-floor path never pays for the engine. Identical to the chain
    ``scripts/run_t3_coverage`` scores RQ2 coverage with."""
    from agents.librarian.adapter.adapt import adapt_spec
    from agents.librarian.pipeline.spec_loader import spec_from_dict
    from agents.quant.config.runner import run_strategy
    return spec_from_dict, adapt_spec, run_strategy


def subs_provider():
    """A lazy ``() -> verified standing-substitution table`` — needed by every compile
    attempt, and (unlike the panel) cheap."""
    def provide():
        from scripts.run_quant import load_standing_subs_verified
        return load_standing_subs_verified()
    return provide


def panel_provider(basis: str | None, source_panel: Path | None = None):
    """A lazy ``() -> engine-shape dev panel`` under the codegen panel's OWN view
    (``panel_export.codegen_run_config`` — one source, never a mirrored literal), reading the
    SAME base panel the export read. Consulted only if an Arm-B spec actually compiles.
    Development data only."""
    def provide():
        from scripts.run_quant import load_inputs, resolve_base_panel
        base = source_panel if source_panel is not None else resolve_base_panel(basis)
        panel, _ = load_inputs(codegen_run_config(), base_panel=base)
        return panel
    return provide


def _view_fields(cfg) -> dict:
    return {
        "price_family": getattr(cfg.panel_view, "price_family", None),
        "stale_mask": getattr(cfg.panel_view, "stale_mask", None),
        "include_terminal_rows": getattr(cfg.panel_view, "include_terminal_rows", None),
        "signal_lag": getattr(cfg.construction, "signal_lag", None),
        "expost_trim": getattr(cfg.construction, "expost_trim", None),
    }


def compiler_view_record(source_panel: Path | None = None, panel_sha: str | None = None) -> dict:
    """What each side of the Arm-B comparison actually ran on.

    Both the export and the compiler take their view from ONE source
    (``panel_export.codegen_run_config``) and the same base panel, so a divergence between a
    generated implementation and the compiler is an implementation difference rather than a
    panel difference. The record states the view, the shared base panel, and the two known
    remaining asymmetries (the models' nine-column projection, and the compiler's wider
    signal merge) rather than implying they do not exist."""
    view = _view_fields(codegen_run_config())
    return {
        "view_source": "evaluation.codegen.panel_export.codegen_run_config (single source)",
        "compiler_view": view,
        "codegen_panel_view": view,
        "views_match": True,
        "shared_base_panel": None if source_panel is None else str(source_panel),
        "codegen_panel_sha256": panel_sha,
        "known_asymmetries": [
            "the models read the panel projected to the frozen nine-column schema (a rename "
            "and column subset, not a value change)",
            "the compiler's loader merges the full dev signal set, so its panel can carry "
            "(cusip, date) rows the projected codegen panel does not",
        ],
    }


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
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="export the engine panel from basis_inputs.PANELS[basis][0] and write "
                         "under the consistent-basis tree (omit for the default panel "
                         "resolution + scratch)")
    ap.add_argument("--below-floor-override", default=None, metavar="DEPARTURE_ID",
                    help="run BOTH arms although |Arm A| is below the registered floor, under "
                         "the named authorised departure (e.g. DEP-1). Every number emitted "
                         "is labelled descriptive-only; the registered suppressed close-out is "
                         "left untouched.")
    ap.add_argument("--max-usd-anthropic", type=float, default=DEFAULT_MAX_USD,
                    help="pre-flight refusal cap on estimated Anthropic spend (default $10)")
    ap.add_argument("--max-usd-gemini", type=float, default=DEFAULT_MAX_USD,
                    help="pre-flight refusal cap on estimated Gemini spend (default $10)")
    ap.add_argument("--refresh-registered", action="store_true",
                    help="deliberately rewrite the registered below-floor close-out "
                         "(results/rq2_p2_boundary.{json,md}); refused by default")
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

    override = (args.below_floor_override or "").strip() or None
    if selection.below_floor and override is None:
        # BELOW FLOOR: the registered under-power rule suppresses agreement/divergence, so
        # NO generation is performed — zero model calls, $0.00, no credential requirement,
        # no panel export, no driver. The evidence (the census + typed exclusions) is the
        # whole of the result.
        if args.basis is not None:
            print("REFUSED: --basis is meaningless for the zero-call below-floor close-out "
                  "(no panel is exported and no series is produced) and would only overwrite "
                  "the registered below-floor output. Drop --basis, or authorise the departure "
                  "with --below-floor-override.", file=sys.stderr)
            return 2
        if _BOUNDARY_JSON.exists() and not args.refresh_registered:
            print(f"REFUSED: {_BOUNDARY_JSON.name} already exists — the registered close-out "
                  "is a recorded result, and regenerating it would silently move its content "
                  "with any change to the metrics schema. Pass --refresh-registered to "
                  "rewrite it deliberately.", file=sys.stderr)
            return 2
        report_meta = {
            "reportable_basis": meta.get("reportable_basis"),
            "corpus_selection_status": corpus_status,
            "zoo_list_sha256": meta.get("candidate_zoo_list_sha256"),
            "spec_run_dirs": sorted({p.get("run_dir") for p in provenance.values() if p.get("run_dir")}),
            "generation": "none (below-floor rule): 0 model calls, $0.00",
        }
        result, report_md = assemble_below_floor(
            census, selection, thresholds, provenance=provenance,
            corpus_status=corpus_status, zoo_sha_ok=zoo_ok, meta=report_meta)
        result["reportable"] = reportable      # honours the --phase dev override
        result["phase"] = args.phase
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _BOUNDARY_JSON.write_text(dump_json(result), encoding="utf-8")
        _BOUNDARY_MD.write_text(report_md, encoding="utf-8")
        print(f"BELOW FLOOR — SUPPRESSED: {result['below_floor_reason']}")
        print(f"cost: $0.00, {result['model_calls']} model calls. reportable={reportable}.")
        print(f"results written: {_BOUNDARY_JSON}")
        print(f"report written:  {_BOUNDARY_MD}")
        return 0

    # --- GENERATION path: above floor, or below it under an authorised departure --------
    if selection.below_floor:
        print(f"BELOW FLOOR, PROCEEDING under the authorised departure {override!r}: "
              f"|Arm A|={selection.arm_a_size} is under the registered floor, so every number "
              "this run emits is DESCRIPTIVE ONLY. The registered outcome remains the "
              f"suppressed close-out ({_BOUNDARY_JSON.name}), which this run does not touch.")

    if override is not None and not selection.below_floor:
        print(f"NOTE: --below-floor-override {override!r} is ignored — this selection is "
              "ABOVE the registered floor, so the numbers are emitted by the registered "
              "path and carry no departure label.")

    paths = resolve_outputs(args.basis, args.phase, args.scratch)
    existing = [p for p in (paths.results_json, paths.report_md) if p.exists()]
    if existing:
        print(f"REFUSED: {', '.join(str(p) for p in existing)} already exist(s) — a run never "
              "overwrites a recorded result; move or delete it to re-run.", file=sys.stderr)
        return 2

    models = load_models(args.phase)
    print(f"{args.phase} pair (from thresholds): {[m['model_id'] for m in models]}")
    if args.phase == "dev":
        print("DEV (free) run — NON-reportable by convention. The extracted_spec values are the "
              "REAL Phase-F Librarian emissions, but the free dev model pair is not reportable; "
              "the reported figures use the Phase-F pair.")
    budget = load_budget()

    # Pre-flight (before any credential, client or call): replay-first census + spend estimate,
    # the same policy the P1 path runs under.
    prompts = {pid: build_prompt_from_spec(census.member(pid).extracted_spec)
               for pid in selection.members()}
    # EVERY vendor in the pair gets a cap: `cap_violations` only flags an uncapped vendor
    # when its estimate is positive, so a zero-priced (free-tier) model under an uncapped
    # vendor would otherwise pass the gate unbounded.
    caps = {m.get("vendor"): DEFAULT_MAX_USD for m in models if m.get("vendor")}
    caps["anthropic"] = args.max_usd_anthropic
    caps["gemini"] = args.max_usd_gemini
    try:
        preflight = run_preflight(selection.members(), models, cache_root=_CACHE_ROOT,
                                  budget=budget, caps_usd=caps,
                                  prompt_fn=lambda pid: prompts[pid])
    except BudgetExceededError as exc:           # an unpriced model cannot be bounded
        print(f"REFUSED (pre-flight): {exc}", file=sys.stderr)
        return 2
    print(f"pre-flight: {preflight['cache']['hits']} cache hits, "
          f"{preflight['cache']['misses']} misses; estimated spend "
          f"{preflight['estimate']['per_vendor_usd']} (caps {caps})")
    if preflight["refused"]:
        print("REFUSED (pre-flight, no call made): " + "; ".join(preflight["violations"]),
              file=sys.stderr)
        return 2

    live_models = preflight["live_models"]
    if live_models:
        _load_dotenv(_REPO_ROOT / ".env")
        missing = [m["api_key_env"] for m in models
                   if m["model_id"] in live_models and not os.environ.get(m.get("api_key_env", ""))]
        if missing:
            print(f"REFUSED: missing {args.phase} credentials {missing} "
                  f"(populate .env / export them)", file=sys.stderr)
            return 2
    else:
        print("pure cache replay — no credentials read, no live client constructed.")
    factory = replay_first_factory(
        live_models, budget,
        live_factory=(build_codegen_factory(budget=budget, temperature=0.0)
                      if live_models else None))

    # Resolve the base panel ONCE, so the export and the Arm-B compiler read the same file
    # (the default resolution honours $BBW_ANCHOR_PANEL; re-deriving it per side can differ).
    resolved_source = source_panel_path(paths.source_panel)
    print("exporting corr-family engine panel …")
    panel_path, panel_sha = export_codegen_panel(paths.panel_out, source_panel=resolved_source)
    print(f"  {panel_path}  sha256={panel_sha[:16]}…  (base {resolved_source.name})")

    driver_result = run_p2_driver(
        census, selection, dry_run=False, census_available=True, phase=args.phase,
        client_factory=factory, panel_path=panel_path, cache_root=_CACHE_ROOT,
        sandbox_root=paths.sandbox_root)
    n_gen_err = len(driver_result.get("generation_errors", []))
    print(f"runs: {len(driver_result['runs'])} ({n_gen_err} generation errors); "
          f"budget spent ${budget.spent_usd:.4f} of ${budget.usd_cap:.0f}")

    # The Arm-B third implementation: the deterministic compiler on each Arm-B member's
    # extracted spec. A refusal is recorded with its typed codes and NO series.
    readback = generated_series(driver_result["runs"])
    for rejected in readback.rejections:
        print(f"  series REJECTED {rejected['paper_id']} / {rejected['model_id']}: "
              f"{rejected['reason']} ({rejected['detail']})")
    load_spec, adapt, run = compiler_chain()
    attempts, compiler_series = arm_b_compiler(
        census, selection, load_spec=load_spec, adapt=adapt, run=run,
        subs_provider=subs_provider(),
        panel_provider=panel_provider(args.basis, resolved_source))
    for attempt in attempts:
        codes = f" ({len(attempt.refusal_codes)} typed refusal codes)" if attempt.refusal_codes else ""
        print(f"  Arm-B compiler {attempt.paper_id}: {attempt.outcome}{codes}")

    # SKU verification is TRI-STATE. A replayed response carries no returned model version
    # (the cache stores the text only), so "no mismatch seen" is NOT a match — recording it
    # as one would assert provenance the run cannot have.
    verified = [r for r in driver_result["runs"] if r.get("returned_model_version") is not None]
    mismatches = [(r["model_id"], r["returned_model_version"]) for r in verified
                  if r["model_id"] not in str(r["returned_model_version"])]
    n_runs = len(driver_result["runs"])
    if args.phase != "reported":
        sku_match, sku_reason = None, "not applicable — the dev pair is never reportable"
    elif mismatches:
        sku_match, sku_reason = False, f"returned SKU did not match the pinned model: {mismatches}"
    elif not verified:
        sku_match, sku_reason = None, (
            "UNVERIFIED — every response replayed from the cache, which stores no returned "
            "model version, so THIS artefact cannot evidence the served SKUs. The generations "
            "themselves are content-addressed in the cache and reproduce by hash; a live run's "
            "own output carries the versions")
    elif len(verified) < n_runs:
        # PARTIAL verification is not verification: the replayed cells' SKUs are evidenced
        # only by the live run that populated the cache, so this artefact cannot assert them.
        sku_match, sku_reason = None, (
            f"PARTIALLY VERIFIED — {len(verified)}/{n_runs} runs carried a returned model "
            "version and all of those matched; the rest replayed from cache, and their SKUs "
            "are carried only by the live run(s) that populated the cache")
    else:
        sku_match, sku_reason = True, (
            f"every returned SKU matched the pinned model ({len(verified)}/{n_runs} runs)")
    report_meta = {
        "reportable_basis": meta.get("reportable_basis"),
        "corpus_selection_status": corpus_status,
        "zoo_list_sha256": meta.get("candidate_zoo_list_sha256"),
        "spec_run_dirs": sorted({p.get("run_dir") for p in provenance.values() if p.get("run_dir")}),
        "generation": (f"{len(driver_result['runs'])} runs = {len(selection.members())} members × "
                       f"{len(models)} models; {budget.summary()['calls']} live model calls, "
                       f"${budget.spent_usd:.4f}"),
        "panel": f"{panel_path} (sha256 {panel_sha[:16]}…)",
        "basis": args.basis or "default (standard panel resolution)",
        "registered_outcome": (f"{_BOUNDARY_JSON.name} — the suppressed below-floor close-out, "
                               "untouched by this run"),
    }
    executed = assemble_executed(
        census, selection, thresholds, driver_result,
        readback=readback, compiler_attempts=attempts, compiler_series=compiler_series,
        model_ids=tuple(m["model_id"] for m in models),
        floor_override=(override if selection.below_floor else None),
        provenance=provenance, meta=report_meta,
        compiler_run_config=compiler_view_record(resolved_source, panel_sha))

    # Reportability is a CONJUNCTION of named blockers, not a single boolean. Each blocker is
    # recorded so a consumer sees WHY, and `evidence_reportable` (the census / phase-gate
    # quality the registered close-out means by "reportable") stays separate.
    kinds = census.refusals_by_kind()
    n_coverage, n_extraction = len(kinds["coverage"]), len(kinds["extraction"])
    dist = executed.result["metrics"].get("distribution", {})
    blockers: list[str] = []
    if selection.below_floor:
        blockers.append(f"below the registered power floor — run under departure {override}, "
                        "descriptive only")
    if not n_coverage:
        # The arm rule measures the COVERAGE boundary. An Arm A made only of extraction
        # refusals clears the floor by arithmetic while measuring a different thing, so it
        # is named here rather than left for a reader to infer from the arm size.
        blockers.append(
            f"Arm A holds no coverage refusals ({n_extraction} extraction refusals, "
            "specifications that never reached the coverage layer) — the boundary the arm "
            "rule measures is not reached on this corpus, whatever the arm size")
    if not selection.arm_b:
        blockers.append("the reference arm (Arm B) is empty — the router compiles no member, "
                        "so the experiment's arm contrast does not exist")
    if not dist.get("n_scored"):
        blockers.append("no member pair was scorable — nothing was measured")
    if not reportable:
        blockers.append("spec provenance or phase gate is not reportable")
    if sku_match is None:
        blockers.append(f"returned model SKUs unverified ({sku_reason})")
    elif sku_match is False:
        blockers.append(sku_reason)
    if driver_result.get("generation_errors"):
        blockers.append("at least one generation failed on infrastructure")

    result = executed.result
    result["phase"] = args.phase
    result["budget"] = budget.summary()
    result["generations"] = len(driver_result["runs"])
    result["live_model_calls"] = int(budget.summary()["calls"])
    result["spend_usd"] = float(budget.spent_usd)
    result["sku_match"] = sku_match
    result["sku_match_reason"] = sku_reason
    result["evidence_reportable"] = bool(reportable)
    result["reportable"] = not blockers
    result["reportable_blockers"] = blockers
    result["panel_sha256"] = panel_sha
    result["inputs"] = {
        "basis": args.basis,
        "panel_path": str(panel_path),
        "source_panel": None if paths.source_panel is None else str(paths.source_panel),
        "sandbox_root": str(paths.sandbox_root),
        "cache_root": str(_CACHE_ROOT),
    }
    result["llm_policy"] = {"preflight": preflight, "metered_by_model": metered_by_model(budget)}
    # An arm size means different things depending on WHY its members refused; both numbers
    # travel with every artefact that quotes one.
    result["arm_a_by_refusal_kind"] = {"coverage": n_coverage, "extraction": n_extraction}
    result["census"] = {
        "path": str(_CENSUS_PATH.relative_to(_REPO_ROOT)
                    if _CENSUS_PATH.is_relative_to(_REPO_ROOT) else _CENSUS_PATH),
        "sha256": hashlib.sha256(_CENSUS_PATH.read_bytes()).hexdigest(),
        "routing_mode": meta.get("routing_mode"),
        "zoo_list_sha256": meta.get("candidate_zoo_list_sha256"),
    }
    result["registered_outcome"] = {
        "path": str(_BOUNDARY_JSON.relative_to(_REPO_ROOT)
                    if _BOUNDARY_JSON.is_relative_to(_REPO_ROOT) else _BOUNDARY_JSON),
        "note": ("the registered below-floor output; its arm sizes need not describe this "
                 "census, and this run does not change it"),
    }

    paths.results_json.parent.mkdir(parents=True, exist_ok=True)
    paths.results_json.write_text(dump_json(result), encoding="utf-8")
    paths.report_md.write_text(executed.report_md, encoding="utf-8")

    print(f"arm A by refusal kind: coverage={n_coverage}, extraction={n_extraction}")
    print(f"agreement: {dist.get('n_agree')}/{dist.get('n_scored')} scored pairs agree; "
          f"correlation median {dist.get('correlation_median')}")
    print(f"reportable={result['reportable']} (sku_match={sku_match}); "
          f"spend ${budget.spent_usd:.4f} over {result['live_model_calls']} live calls "
          f"({result['generations']} generations)")
    for blocker in blockers:
        print(f"  not reportable: {blocker}")
    print(f"results written: {paths.results_json}")
    print(f"report written:  {paths.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
