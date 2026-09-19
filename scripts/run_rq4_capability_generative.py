"""RQ4 Scientist CAPABILITY BENCHMARK — GENERATIVE ARMS (SC-SCI-17), sibling to SC-SCI-16.

SC-SCI-16 defines the capability benchmark (random_eligible + retrieval_only) over the two
non-entering corrected parents (drf, mom6) and DEFERS the generative-LLM arms: the Phase-F
response cache is parent-specific (str prompts), so new parents need new live vendor calls gated
on SKU/cost authorization, which SC-SCI-17 (2026-09-07) grants.

This SIBLING runs the UNCHANGED generation + gate stack over drf and mom6 at their fully corrected
(all-ON) lattice cells with ALL THREE sources — random_eligible + retrieval_only (MiniLM) + the
Phase-F generative pair (claude-sonnet-4-6, gemini-3.5-flash). Every other registered parameter
holds at its registered value (m=6, k=5, q=0.10, BBW4, realised-parent-sign direction,
compiled-transform dedup, cap=null, the generative wall). The random/retrieval arms must reproduce
the SC-SCI-16 advancement outcomes (a determinism cross-check; only the generated_at stamp
differs); the generative arm is what this sibling adds.

SC-SCI-16's driver (run_rq4_capability_benchmark.py) and its deferral-guard test STAY FROZEN; this
sibling reuses SC-SCI-16's frozen helpers (corrected_parent, directional_ceiling, _ANCHORS) and
adds the generative sources exactly as the funnel wires them.

STATUS: descriptive proposal-quality diagnostic. No confirmatory selection claim attaches to a
non-entering parent; the HOLDOUT STAYS SHUT (no rehearsal, no one-shot holdout, dev window only).
The per-parent EXHAUSTIVE CEILING is computed beside the funnel.

DEV / HOLDOUT DISCIPLINE: every read is under data/development/; corrected_parent raises if any
parent month reaches 2022-01. thresholds.yaml is unchanged (no FROZEN_THRESHOLDS_SHA256 re-pin).

INPUTS (defaults = the recorded run): --basis, --audit-dir,
--factors-dir, --out exactly as run_rq4_capability_benchmark.py (basis audit default
results/consistent_basis/<basis>/audit/full_run; basis output default
results/consistent_basis/<basis>/rq4/capability_generative).

LLM POLICY (design decision 2026-09-11): cache-first + budgeted, via the funnel's
prepare_budgeted_clients (same per-call upper bound, caps and halting rules — see run_rq4_funnel).
main() rebuilds EVERY anchor's prompt through the wall before any call, looks each (model, seed,
prompt) key up in runs/rq4_capability_generative/cache, prices the misses and refuses before any
call when Anthropic or Gemini exceeds --max-usd-anthropic / --max-usd-gemini (default 10.0; a
non-finite, negative or > 10.0 cap is rejected). A zero-miss run constructs no live client; a live
client is built only for a model with a miss; an unplanned call halts the run. Hits/misses, the
estimate and the metered usage are embedded in each output JSON (`llm_budget`) of a flagged run; a
recorded-inputs run writes them to the gitignored runs/rq4_capability_generative/llm_budget/
<artifact stem>.llm_budget.json instead, so the output JSON carries only the recorded keys.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_rq4_capability_benchmark as C  # noqa: E402  (SC-SCI-16 frozen helpers; C is never mutated)
import run_rq4_exhaustive_benchmark as B  # noqa: E402
import run_rq4_funnel as F  # noqa: E402
from agents.scientist.experimentalist.audit_checks import load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.reporting.funnels import agent_quality_funnel, economic_funnel  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from agents.scientist.researcher.llm_source import LLMResearcherSource  # noqa: E402
from agents.scientist.schemas.outcomes import REFUSAL_CODES  # noqa: E402
from shared.evaluation.crowding import load_crowding_factor_bundle  # noqa: E402

_GEN_AT = "2026-09-07T00:00:00Z"  # SC-SCI-17 authorization date (metadata stamp only)
# A dedicated parent-specific cache: content-addressed keys make collision impossible, but a
# separate root keeps the drf/mom6 responses out of the str funnel cache (runs/rq4_funnel/cache).
_CACHE_ROOT = _REPO_ROOT / "runs" / "rq4_capability_generative" / "cache"
_DEFAULT_OUT = _REPO_ROOT / "results" / "scientist" / "rq4_capability_generative"
_BUDGET_SIDECAR_DIR = _REPO_ROOT / "runs" / "rq4_capability_generative" / "llm_budget"  # gitignored


def resolve_generative_paths(basis: str | None = None, *, audit_dir=None, factors_dir=None,
                             out=None) -> dict:
    """explicit flag > basis default (.../<basis>/{audit/full_run, factors,
    rq4/capability_generative}) > recorded default."""
    return F.resolve_basis_paths(basis, audit=audit_dir, factors_dir=factors_dir, out=out,
                                 default_audit=None, default_out=_DEFAULT_OUT,
                                 out_leaf="capability_generative")


def build_generative_clients(models=None):
    """Live Phase-F generative clients (scientist.model_stack.phase_f), wired exactly as the funnel
    does (F.build_live_pair: .env loaded, each client built as build_phase_f_clients builds it),
    restricted to the model ids in `models`. Reached only via the budget pre-flight, and only for
    the models with a cache miss within budget."""
    return F.build_live_pair("reported", models)


def _report_path(anchor_id: str, audit_dir) -> Path:
    """The anchor's corrected AuditReport: the per-anchor recorded run when `audit_dir` is None."""
    return C._audit_report(anchor_id) if audit_dir is None else Path(audit_dir) / f"{anchor_id}_report.json"


def _case(anchor_id: str, audit_dir, **case_kw):
    case_kw.setdefault("corrected_run_ref", f"{anchor_id}_corrected")
    return F.build_case(
        _report_path(anchor_id, audit_dir), strategy_id=anchor_id,
        case_id=f"rq4capgen_{anchor_id}",
        corrected_quant_config_ref=f"qc_{anchor_id}_corrected", **case_kw)


def _eligibility(library, available, holding_period: int) -> list:
    return [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=holding_period,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]


def planned_prompts(anchor_ids, *, audit_dir=None, m: int = 6) -> list[str]:
    """Every generative prompt the run will send, rebuilt through the wall (F.planned_prompt)
    BEFORE any call. Eligibility uses the registered _ANCHORS holding period, which
    corrected_parent asserts equals the anchor's QuantConfig — so it is the run's eligibility."""
    audit_dir = Path(audit_dir) if audit_dir is not None else None
    library = F.load_library()
    available = F.available_conditioning_variables()
    prompts = []
    for anchor_id in anchor_ids:
        case, _params = _case(anchor_id, audit_dir)
        elig = _eligibility(library, available, C._ANCHORS[anchor_id]["holding_period"])
        prompts.append(F.planned_prompt(case, elig, library, m))
    return prompts


def prepare_generative_clients(anchor_ids, *, audit_dir=None, k: int = 5, m: int = 6,
                               max_usd: dict | None = None, client_factory=None):
    """(BudgetedClients, record) for the Phase-F pair over every anchor's prompts; raises
    F.LLMBudgetExceeded before any client is built when an estimate exceeds its cap."""
    return F.prepare_budgeted_clients(
        planned_prompts(anchor_ids, audit_dir=audit_dir, m=m), stack_key="phase_f", k=k,
        cache_root=_CACHE_ROOT, max_usd=max_usd,
        client_factory=client_factory or build_generative_clients)


def run_capability_generative(anchor_id: str, *, embedder_mode: str = "minilm", k: int = 5,
                              m: int = 6, embedder=None, llm_clients=None,
                              basis: str | None = None, audit_dir=None, factors_dir=None,
                              max_usd: dict | None = None, client_factory=None) -> dict:
    """One parent through generation (random + retrieval + generative) -> G0-G5 -> ceiling arm.
    Entry is evaluated honestly and then bypassed (recorded); nothing advances to holdout.
    `llm_clients` (list of ModelClient) overrides the generative pair — tests inject a stub or [];
    default = the Phase-F pair behind the cache-first budget pre-flight for this anchor.
    `basis` / `audit_dir` / `factors_dir` select the inputs (None = the recorded run)."""
    meta = C._ANCHORS[anchor_id]
    recorded = F.is_recorded_inputs(basis, audit_dir, factors_dir)
    paths = resolve_generative_paths(basis, audit_dir=audit_dir, factors_dir=factors_dir)
    case_kw = ({} if recorded
               else {"corrected_run_ref": F.repo_relative(_report_path(anchor_id, paths["audit"]).parent)})
    case, params = _case(anchor_id, paths["audit"], **case_kw)
    entered_would_be = bool(case.failed_check_ids)

    llm_budget = None
    if llm_clients is None:                    # pre-flight BEFORE any data load or live call
        llm_clients, llm_budget = prepare_generative_clients(
            [anchor_id], audit_dir=paths["audit"], k=k, m=m, max_usd=max_usd,
            client_factory=client_factory)

    panel, base_rulebook, parent_returns, direction, holding_period = C.corrected_parent(
        anchor_id, basis)
    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = _eligibility(library, available, holding_period)
    n_eligible = sum(1 for r in elig if r.eligible)

    macros = F.load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = F.bbw4_frame(paths["factors_dir"])
    crowding_cfg = F.crowding_config_for(paths["factors_dir"])
    crowding_factors = load_crowding_factor_bundle(crowding_cfg)
    reporting_delays = load_reporting_delays()
    embed, retrieval_label, retrieval_reportable, embedder_header = F.resolve_embedder(
        embedder_mode, embedder)

    source_specs = [
        ("random_eligible", F.S.RandomEligibleSource(), "random_eligible"),
        ("retrieval_only", F.S.RetrievalOnlySource(embed), retrieval_label),
    ]
    for c in llm_clients:
        source_specs.append((f"llm_{c.name}",
                             LLMResearcherSource(c, cache=ResponseCache(_CACHE_ROOT)), c.name))

    generation: dict[str, list] = {}
    gen_errors: list[dict] = []
    for src_name, source, model_name in source_specs:
        try:
            generation[src_name] = F.S.run_all_seeds(
                source, case, elig, library, k=k, m=m, model=model_name,
                prompt_version="v1", generated_at=_GEN_AT)
        except F.UnplannedCallRefused:                 # a stale pre-flight halts the run — never
            raise                                      # a silently dropped arm
        except Exception as exc:                       # a generative pair member can rate-limit out
            gen_errors.append({"source": src_name, "error": f"{type(exc).__name__}: {exc}"[:300]})

    reports: dict[str, dict] = {}
    for src_name, seed_results in generation.items():
        proposals0 = seed_results[0].proposal_set.proposals if seed_results else ()
        rep = run_experimentalist(
            case, proposals0, library, panel=panel, base_rulebook=base_rulebook,
            bbw4_factors=bbw4, holding_period=holding_period,
            signal_lookback=meta["signal_lookback"], available_variables=available,
            macros=macros, m=m, q=params.q, cap=None, direction=direction,
            crowding_config=crowding_cfg, crowding_factors=crowding_factors,
            reporting_delays=reporting_delays)
        refusal_profile = {code: 0 for code in REFUSAL_CODES}
        refusal_profile.update(Counter(r.refusal_code.value for r in rep.records
                                       if getattr(r, "refusal_code", None) is not None))
        reports[src_name] = {
            "economic_funnel": economic_funnel(rep.records),
            "agent_quality_funnel": agent_quality_funnel(seed_results),
            "outcome_funnel": rep.funnel,
            "wrong_signed": rep.wrong_signed,
            "refusal_profile": refusal_profile,
            "advanced": list(rep.advanced),
        }

    # --- exhaustive ceiling: one canonical implementation per eligible mechanism ------------
    canon = B.canonical_proposals(case, elig, library)
    canonical_rows = B.evaluate_batch(
        canon, case=case, library=library, panel=panel, base_rulebook=base_rulebook, bbw4=bbw4,
        macros=macros, available=available, reporting_delays=reporting_delays,
        parent_mean_bp=float(parent_returns.mean()) * 1e4, holding_period=holding_period)
    ceiling = C.directional_ceiling(canonical_rows, direction)

    result = {
        "amendment": "SC-SCI-17", "generated_at": _GEN_AT,
        "strategy_id": anchor_id, "case_id": case.case_id,
        "entered_would_be": entered_would_be, "entry_bypassed": True,
        "failed_check_ids": list(case.failed_check_ids),
        "holding_period": holding_period, "signal_lookback": meta["signal_lookback"],
        "direction": direction,
        "corrected_parent_mean_per_month": float(parent_returns.mean()),
        "n_parent_months": int(len(parent_returns)),
        "n_eligible_mechanisms": n_eligible,
        "k": k, "m": m,
        "sources": list(generation.keys()),
        "generative_models": [c.name for c in llm_clients],
        "generation_errors": gen_errors,
        "reports": reports,
        "canonical_ceiling": ceiling,
        "canonical_rows": canonical_rows,
        "holdout": "NOT OPENED — capability parents never advance (SC-SCI-16/17); dev window only",
    }
    if not recorded:
        result["inputs"] = F.inputs_record(basis, _report_path(anchor_id, paths["audit"]),
                                           paths["factors_dir"])
    if llm_budget is not None:
        llm_budget["metered_usage"] = F.llm_usage(llm_clients)
        if recorded:                                   # recorded keys only; budget to the sidecar
            write_budget_sidecar(anchor_id, embedder_mode, llm_budget)
        else:
            result["llm_budget"] = llm_budget
    return result


def output_path(out_dir: Path, anchor_id: str, embedder_mode: str) -> Path:
    return out_dir / f"rq4_capability_generative_{anchor_id}_{embedder_mode}.json"


def write_budget_sidecar(anchor_id: str, embedder_mode: str, budget: dict) -> Path:
    """A recorded-inputs run's LLM budget record, under the gitignored _BUDGET_SIDECAR_DIR (never
    the output folder): <artifact stem>.llm_budget.json."""
    name = output_path(Path(), anchor_id, embedder_mode).name[: -len(".json")]
    return F._write_json(_BUDGET_SIDECAR_DIR, f"{name}.llm_budget.json", budget)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", nargs="+", default=["drf", "mom6"],
                    choices=sorted(C._ANCHORS),
                    help="capability parents (default: drf and mom6)")
    ap.add_argument("--embedder", choices=("offline", "minilm"), default="minilm",
                    help="minilm = registered production embedder; offline = stub (smoke only)")
    ap.add_argument("--basis", choices=F.BI.BASES, default=None,
                    help="return basis of the maximal panel (default: recorded clean run)")
    ap.add_argument("--audit-dir", default=None,
                    help="dir holding <anchor>_report.json (default results/auditor/"
                         "<anchor>_corrected/; with --basis: .../<basis>/audit/full_run)")
    ap.add_argument("--factors-dir", default=None,
                    help="dir with bbw_factors/mktb/str/mom6 parquets (default data/development/"
                         "factors; with --basis: .../<basis>/factors)")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/scientist/rq4_capability_generative; with "
                         "--basis: results/consistent_basis/<basis>/rq4/"
                         "capability_generative)")
    ap.add_argument("--max-usd-anthropic", type=F.usd_cap_arg,
                    default=F.DEFAULT_MAX_USD["anthropic"],
                    help="refuse before any call if the estimated Anthropic spend exceeds this "
                         "(finite, 0..10.0 — the per-vendor policy ceiling)")
    ap.add_argument("--max-usd-gemini", type=F.usd_cap_arg, default=F.DEFAULT_MAX_USD["gemini"],
                    help="refuse before any call if the estimated Gemini spend exceeds this "
                         "(finite, 0..10.0 — the per-vendor policy ceiling)")
    args = ap.parse_args(argv)
    paths = resolve_generative_paths(args.basis, audit_dir=args.audit_dir,
                                     factors_dir=args.factors_dir, out=args.out)
    recorded = F.is_recorded_inputs(args.basis, args.audit_dir, args.factors_dir)

    try:
        llm_clients, llm_budget = prepare_generative_clients(
            args.anchors, audit_dir=paths["audit"],
            max_usd={"anthropic": args.max_usd_anthropic, "gemini": args.max_usd_gemini})
    except F.LLMBudgetExceeded as exc:
        print(f"REFUSED (no live call made): {exc}")
        print(json.dumps(exc.record, indent=2, sort_keys=True, default=str))
        return 2
    out_dir = paths["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "amendment": "SC-SCI-17",
        "run_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "embedder_mode": args.embedder,
        "thresholds_sha256": B._thresholds_sha256(),
    }
    if args.embedder == "minilm":
        header["embedder"] = F.minilm_header()

    print(f"generative pair: {[c.name for c in llm_clients]} — {llm_budget['decision']} "
          f"(cache {llm_budget['cache']})")
    for anchor_id in args.anchors:
        print(f"=== capability parent (generative): {anchor_id} ===")
        result = run_capability_generative(anchor_id, embedder_mode=args.embedder,
                                           llm_clients=llm_clients, basis=args.basis,
                                           audit_dir=args.audit_dir,
                                           factors_dir=args.factors_dir)
        result["header"] = header
        budget = {**llm_budget, "scope": "pre-flight over every anchor in this invocation",
                  "metered_usage_cumulative": F.llm_usage(llm_clients)}
        if recorded:                                   # recorded keys only; budget to the sidecar
            print(f"  llm budget sidecar: "
                  f"{write_budget_sidecar(anchor_id, args.embedder, budget)}")
        else:
            result["llm_budget"] = budget
        path = output_path(out_dir, anchor_id, args.embedder)
        path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str),
                        encoding="utf-8")
        print(f"entered_would_be={result['entered_would_be']} (bypassed) "
              f"holding={result['holding_period']} direction={result['direction']} "
              f"parent_mean={result['corrected_parent_mean_per_month']:.6f}/mo "
              f"eligible={result['n_eligible_mechanisms']}")
        for src, rep in result["reports"].items():
            print(f"  [{src}] outcomes={rep['outcome_funnel']} advanced={len(rep['advanced'])} "
                  f"wrong_signed={rep['wrong_signed']}")
        if result["generation_errors"]:
            print(f"  generation_errors: {result['generation_errors']}")
        c = result["canonical_ceiling"]
        if c is not None:
            print(f"  ceiling: {c['mechanism']} alpha_t={c['alpha_t']} "
                  f"alpha_bp={c['alpha_bp_per_month']} p_raw={c['p_raw_descriptive']}")
        print(f"  written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
