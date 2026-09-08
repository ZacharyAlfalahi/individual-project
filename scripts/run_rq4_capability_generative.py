"""RQ4 Scientist CAPABILITY BENCHMARK — GENERATIVE ARMS (SC-SCI-17), sibling to SC-SCI-16.

SC-SCI-16 ran the capability benchmark (random_eligible + retrieval_only) over the two
non-entering corrected parents (drf, mom6) and DEFERRED the generative-LLM arms: the Phase-F
response cache is parent-specific (str prompts), so new parents need new live vendor calls gated
on SKU/cost authorization. That authorization was granted on 2026-09-07 (SC-SCI-17).

This SIBLING runs the UNCHANGED generation + gate stack over drf and mom6 at their fully corrected
(all-ON) lattice cells with ALL THREE sources — random_eligible + retrieval_only (MiniLM) + the
Phase-F generative pair (claude-sonnet-4-6, gemini-3.5-flash). Every other registered parameter
holds at its committed value (m=6, k=5, q=0.10, BBW4, realised-parent-sign direction,
compiled-transform dedup, cap=null, the generative wall). The random/retrieval arms reproduce the
SC-SCI-16 advancement outcomes (a determinism cross-check; only the generated_at stamp differs);
the generative arm is the new content.

SC-SCI-16's driver (run_rq4_capability_benchmark.py) and its deferral-guard test STAY FROZEN; this
sibling reuses SC-SCI-16's frozen helpers (corrected_parent, directional_ceiling, _ANCHORS) and
adds the generative sources exactly as the funnel wires them.

STATUS: descriptive proposal-quality diagnostic. No confirmatory selection claim attaches to a
non-entering parent; the HOLDOUT STAYS SHUT (no rehearsal, no one-shot holdout, dev window only).
The per-parent EXHAUSTIVE CEILING is computed beside the funnel.

DEV / HOLDOUT DISCIPLINE: every read is under data/development/; corrected_parent raises if any
parent month reaches 2022-01. thresholds.yaml is unchanged (no FROZEN_THRESHOLDS_SHA256 re-pin).
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
from shared.evaluation.thresholds import load_crowding_config  # noqa: E402

_GEN_AT = "2026-09-07T00:00:00Z"  # SC-SCI-17 authorization date (metadata stamp only)
# A dedicated parent-specific cache: content-addressed keys make collision impossible, but a
# separate root keeps the new drf/mom6 responses out of the committed str funnel cache.
_CACHE_ROOT = _REPO_ROOT / "runs" / "rq4_capability_generative" / "cache"


def build_generative_clients():
    """The Phase-F generative pair, wired exactly as the funnel does: load .env so the vendor
    key env-vars are present, then build from scientist.model_stack.phase_f."""
    F._load_dotenv(_REPO_ROOT / ".env")  # phase_d_client reads os.environ directly
    from agents.scientist.researcher.phase_d_client import build_phase_f_clients
    return list(build_phase_f_clients())


def run_capability_generative(anchor_id: str, *, embedder_mode: str = "minilm", k: int = 5,
                              m: int = 6, embedder=None, llm_clients=None) -> dict:
    """One parent through generation (random + retrieval + generative) -> G0-G5 -> ceiling arm.
    Entry is evaluated honestly and then bypassed (recorded); nothing advances to holdout.
    `llm_clients` (list of ModelClient) overrides the generative pair — tests inject a stub or [];
    default = the live Phase-F pair."""
    meta = C._ANCHORS[anchor_id]
    case, params = F.build_case(
        C._audit_report(anchor_id), strategy_id=anchor_id,
        case_id=f"rq4capgen_{anchor_id}",
        corrected_quant_config_ref=f"qc_{anchor_id}_corrected",
        corrected_run_ref=f"{anchor_id}_corrected")
    entered_would_be = bool(case.failed_check_ids)

    panel, base_rulebook, parent_returns, direction, holding_period = C.corrected_parent(anchor_id)
    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=holding_period,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]
    n_eligible = sum(1 for r in elig if r.eligible)

    macros = F.load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = F.bbw4_frame()
    crowding_cfg = load_crowding_config()
    crowding_factors = load_crowding_factor_bundle(crowding_cfg)
    reporting_delays = load_reporting_delays()
    embed, retrieval_label, retrieval_reportable, embedder_header = F.resolve_embedder(
        embedder_mode, embedder)

    if llm_clients is None:
        llm_clients = build_generative_clients()

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

    return {
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


def output_path(out_dir: Path, anchor_id: str, embedder_mode: str) -> Path:
    return out_dir / f"rq4_capability_generative_{anchor_id}_{embedder_mode}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchors", nargs="+", default=["drf", "mom6"],
                    choices=sorted(C._ANCHORS),
                    help="capability parents (default: the two non-entering anchors)")
    ap.add_argument("--embedder", choices=("offline", "minilm"), default="minilm",
                    help="minilm = registered production embedder; offline = stub (smoke only)")
    ap.add_argument("--out", default=str(
        _REPO_ROOT / "results" / "scientist" / "rq4_capability_generative"))
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "amendment": "SC-SCI-17",
        "run_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "embedder_mode": args.embedder,
        "git_commit": B._git_commit(), "thresholds_sha256": B._thresholds_sha256(),
    }
    if args.embedder == "minilm":
        header["embedder"] = F.minilm_header()

    llm_clients = build_generative_clients()
    print(f"generative pair: {[c.name for c in llm_clients]}")
    for anchor_id in args.anchors:
        print(f"=== capability parent (generative): {anchor_id} ===")
        result = run_capability_generative(anchor_id, embedder_mode=args.embedder,
                                           llm_clients=llm_clients)
        result["header"] = header
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
