"""RQ4 exhaustive canonical-mechanism benchmark — descriptive diagnostic.

One canonical implementation per eligible mechanism for the eligible parent (str),
evaluated through the SAME gate stack the development funnel uses, plus the rank of
every funnel proposal inside that canonical distribution. The question it answers:
did the proposal sources miss stronger eligible mechanisms, or does the library's
own span contain nothing that qualifies?

Canonical mapping rule: the deterministic default-config rule
`sources._first_config` (spec §8.1 — "first reachable (template, variable) in
canonical order, first lag, first form"), committed as part of the retrieval rung
BEFORE any funnel performance run. This driver post-dates the recorded funnel runs
and exercises that rule verbatim, so no per-mechanism choice is made after results
were observed; every canonical implementation is a pure function of the frozen
mechanism library and the pre-existing rule.

Constraints honoured STRUCTURALLY, not by convention:
  * not another significance family: no BH-FDR is computed (run_fdr is never
    imported); raw p-values are descriptive annotations only;
  * cannot nominate holdout candidates: no G4/G5/G6 code path exists here;
  * exclusions logged: gate-refused canonical candidates carry their typed refusal
    code; compiled-transform collisions log DUPLICATE_PROPOSAL; counts reported.

Generative arms replay from the immutable content-addressed cache
(runs/rq4_funnel/cache); a cache miss RAISES (CacheOnlyClient) — a live vendor
call is impossible and the run costs nothing. Development window only; the
holdout is never read.

Output:
  results/scientist/rq4_exhaustive_benchmark.json
  results/rq4_exhaustive_benchmark.md
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_rq4_funnel as F  # noqa: E402  (reuses the funnel's own seams)

from agents.quant.library.characteristic_sort import regress_on_benchmark  # noqa: E402
from agents.scientist.experimentalist.audit_checks import audit_g2, load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.compiler import compile_g1a  # noqa: E402
from agents.scientist.experimentalist.execution_verifier import execute_g1b  # noqa: E402
from agents.scientist.experimentalist.inference import two_sided_p  # noqa: E402
from agents.scientist.experimentalist.orchestrator import _transform_signature  # noqa: E402
from agents.scientist.experimentalist.validator import validate_g0  # noqa: E402
from agents.scientist.researcher.phase_d_client import load_model_stack  # noqa: E402
from agents.scientist.schemas.equivalence import equivalence_key  # noqa: E402

MAPPING_RULE = (
    "sources._first_config (spec §8.1): first reachable (template, variable) in canonical "
    "order, first lag, first form — zero per-mechanism discretion"
)
CONSTRAINTS = (
    "descriptive only — no BH-FDR family (raw p annotations only), no G4/G5/G6, "
    "no holdout nomination; exclusions and collisions logged with typed codes"
)
CACHE_ROOT = _REPO_ROOT / "runs" / "rq4_funnel" / "cache"


class CacheOnlyClient:
    """A ModelClient that can NEVER call a vendor: replay-only. LLMResearcherSource consults the
    cache first and calls generate() only on a miss — which here raises instead of calling out."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, *, seed: int) -> str:
        raise RuntimeError(
            f"cache miss for {self.name!r} seed {seed} — live calls are forbidden in the "
            "exhaustive benchmark (replay-only)"
        )


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _thresholds_sha256() -> str:
    return hashlib.sha256((_REPO_ROOT / "docs" / "thresholds.yaml").read_bytes()).hexdigest()


def phase_f_model_names() -> list[str]:
    """The reported generative pair's model ids, read from the live model-stack config
    (scientist.model_stack.phase_f) — the same block build_phase_f_clients consumes. No API
    key is needed to resolve names, and no run artifact is read."""
    pf = load_model_stack()["phase_f"]
    return [pf[k]["model_id"] for k in ("model_a", "model_b")]


def canonical_proposals(case, elig, library) -> list:
    """One canonical ExtensionProposal per eligible mechanism via the frozen §8.1 rule.
    Decode failures are impossible by construction — fail loud if one occurs."""
    proposals = []
    for r in elig:
        if not r.eligible:
            continue
        spec = F.S._first_config(r, library)
        raw = F.S._spec_to_raw(spec, case, library, source="exhaustive_canonical", seed=0,
                               model="none", prompt_version="v1", generated_at=F._GEN_AT)
        dec = F.S.decode_proposal(raw)
        if not dec.ok:
            raise RuntimeError(f"canonical proposal for {r.mechanism_id} failed decode: {dec.error}")
        proposals.append(dec.proposal)
    return proposals


def evaluate_batch(proposals, *, case, library, panel, base_rulebook, bbw4, macros,
                   available, reporting_delays, parent_mean_bp) -> list[dict]:
    """Mirror of the orchestrator's Phase A (G0→G1a→dedup→G1b→G2) plus per-candidate G3
    regression statistics. Deliberately NO run_fdr and NO G4/G5 — see module docstring."""
    rows: list[dict] = []
    seen: set = set()
    seen_transforms: set = set()
    for p in proposals:
        row = {
            "proposal_id": p.proposal_id,
            "mechanism": p.mechanism_ref,
            "template": p.template_ref,
            "conditioning_variable": p.config_delta.conditioning_variable,
            "conditioning_lag_months": p.config_delta.conditioning_lag_months,
            "interaction_form": p.config_delta.interaction_form,
        }
        g0 = validate_g0(p, case, library, seen_keys=seen, available_variables=available)
        if not g0.passed:
            row.update(status="refused", refusal_code=g0.refusal_code.value, gate="G0")
            rows.append(row)
            continue
        seen.add(equivalence_key(p))
        template = library.templates[p.template_ref]
        g1a, compiled = compile_g1a(p, template, case, holding_period=1,
                                    available_variables=available)
        if not g1a.passed:
            row.update(status="refused", refusal_code=g1a.refusal_code.value, gate="G1a")
            rows.append(row)
            continue
        sig = _transform_signature(compiled)
        if sig in seen_transforms:
            row.update(status="duplicate", refusal_code="DUPLICATE_PROPOSAL", gate="dedup")
            rows.append(row)
            continue
        seen_transforms.add(sig)
        macro_series = None
        if compiled.template_mode == "month_filter" and compiled.panel_transform is not None:
            macro_series = (macros or {}).get(compiled.panel_transform.variable)
        g1b, exec_res = execute_g1b(compiled, panel, base_rulebook, macro=macro_series)
        if not g1b.passed:
            row.update(status="refused", refusal_code=g1b.refusal_code.value, gate="G1b")
            rows.append(row)
            continue
        g2 = audit_g2(compiled, reporting_delays=reporting_delays)
        if not g2.passed:
            row.update(status="refused", refusal_code=g2.refusal_code.value, gate="G2")
            rows.append(row)
            continue
        cand = exec_res.candidate_returns
        reg = regress_on_benchmark(cand, bbw4, None)
        mean_bp = float(cand.mean()) * 1e4
        row.update(
            status="evaluated",
            n_months=int(len(cand)),
            mean_bp_per_month=round(mean_bp, 2),
            delta_vs_corrected_parent_bp=round(mean_bp - parent_mean_bp, 2),
            alpha_bp_per_month=round(float(reg["alpha"]) * 1e4, 2),
            alpha_t=round(float(reg["alpha_t"]), 3),
            p_raw_descriptive=round(float(two_sided_p(reg["alpha_t"])), 4),
            nw_lags_used=int(reg["nw_lags_used"]),
        )
        rows.append(row)
    return rows


def rank_within(canonical_rows: list[dict], alpha_t: float) -> int:
    """1-based rank of an alpha_t inside the canonical evaluated distribution (higher t = better,
    direction +1; ties rank equal-best)."""
    better = sum(1 for r in canonical_rows
                 if r["status"] == "evaluated" and r["alpha_t"] > alpha_t)
    return better + 1


def main() -> None:
    case, params = F.build_case()
    if not case.failed_check_ids:
        print("str did not enter — no correction cleared entry; nothing to benchmark")
        sys.exit(0)

    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=1,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]
    n_eligible = sum(1 for r in elig if r.eligible)

    panel, base_rulebook, parent_returns, direction = F.corrected_str_parent()
    parent_mean_bp = float(parent_returns.mean()) * 1e4
    macros = F.load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = F.bbw4_frame()
    reporting_delays = load_reporting_delays()

    kw = dict(case=case, library=library, panel=panel, base_rulebook=base_rulebook, bbw4=bbw4,
              macros=macros, available=available, reporting_delays=reporting_delays,
              parent_mean_bp=parent_mean_bp)

    # --- the canonical arm: one frozen-rule implementation per eligible mechanism ----------
    canon = canonical_proposals(case, elig, library)
    print(f"canonical arm: {len(canon)} proposals over {n_eligible} eligible mechanisms")
    canonical_rows = evaluate_batch(canon, **kw)
    evaluated = [r for r in canonical_rows if r["status"] == "evaluated"]
    excluded = [r for r in canonical_rows if r["status"] != "evaluated"]
    for r in canonical_rows:
        if r["status"] == "evaluated":
            r["rank_in_canonical"] = rank_within(canonical_rows, r["alpha_t"])

    # --- funnel-source arms, seed 0, replayed for ranks (cache-only for the LLMs) ----------
    embed, retrieval_label, _reportable, embedder_header = F.resolve_embedder("minilm", None)

    source_specs = [
        ("random_eligible", F.S.RandomEligibleSource(), "none"),
        ("retrieval_only", F.S.RetrievalOnlySource(embed), retrieval_label),
    ] + [
        (f"llm_{n}", F.LLMResearcherSource(CacheOnlyClient(n), cache=F.ResponseCache(CACHE_ROOT)), n)
        for n in phase_f_model_names()
    ]

    source_reports: dict[str, dict] = {}
    for src_name, source, model_name in source_specs:
        gen = F.S.run_generation(source, case, elig, library, seed=0, m=6, model=model_name,
                                 prompt_version="v1", generated_at=F._GEN_AT)
        rows = evaluate_batch(list(gen.proposal_set.proposals), **kw)
        for r in rows:
            if r["status"] == "evaluated":
                r["rank_in_canonical"] = rank_within(canonical_rows, r["alpha_t"])
        source_reports[src_name] = {
            "n_requested": gen.n_requested, "n_invalid": gen.n_invalid,
            "n_duplicate_at_generation": gen.n_duplicate, "rows": rows,
        }
        print(f"{src_name}: {len(rows)} proposals, "
              f"{sum(1 for r in rows if r['status'] == 'evaluated')} evaluated")

    report = {
        "run_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "thresholds_sha256": _thresholds_sha256(),
        "design": "exhaustive canonical-mechanism benchmark — descriptive diagnostic ranking "
                  "each proposal source inside the full eligible-mechanism distribution",
        "mapping_rule": MAPPING_RULE,
        "constraints": CONSTRAINTS,
        "window": "development 2002-2021 (holdout untouched)",
        "parent": {"strategy": "str", "corrected_parent_mean_bp_per_month": round(parent_mean_bp, 2),
                   "direction": direction},
        "eligible_mechanisms": n_eligible,
        "embedder": embedder_header,
        "generative_models_from": "scientist.model_stack.phase_f (docs/thresholds.yaml)",
        "canonical": {
            "n_proposals": len(canon),
            "n_evaluated_distinct": len(evaluated),
            "n_excluded": len(excluded),
            "exclusion_codes": sorted({r["refusal_code"] for r in excluded}) if excluded else [],
            "rows": canonical_rows,
        },
        "sources": source_reports,
    }

    out_json = _REPO_ROOT / "results" / "scientist" / "rq4_exhaustive_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written: {out_json}")

    # --- md summary ------------------------------------------------------------------------
    def _fmt(r: dict) -> str:
        if r["status"] != "evaluated":
            return (f"| {r['mechanism']} | {r['template']} | {r['conditioning_variable']} "
                    f"| — | — | — | — | {r['refusal_code']} ({r['gate']}) |")
        return (f"| {r['mechanism']} | {r['template']} | {r['conditioning_variable']} "
                f"| {r['mean_bp_per_month']:+.1f} | {r['alpha_bp_per_month']:+.1f} "
                f"| {r['alpha_t']:+.2f} | {r['rank_in_canonical']} | evaluated |")

    lines = [
        "# RQ4 exhaustive canonical-mechanism benchmark",
        "",
        f"**Mapping rule:** {MAPPING_RULE}. The rule pre-dates every funnel performance run;",
        "this driver post-dates them and exercises the rule verbatim, so no per-mechanism",
        "choice is made after results were observed. **Constraints:** " + CONSTRAINTS + ".",
        "",
        f"Corrected parent (str, all-ON): {parent_mean_bp:+.1f} bp/mo, direction {direction:+d}. "
        f"Eligible mechanisms: {n_eligible}; canonical proposals: {len(canon)}; evaluated distinct: "
        f"{len(evaluated)}; excluded/collided: {len(excluded)}.",
        "",
        "## Canonical distribution (rank by BBW-4 alpha t, descriptive)",
        "",
        "| mechanism | template | variable | mean bp/mo | alpha bp/mo | alpha t | rank | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [_fmt(r) for r in sorted(canonical_rows,
                                      key=lambda r: r.get("rank_in_canonical", 99))]
    for src_name, rep in source_reports.items():
        lines += ["", f"## {src_name} (seed 0 replay) — rank within the canonical distribution", "",
                  "| mechanism | template | variable | mean bp/mo | alpha bp/mo | alpha t | rank | status |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [_fmt(r) for r in rep["rows"]]
    lines += ["", "Raw p-values live in the JSON as descriptive annotations only; no BH family was",
              "formed and nothing here can nominate a holdout candidate.", ""]
    out_md = _REPO_ROOT / "results" / "rq4_exhaustive_benchmark.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out_md}")


if __name__ == "__main__":
    main()
