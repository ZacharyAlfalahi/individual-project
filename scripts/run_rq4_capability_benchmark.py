"""RQ4 Scientist CAPABILITY BENCHMARK (SC-SCI-16) — the post-hoc second layer.

The registered RQ4 design joins two questions: (1) does the pipeline produce parents deserving
extension, and (2) conditional on a suitable parent, does the Scientist produce quality
extensions? When a single parent is eligible, every funnel proposal shares that parent, so
"the Scientist is weak" and "the Scientist is barely exercised" are indistinguishable.

This driver runs the UNCHANGED generation + gate stack over the audit-set anchors that
do not clear entry (by default drf and mom6), at their fully corrected (all-ON) lattice
cells. Exactly one pre-registered gate is relaxed — entry — and the bypass is recorded per
parent (entered_would_be, entry_bypassed). Every other registered parameter holds at its
registered value: m = 6 (per_strategy, fixed ex ante for all three anchors), k = 5, q = 0.10,
BBW4 inference, realised-parent-sign direction, compiled-transform dedup, cap = null, the
generative wall. See docs/scientist_protocol.yaml SC-SCI-16 (2026-09-06).

STATUS: descriptive proposal-quality diagnostic. No confirmatory selection claim attaches to
a non-entering parent; the HOLDOUT STAYS SHUT (no rehearsal, no one-shot holdout, dev window only). The
per-parent EXHAUSTIVE CEILING (scripts/run_rq4_exhaustive_benchmark.py, generalised) is computed
beside the funnel so each parent's outcome is read against what the mechanism library could
possibly achieve on that parent.

ARMS: random_eligible + retrieval_only. Generative-LLM arms are DEFERRED, not silently
dropped: the Phase-F response cache is parent-specific (str prompts only), so new parents
need new live vendor calls gated on cost authorization — recorded in the artifact.

DEV / HOLDOUT DISCIPLINE: every read is under data/development/; the corrected-parent
builder raises if any parent month reaches 2022-01. bbw4/crowding/macros are the funnel's
own loaders (dev-window).

INPUTS (defaults = the recorded run):
  --basis {total_return,clean}  maximal panel via scripts/basis_inputs.load_basis_inputs; audit dir,
                                factors dir and output dir default to
                                results/consistent_basis/<basis>/{audit/full_run,
                                factors, rq4/capability}.
  --audit-dir DIR               holds <anchor>_report.json
                                (default results/auditor/<anchor>_corrected/).
  --factors-dir DIR             BBW-4 + crowding bundles by basename (thresholds.yaml untouched).
  --out DIR                     default results/scientist/rq4_capability.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_rq4_exhaustive_benchmark as B  # noqa: E402
import run_rq4_funnel as F  # noqa: E402
from agents.scientist.experimentalist.audit_checks import load_reporting_delays  # noqa: E402
from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.reporting.funnels import agent_quality_funnel, economic_funnel  # noqa: E402
from agents.scientist.schemas.outcomes import REFUSAL_CODES  # noqa: E402
from shared.evaluation.crowding import load_crowding_factor_bundle  # noqa: E402

_GEN_AT = "2026-09-06T00:00:00Z"


def _audit_report(anchor_id: str) -> Path:
    """Each corrected anchor has its own audit run dir (str_corrected,
    drf_corrected, mom6_corrected); the report is {anchor}_corrected/{anchor}_report.json."""
    return _REPO_ROOT / "results" / "auditor" / f"{anchor_id}_corrected" / f"{anchor_id}_report.json"


_DEFAULT_OUT = _REPO_ROOT / "results" / "scientist" / "rq4_capability"

# Per-parent constants. holding_period is a TRIPWIRE (asserted against the anchor's
# QuantConfig at run time, never trusted); signal_lookback feeds the CPCV information
# span (orchestrator._info_span) and is derived from each anchor's gold spec:
#   str  — prior 1-month excess return (gold_str_drr_2026.md)                    -> 1
#   drf  — VaR_5% from a 36-month trailing window (build_var_5pct.py, window=36) -> 36
#   mom6 — 6 formation months + 1 skip month (gold_mom6_jnps_2013.md, p.9)       -> 7
_ANCHORS = {
    "str": {"holding_period": 1, "signal_lookback": 1},
    "drf": {"holding_period": 1, "signal_lookback": 36},
    "mom6": {"holding_period": 6, "signal_lookback": 7},
}


def resolve_capability_paths(basis: str | None = None, *, audit_dir=None, factors_dir=None,
                             out=None) -> dict:
    """explicit flag > basis default (.../<basis>/{audit/full_run, factors, rq4/capability}) >
    recorded."""
    return F.resolve_basis_paths(basis, audit=audit_dir, factors_dir=factors_dir, out=out,
                                 default_audit=None, default_out=_DEFAULT_OUT,
                                 out_leaf="capability")


def corrected_parent(anchor_id: str, basis: str | None = None):
    """(panel, base_rulebook, parent_returns, direction, holding_period) for the anchor's
    auditor-EXACT all-ON (fully corrected) lattice cell — F.corrected_str_parent generalised.
    Engine dispatch mirrors agents/quant/config/runner.py: holding 1 -> run_characteristic_sort,
    > 1 -> overlap.run_with_holding_period. The run_cell self-verify (zero tolerance) is the
    load-bearing check that the experimentalist seam reproduces the auditor cell for EVERY
    parent — if an anchor violates the single-leg reduction this fails loud, never silently
    grades against a wrong parent. `basis` None = load_dev_inputs() (recorded); a basis name =
    basis_inputs.load_basis_inputs(basis)."""
    from agents.auditor.checks.cell_runner import _override_construction, run_cell
    from agents.auditor.checks.lattice import build_lattice_configs
    from agents.auditor.checks.preflight import derive_scope
    from agents.auditor.ipca_differential.runner import load_dev_inputs
    from agents.quant.config.quant_config import to_rulebook
    from agents.quant.library.characteristic_sort import run_characteristic_sort
    from agents.quant.library.overlap import run_with_holding_period
    from agents.quant.library.views import view
    from scripts.run_auditor import (
        default_anchor_facts,
        load_anchor_expost_trim_off,
        load_anchor_meas_err_off_family,
        load_anchor_strategy,
    )

    if basis is None:
        maximal, signals, _registry = load_dev_inputs()
    else:
        maximal, signals, _registry = F.BI.load_basis_inputs(basis)
    strategy = load_anchor_strategy(anchor_id)
    facts = default_anchor_facts(anchor_id)
    meas_err_off_family = load_anchor_meas_err_off_family(anchor_id)
    expost_trim_off = load_anchor_expost_trim_off(anchor_id, None)

    pf = derive_scope(strategy.strategy_label, facts)
    configs = build_lattice_configs(
        pf.runnable_toggles, pf.fixed_states, lib_gap_lags=(0, 1),
        not_applicable_toggles=pf.not_applicable_toggles, meas_err_off_family=meas_err_off_family)
    all_on = frozenset(pf.runnable_toggles)
    rc_all_on = next(rc for on_set, rc in configs if on_set == all_on)

    panel = view(maximal, rc_all_on, signals=signals)
    overridden = _override_construction(strategy, rc_all_on, expost_trim_off)
    if len(overridden.leg_calls) != 1:
        raise RuntimeError(f"{anchor_id}: expected a single-leg anchor, "
                           f"got {len(overridden.leg_calls)} legs — seam reduction invalid")
    qc = overridden.leg_calls[0].result
    holding_period = int(qc.holding_period.value)
    if holding_period != _ANCHORS[anchor_id]["holding_period"]:
        raise RuntimeError(f"{anchor_id}: config holding_period={holding_period} != registered "
                           f"constant {_ANCHORS[anchor_id]['holding_period']}")
    base_rulebook = to_rulebook(qc)

    if holding_period == 1:
        mr = run_characteristic_sort(panel, base_rulebook)["monthly_returns"]
    else:
        mr = run_with_holding_period(panel, base_rulebook, holding_period)
    parent = pd.Series(mr["strategy_ret"].to_numpy(),
                       index=pd.DatetimeIndex(mr["date"].to_numpy())).sort_index()

    cell = run_cell(strategy, rc_all_on, all_on, panel, expost_trim_off=expost_trim_off)
    cell_ret = cell.returns.sort_index()
    assert parent.index.equals(cell_ret.index) and \
        np.allclose(parent.to_numpy(), cell_ret.to_numpy(), rtol=0, atol=0), \
        f"{anchor_id}: corrected parent diverges from the auditor all-ON cell"
    # DEV-WINDOW GUARD: the direction sign derives from this premium — dev reads only.
    if pd.Timestamp(parent.index.max()) >= pd.Timestamp("2022-01-01"):
        raise RuntimeError(f"{anchor_id}: corrected-parent premium reads beyond dev "
                           f"(max {parent.index.max()})")
    direction = 1 if float(parent.mean()) > 0 else -1
    return panel, base_rulebook, parent, direction, holding_period


def directional_ceiling(rows: list[dict], direction: int):
    """Direction-aware canonical ceiling + ranks. B.rank_within assumes direction +1 (the
    recorded str benchmark); a direction=-1 parent's improvement is NEGATIVE alpha, so the
    ceiling is max(direction * alpha_t), never raw max(alpha_t). Mutates evaluated rows'
    rank_in_canonical in place; returns the ceiling row (None if nothing evaluated)."""
    ev = [r for r in rows if r["status"] == "evaluated"]
    for r in ev:
        r["rank_in_canonical"] = 1 + sum(
            1 for o in ev if direction * o["alpha_t"] > direction * r["alpha_t"])
    return max(ev, key=lambda r: direction * r["alpha_t"]) if ev else None


def run_capability(anchor_id: str, *, embedder_mode: str = "minilm", k: int = 5, m: int = 6,
                   embedder=None, basis: str | None = None, audit_dir=None,
                   factors_dir=None) -> dict:
    """One parent through generation (random + retrieval) -> G0-G5 -> ceiling arm.
    Entry is evaluated honestly and then bypassed (recorded); nothing advances to holdout.
    `basis` / `audit_dir` / `factors_dir` select the inputs (None = the recorded run)."""
    meta = _ANCHORS[anchor_id]
    recorded = F.is_recorded_inputs(basis, audit_dir, factors_dir)
    paths = resolve_capability_paths(basis, audit_dir=audit_dir, factors_dir=factors_dir)
    report_path = (_audit_report(anchor_id) if paths["audit"] is None
                   else paths["audit"] / f"{anchor_id}_report.json")
    run_ref = f"{anchor_id}_corrected" if recorded else F.repo_relative(report_path.parent)
    case, params = F.build_case(
        report_path, strategy_id=anchor_id,
        case_id=f"rq4cap_{anchor_id}",
        corrected_quant_config_ref=f"qc_{anchor_id}_corrected",
        corrected_run_ref=run_ref)
    entered_would_be = bool(case.failed_check_ids)

    panel, base_rulebook, parent_returns, direction, holding_period = corrected_parent(
        anchor_id, basis)
    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=holding_period,
                       templates=library.templates, variable_families=library.variable_families,
                       available_variables=available) for mm in library.mechanisms]
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
    generation: dict[str, list] = {}
    for src_name, source, model_name in source_specs:
        generation[src_name] = F.S.run_all_seeds(
            source, case, elig, library, k=k, m=m, model=model_name,
            prompt_version="v1", generated_at=_GEN_AT)

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

    # Per-arm reportability stamp (mirrors run_rq4_funnel): random_eligible is fully
    # offline and reportable; retrieval_only is reportable only with the production embedder.
    if "random_eligible" in reports:
        reports["random_eligible"]["reportable"] = True
    if "retrieval_only" in reports:
        reports["retrieval_only"]["reportable"] = bool(retrieval_reportable)
        if retrieval_reportable and embedder_header is not None:
            reports["retrieval_only"]["embedder_note"] = embedder_header["note"]

    # --- exhaustive ceiling: one canonical implementation per eligible mechanism ------------
    canon = B.canonical_proposals(case, elig, library)
    canonical_rows = B.evaluate_batch(
        canon, case=case, library=library, panel=panel, base_rulebook=base_rulebook, bbw4=bbw4,
        macros=macros, available=available, reporting_delays=reporting_delays,
        parent_mean_bp=float(parent_returns.mean()) * 1e4, holding_period=holding_period)
    ceiling = directional_ceiling(canonical_rows, direction)

    result = {
        "amendment": "SC-SCI-16", "generated_at": _GEN_AT,
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
        "generative_arms": "DEFERRED — Phase-F cache is str-specific; live calls need cost "
                           "authorization (SC-SCI-16)",
        "reports": reports,
        "canonical_ceiling": ceiling,
        "canonical_rows": canonical_rows,
        "holdout": "NOT OPENED — capability parents never advance (SC-SCI-16); dev window only",
    }
    if not recorded:
        result["inputs"] = F.inputs_record(basis, report_path, paths["factors_dir"])
    return result


def output_path(out_dir: Path, anchor_id: str, embedder_mode: str) -> Path:
    return out_dir / f"rq4_capability_{anchor_id}_{embedder_mode}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", nargs="+", default=["drf", "mom6"],
                    choices=sorted(_ANCHORS),
                    help="capability parents (default: drf and mom6; str is "
                         "accepted only as a cross-check against the recorded funnel)")
    ap.add_argument("--embedder", choices=("offline", "minilm"), default="minilm",
                    help="minilm = registered production embedder (retrieval arm reportable); "
                         "offline = deterministic stub (smoke only, NON-reportable)")
    ap.add_argument("--basis", choices=F.BI.BASES, default=None,
                    help="return basis of the maximal panel (default: recorded clean run)")
    ap.add_argument("--audit-dir", default=None,
                    help="dir holding <anchor>_report.json (default results/auditor/"
                         "<anchor>_corrected/; with --basis: .../<basis>/audit/full_run)")
    ap.add_argument("--factors-dir", default=None,
                    help="dir with bbw_factors/mktb/str/mom6 parquets (default data/development/"
                         "factors; with --basis: .../<basis>/factors)")
    ap.add_argument("--out", default=None,
                    help="output dir (default results/scientist/rq4_capability; with --basis: "
                         "results/consistent_basis/<basis>/rq4/capability)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    paths = resolve_capability_paths(args.basis, audit_dir=args.audit_dir,
                                     factors_dir=args.factors_dir, out=args.out)

    out_dir = paths["out"]
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "amendment": "SC-SCI-16",
        "run_timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "embedder_mode": args.embedder,
        "thresholds_sha256": B._thresholds_sha256(),
    }
    if args.embedder == "minilm":
        header["embedder"] = F.minilm_header()

    for anchor_id in args.anchors:
        print(f"=== capability parent: {anchor_id} ===")
        result = run_capability(anchor_id, embedder_mode=args.embedder, basis=args.basis,
                                audit_dir=args.audit_dir, factors_dir=args.factors_dir)
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
        c = result["canonical_ceiling"]
        if c is not None:
            print(f"  ceiling: {c['mechanism']} alpha_t={c['alpha_t']} "
                  f"alpha_bp={c['alpha_bp_per_month']} p_raw={c['p_raw_descriptive']}")
        else:
            print("  ceiling: no canonical proposal evaluated")
        print(f"  written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
