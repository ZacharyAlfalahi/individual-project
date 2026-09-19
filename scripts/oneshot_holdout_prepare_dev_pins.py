"""DEVELOPMENT-ONLY preparation for the one-shot holdout real run (run BEFORE the holdout is opened; never reads it).

1. Re-runs the clean-price RQ4 funnel's experimentalist stage per source (deterministic: random / retrieval /
   cached LLM outputs) and asserts the advanced ids equal the funnel artefact's — the survivor set is the
   funnel's, not this script's.
2. P3 moderate prior (SC-SCI-11 registered formula): the sample SD of the development-window G3 extension alphas
   across the confirmatory (audit-clean, BH-family) proposals, logged before holdout access.
3. Development pins for the stage-2 inputs: the corrected parent's dev mean, each survivor's dev mean and the
   BBW-4 column means, computed through ``oneshot_holdout_survivor_inputs`` on the development frames — and cross-checked
   against the experimentalist's own G3 measurements, so the deriver is proven to reproduce the funnel's
   execution path before it is pointed at the seeded holdout panel.
Writes <out>/p3_moderate_prior.json and <out>/oneshot_dev_pins.json.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import oneshot_holdout_survivor_inputs as SI  # noqa: E402
import run_rq4_funnel as F  # noqa: E402
from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.researcher import sources as S  # noqa: E402
from agents.scientist.researcher.cache import ResponseCache  # noqa: E402
from scripts import basis_inputs as BI  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

ROOT = BI.RESULTS_ROOT


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", default="clean", choices=BI.BASES)
    ap.add_argument("--audit-report", type=Path, default=ROOT / "clean" / "audit" / "full_run" / "str_report.json")
    ap.add_argument("--funnel-artefact", type=Path, default=None)
    ap.add_argument("--factors-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "clean" / "rq4" / "oneshot_holdout")
    args = ap.parse_args(argv)
    funnel = args.funnel_artefact or (ROOT / args.basis / "rq4" / "funnel" / "rq4_funnel_reported_minilm.json")
    factors_dir = args.factors_dir or BI.factors_dir(args.basis)
    args.out.mkdir(parents=True, exist_ok=True)

    # --- the funnel's exact development context -------------------------------------------------------
    case, params = F.build_case(args.audit_report, corrected_run_ref=str(args.audit_report.parent))
    library = F.load_library()
    available = F.available_conditioning_variables()
    elig = [F.evaluate(mm, strategy_family=F._STRATEGY_FAMILY, holding_period=1, templates=library.templates,
                       variable_families=library.variable_families, available_variables=available) for mm in library.mechanisms]
    k, m = 5, 6
    clients, budget = F.prepare_budgeted_clients(
        [F.planned_prompt(case, elig, library, m)], stack_key="phase_f", k=k, cache_root=F._CACHE_ROOT,
        max_usd={"anthropic": 0.0, "gemini": 0.0}, client_factory=lambda models: F.build_live_pair("reported", models))
    assert budget["decision"] == "replay_only", budget
    panel, base_rulebook, parent_returns, direction = F.corrected_str_parent(args.basis)
    macros = F.load_macro_series(panel["date"].min(), panel["date"].max())
    bbw4 = F.bbw4_frame(factors_dir)
    crowding_cfg = F.crowding_config_for(factors_dir)
    crowding_factors = F.load_crowding_factor_bundle(crowding_cfg)
    reporting_delays = F.load_reporting_delays()
    embed, label, _r, _h = F.resolve_embedder("minilm", None)
    specs = [("random_eligible", S.RandomEligibleSource(), "random_eligible"),
             ("retrieval_only", S.RetrievalOnlySource(embed), label)]
    for c in clients:
        specs.append((f"llm_{c.name}", F.LLMResearcherSource(c, cache=ResponseCache(F._CACHE_ROOT)), c.name))

    doc = json.loads(require_licensed_input(
        funnel, "RQ4 funnel artefact (local pipeline output, not shipped with the repository)").read_text())
    alphas, per_source, g3_by_id = [], {}, {}
    for src_name, source, model in specs:
        gen = S.run_all_seeds(source, case, elig, library, k=k, m=m, model=model, prompt_version="v1", generated_at=F._GEN_AT)
        rep = run_experimentalist(case, gen[0].proposal_set.proposals, library, panel=panel, base_rulebook=base_rulebook,
                                  bbw4_factors=bbw4, holding_period=1, signal_lookback=1, available_variables=available,
                                  macros=macros, m=m, q=params.q, cap=None, direction=direction,
                                  crowding_config=crowding_cfg, crowding_factors=crowding_factors,
                                  reporting_delays=reporting_delays)
        if sorted(rep.advanced) != sorted(doc["reports"][src_name]["advanced"]):
            raise RuntimeError(f"{src_name}: advanced {sorted(rep.advanced)} != funnel artefact {doc['reports'][src_name]['advanced']}")
        src_alphas = []
        for r in rep.records:
            g = r.measurements.gross
            if g is not None and g.alpha_bbw4 is not None:
                src_alphas.append(float(g.alpha_bbw4))
                g3_by_id[f"{src_name}:{r.proposal_id}"] = {"alpha_bbw4": float(g.alpha_bbw4), "mean_return": float(g.mean_return),
                                                          "t_stat": float(g.t_stat), "bh_rejected": g.bh_rejected}
        per_source[src_name] = {"n_g3": len(src_alphas), "sd_alpha": (statistics.stdev(src_alphas) if len(src_alphas) > 1 else None),
                                "advanced": list(rep.advanced)}
        alphas += src_alphas
    sigma = statistics.stdev(alphas)
    p3 = {"amendment": "SC-SCI-11", "prior": "moderate", "formula": ("sample SD of the development-window G3 extension alphas "
          "(monthly alpha vs BBW-4) across the confirmatory audit-clean proposal family of the clean-price funnel, "
          "all sources, seed 0; logged before holdout access"), "sigma_moderate": sigma, "n_alphas": len(alphas),
          "alphas": alphas, "per_source": per_source, "fallback_sigma_not_used": 0.005,
          "sigma_wide": 0.02, "sigma_sceptical": 0.0025, "computed_at": datetime.now(timezone.utc).isoformat(),
          "inputs": {"audit_report": str(args.audit_report.relative_to(REPO_ROOT)), "audit_sha256": SI.sha256(args.audit_report),
                     "funnel_artefact": str(funnel.relative_to(REPO_ROOT)), "funnel_sha256": SI.sha256(funnel),
                     "factors_dir": str(Path(factors_dir).relative_to(REPO_ROOT)), "basis": args.basis}}
    (args.out / "p3_moderate_prior.json").write_text(json.dumps(p3, indent=2))

    # --- the deriver on the development frames must reproduce the experimentalist's own execution --------
    maximal, signals, _reg = BI.load_basis_inputs(args.basis)
    import pandas as pd
    seeded = {"maximal": maximal, "signals": signals, "macros": macros,
              "factors_bbw": pd.read_parquet(require_licensed_input(
                  Path(factors_dir) / "bbw_factors.parquet",
                  "BBW factor panel (local pipeline output, not shipped with the repository)")),
              "factors_mktb": pd.read_parquet(require_licensed_input(
                  Path(factors_dir) / "mktb.parquet",
                  "market-beta factor (local pipeline output, not shipped with the repository)"))}
    inputs, benchmarks, record = SI.derive_stage2_inputs(seeded, audit_report=args.audit_report, funnel_artefact=funnel)
    parent = inputs[0].parent_returns
    if not (parent.index.equals(parent_returns.index) and np.array_equal(parent.to_numpy(), parent_returns.to_numpy())):
        raise RuntimeError("deriver parent != funnel corrected_str_parent")
    survivors_out = []
    for si in inputs:
        g3 = g3_by_id[si.survivor_id]
        got = float(si.returns.mean())
        if not abs(got - g3["mean_return"]) <= 1e-12:
            raise RuntimeError(f"{si.survivor_id}: deriver dev mean {got!r} != experimentalist G3 mean_return {g3['mean_return']!r}")
        survivors_out.append({"survivor_id": si.survivor_id, "dev_mean": got, "n_months_dev": int(len(si.returns)), "g3": g3})
    pins = {"basis": args.basis, "parent_dev_mean": float(parent.mean()), "n_months_dev_parent": int(len(parent)),
            "survivors": survivors_out, "survivor_specs": record["survivors"],
            "bbw4_dev_means": {c: float(benchmarks["bbw4"][c].mean()) for c in ("mktb", "drf", "crf", "lrf")},
            "computed_at": datetime.now(timezone.utc).isoformat(), "inputs": p3["inputs"],
            "note": "every value is a development-window quantity; the seeded holdout build's development portion must reproduce each (atol 1e-9) before stage 2 runs"}
    (args.out / "oneshot_dev_pins.json").write_text(json.dumps(pins, indent=2))
    print(json.dumps({"sigma_moderate": sigma, "n_alphas": len(alphas), "per_source": per_source,
                      "survivors": [(s["survivor_id"], s["dev_mean"]) for s in survivors_out]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
