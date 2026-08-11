"""
Phase 3 — real bond-centric IPCA shakedown (INTERFACE-VALIDATION ONLY).

Loads the Workstream B feed (data/development/ipca_panel_<family>.parquet), runs the as-built,
certified ipca module end-to-end on real development-window data (2002–2021), and emits a run-log
+ context-table. Its PURPOSE is to exercise the module ↔ panel ↔ wall interface, complete-case
selection, family-indexing, and runtime — NOT to produce findings.

NON-COMPARABLE TO KPP: a 7-instrument bond-only, VOL-scaled model is structurally different from
KPP's 29-instrument DtS model. It CANNOT speak to RQ1/RQ2/RQ3. Every artifact carries the
comparability stamp. KPP's published VOL-lane numbers (Table CI-B OOS total R² 54.4% individual;
Table CII vol-scaled tangency 4.94 gross) are the FULL model's and are NOT a target.
See docs/quant/specs/characteristic_registry_spec.md §7.

Usage:  python scripts/run_ipca_shakedown.py [--family corr|raw] [--k-recursive 4]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from agents.quant.library.ipca import (  # noqa: E402
    IPCAConfig,
    assert_within_wall,
    build_run_log,
    build_sufficient_stats,
    context_table,
    cross_section_r2,
    diagnose_scaling_lane,
    fit_ipca,
    fit_ipca_recursive,
    oos_total_r2,
    smoothing_cost_curve,
    spread_strategy,
    tangency_insample,
    tangency_strategy,
    total_r2,
    validate_panel,
)
from agents.quant.library.ipca_feed import INSTRUMENTS, L, load_feed  # noqa: E402

DEV = REPO_ROOT / "data" / "development"
HEADLINES = DEV / "headlines"
GOLD = REPO_ROOT / "agents" / "quant" / "library" / "configs" / "kpp_ipca.yaml"
INSTR = INSTRUMENTS  # the 7 bond instruments; definition + load_feed lifted to agents/quant/library/ipca_feed.py
TRAIN_END = pd.Period("2021-12", "M").ordinal   # inclusive: last in-window return month
COMPARABILITY = (
    "NON_COMPARABLE_TO_KPP — interface-validation only "
    "(7-instrument bond-only, no equity, VOL-scaled not DtS)"
)


# load_feed lifted to agents/quant/library/ipca_feed.py (imported above).


def shakedown_config(k_recursive: int) -> IPCAConfig:
    """Gold spec with the shakedown's actual lane/L/K overridden, for a faithful run-log."""
    gold = yaml.safe_load(GOLD.read_text())
    gold["scaling"]["lane"] = "VOLScaled010"
    gold["data_contract"]["L"] = L
    gold["model"]["K"] = k_recursive
    return IPCAConfig.from_dict(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="corr", choices=["corr", "raw"])
    ap.add_argument("--k-recursive", type=int, default=4)
    args = ap.parse_args()
    if not 1 <= args.k_recursive <= 5:
        print(f"ERROR: --k-recursive must be in 1..5 (the in-sample sweep range); got {args.k_recursive}",
              file=sys.stderr)
        sys.exit(1)
    feed = DEV / f"ipca_panel_{args.family}.parquet"
    if not feed.exists():
        print(f"ERROR: feed not found: {feed}; run build_ipca_panel.py first", file=sys.stderr)
        sys.exit(1)

    print(f"Loading feed {feed.name} ...")
    Z, R, months, asof, vol_all = load_feed(feed)
    print(f"  T={len(months)} months, N_m median={int(np.median([len(r) for r in R]))}")
    validate_panel(Z, R, months, L=L, family=args.family, characteristic_asof=asof)
    assert_within_wall(months, train_end=TRAIN_END)
    stats = build_sufficient_stats(Z, R, months)

    print("In-sample K sweep (1..5) ...")
    insample = {}
    fit_for_log = None
    for k in range(1, 6):
        fit = fit_ipca(stats, K=k)
        insample[k] = {
            "total_r2": round(total_r2(stats, fit), 6),
            "cross_section_r2": round(cross_section_r2(Z, R, fit), 6),
            "n_iter": fit.n_iter,
            "converged": fit.converged,
        }
        if k == args.k_recursive:
            fit_for_log = fit
        print(f"  K={k}: total_R2={insample[k]['total_r2']:.4f} "
              f"cs_R2={insample[k]['cross_section_r2']:.4f} iters={fit.n_iter}")

    cfg = shakedown_config(args.k_recursive)
    assert fit_for_log is not None
    print(f"Recursive OOS (K={args.k_recursive}, burn_in=36) ...")
    rec = fit_ipca_recursive(stats, K=args.k_recursive, burn_in=36)
    tan = tangency_strategy(rec)
    spr = spread_strategy(Z, R, rec)
    insample_tan = tangency_insample(fit_for_log)                         # D1 (§4.1) plain in-sample tangency
    grid = [float(g) for g in cfg.strategies.smoothing_gamma_grid]
    assert tan.weights is not None
    curve = smoothing_cost_curve(tan.weights, rec.f_oos, grid)            # D2 (§4.3) net-of-cost γ-grid
    recursive = {
        "oos_months": int(rec.f_oos.shape[1]),
        "oos_total_r2": round(oos_total_r2(rec), 6),
        "tangency_sharpe": round(tan.sharpe, 4),
        "spread_sharpe": round(spr.sharpe, 4),
        "insample_tangency_sharpe": round(insample_tan.sharpe, 4),
        "smoothing_net_sharpe_by_gamma": {str(g): round(curve[g].sharpe, 4) for g in grid},
        "mean_n_iter": round(float(np.mean(rec.n_iters)), 1),
    }
    print(f"  OOS total_R2={recursive['oos_total_r2']:.4f} tangency_SR={recursive['tangency_sharpe']} "
          f"spread_SR={recursive['spread_sharpe']} in-sample tangency_SR={recursive['insample_tangency_sharpe']}")

    scaldiag = diagnose_scaling_lane("VOLScaled010", vol_values=vol_all)
    log = build_run_log(
        cfg, fits=[fit_for_log], train_end=TRAIN_END,
        window_bounds=(int(months.min()), int(months.max())),
        dropped_months=[], below_floor_shares={"bond_vol": scaldiag.get("below_floor_share", float("nan"))},
        lib_versions={"numpy": np.__version__, "pandas": pd.__version__},
        comparability_label=COMPARABILITY,
    )
    log.update({
        "purpose": "interface-validation (module<->panel<->wall, complete-case, family-index, runtime)",
        "rq_usability": "NONE — cannot be cited for RQ1/RQ2/RQ3",
        "instruments": INSTR,
        "scaling_lane": "VOLScaled010",
        "scaling_diagnostics": scaldiag,
        "in_sample_K_sweep": insample,
        "recursive": recursive,
        "kpp_context_values_full_model_NOT_a_target": {
            "table_CI_B_oos_total_r2_individual": 0.544,
            "table_CII_vol_scaled_tangency_gross": 4.94,
            "note": "KPP's FULL 29-instrument model — recorded for lineage only; NOT a target here.",
        },
    })

    HEADLINES.mkdir(parents=True, exist_ok=True)
    out_json = HEADLINES / "ipca_shakedown.json"
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(log, indent=2))
    os.replace(tmp, out_json)

    # Context table (markdown) — same banner on the artifact.
    lines = [
        f"# IPCA shakedown — {COMPARABILITY}",
        "",
        f"_Generated {datetime.now(timezone.utc).isoformat()}; family={args.family}; "
        f"T={len(months)} months; config_hash={cfg.hash()[:16]}._",
        "",
        "**Interface-validation only. Not a finding. Cannot be cited for RQ1/RQ2/RQ3.**",
        "",
        "## In-sample (managed-portfolio total R²; cross-section R²)",
        "",
        "| K | total R² | cross-section R² | iters | converged |",
        "|---|----------|------------------|-------|-----------|",
    ]
    for k in range(1, 6):
        s = insample[k]
        lines.append(f"| {k} | {s['total_r2']:.4f} | {s['cross_section_r2']:.4f} | "
                     f"{s['n_iter']} | {s['converged']} |")
    lines += [
        "",
        f"## Recursive OOS (K={args.k_recursive}, burn-in 36)",
        "",
        f"- OOS months: {recursive['oos_months']}",
        f"- OOS total R²: {recursive['oos_total_r2']:.4f}",
        f"- Tangency Sharpe (ann., OOS): {recursive['tangency_sharpe']}",
        f"- In-sample tangency Sharpe (plain, §4.1): {recursive['insample_tangency_sharpe']}",
        f"- Spread (Q5−Q1) Sharpe (ann.): {recursive['spread_sharpe']}",
        f"- Net-of-cost tangency Sharpe by smoothing γ: {recursive['smoothing_net_sharpe_by_gamma']}",
        "",
        "## §10.4 context-value comparison (module `context_table` harness)",
        "",
        context_table(
            {
                "oos_total_r2": recursive["oos_total_r2"],
                "tangency_sharpe_gross": recursive["tangency_sharpe"],
                "insample_tangency_sharpe": recursive["insample_tangency_sharpe"],
            },
            {"oos_total_r2": 0.544, "tangency_sharpe_gross": 4.94},   # KPP §10.4 placeholders (FULL model)
            comparability_label=COMPARABILITY,
        ),
        "_Context = KPP's FULL 29-instrument model (Table CI-B OOS total R² 54.4%; Table CII "
        "vol-scaled tangency 4.94 gross) — recorded for lineage only; NOT a target for this "
        "7-instrument run._",
    ]
    out_md = HEADLINES / "ipca_shakedown_context_table.md"
    mtmp = out_md.with_suffix(".tmp")
    mtmp.write_text("\n".join(lines) + "\n")
    os.replace(mtmp, out_md)

    print(f"\nDone. → {out_json}\n       → {out_md}")
    print(f"  STAMP: {COMPARABILITY}")


if __name__ == "__main__":
    main()
