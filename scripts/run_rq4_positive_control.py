#!/usr/bin/env python
"""RQ4 development-funnel POSITIVE CONTROL (a validation instrument — SYNTHETIC, never a finding).

Establishes that the funnel has DISCRIMINATING POWER: when a genuinely strong extension exists it
ADVANCES (BH-FDR survivor -> CPCV-qualified -> G5), and when only noise exists it is correctly
REJECTED. This makes the real-data result (0 / 24 advanced) a TRUE null rather than a dead pipeline.

The signal is PLANTED by construction (a known-answer test). Nothing here is a real strategy, reads
real data, or ever touches /data/holdout/ -- it runs entirely on a synthetic in-memory panel, is
deterministic (seeded), free ($0, no model calls), and is NEVER reported as an RQ4 survivor.

beta <-> bp/mo mapping (used by --sweep): the plant is `ret = beta * (score - 14.5)` in the IG
segment; the funnel longs the top score quintile (mean score 26.5) and shorts the bottom (mean 2.5),
so the EXPECTED planted long-short mean return is beta * 24 per month (48 bp/mo at the default
beta = 0.00020). The REALISED anchor on the historical default config (seed 20260906) is
mean_return = 0.0044 = 44 bp/mo (expectation minus that seed's noise draw), so the ladder maps
linearly off the realised anchor: beta(bp) = 0.00020 * bp / 44.

    ./.venv/bin/python scripts/run_rq4_positive_control.py     # writes results/scientist/rq4_positive_control.json
    ./.venv/bin/python scripts/run_rq4_positive_control.py --sweep   # power curve -> results/scientist/rq4_power_curve.{json,md}
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from agents.scientist.experimentalist.orchestrator import run_experimentalist  # noqa: E402
from agents.scientist.researcher.library import load_library  # noqa: E402
from agents.scientist.schemas.case import DevelopmentWindow, HoldoutStatus, ScientistCase  # noqa: E402
from agents.scientist.schemas.proposal import decode_proposal  # noqa: E402
from shared.evaluation.thresholds import CrowdingConfig  # noqa: E402

_FACTORS = ("mktb", "drf", "crf", "lrf", "str", "mom6")
_MONTHS = pd.date_range("2004-08-31", periods=120, freq="ME")   # long enough for CPCV (28 splits / 7 paths)
_N = 60                                                         # 30 IG (even idx) + 30 HY (odd idx)
_BETA = 0.00020     # planted per-score-unit monthly return in the investment-grade segment (default)
_SIGMA = 0.020      # idiosyncratic noise (default)
_SEED = 20260906    # panel RNG seed (default; matches the recorded run)

# Realised anchor for the beta <-> bp/mo mapping (see module docstring): the historical default
# config (_BETA, _SEED) realises a planted mean monthly extension return of 44 bp/mo.
_ANCHOR_BETA = 0.00020
_ANCHOR_BP = 44.0
# Default sweep ladder in planted bp/mo (approximate realised mean monthly extension returns).
_DEFAULT_LADDER_BP = (5.0, 10.0, 15.0, 20.0, 30.0, 44.0)


def beta_for_bp(bp_per_month: float) -> float:
    """Per-score-unit beta whose planted mean monthly extension return is ~`bp_per_month` bp/mo.

    Linear in the realised anchor _ANCHOR_BETA = 0.00020 <=> _ANCHOR_BP = 44 bp/mo (module
    docstring): beta = 0.00020 * bp / 44."""
    return _ANCHOR_BETA * (bp_per_month / _ANCHOR_BP)


def _panel(beta: float = _BETA, sigma: float = _SIGMA, seed: int = _SEED) -> pd.DataFrame:
    """The `score` signal genuinely predicts returns IN the investment-grade segment (planted), and
    is pure NOISE in the high-yield segment -- so restrict_investment_grade isolates a real strong
    signal and restrict_high_yield isolates noise. Deterministic (seeded)."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(_N):
        ig = (b % 2 == 0)
        score = b // 2                     # both classes span scores 0..29
        for d in _MONTHS:
            noise = rng.normal(0.0, sigma)
            ret = beta * (score - 14.5) + noise if ig else noise
            rows.append(dict(cusip=f"B{b:02d}", date=d, ret=ret, size=100.0, score=float(score),
                             investment_grade=ig, rating=float(score), gamma_illiq=float(_N - b)))
    return pd.DataFrame(rows)


def _bbw4() -> pd.DataFrame:
    # BBW-4 factors uncorrelated with the planted signal -> the planted return is real ALPHA.
    rng = np.random.default_rng(7)
    return pd.DataFrame({"date": _MONTHS, **{f: rng.normal(scale=0.01, size=len(_MONTHS))
                                             for f in ("mktb", "drf", "crf", "lrf")}})


def _crowding_factors() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    return pd.DataFrame({"date": _MONTHS, **{f: rng.normal(scale=0.02, size=len(_MONTHS))
                                             for f in _FACTORS}})


def _cfg() -> CrowdingConfig:
    return CrowdingConfig(factor_set=_FACTORS, hac_lag_rule="newey_west_auto", min_obs=10,
                          bundles={f: (f"{f}.parquet", f"{f}_corr") for f in _FACTORS})


def _proposal(pid: str, form: str):
    raw = {
        "proposal_id": pid, "case_id": "case_pc", "parent_strategy_id": "str",
        "mechanism_ref": "mech_009", "template_ref": "ex_ante_universe_conditioning_v1",
        "rationale": "positive-control synthetic instrument (not a real proposal)",
        "prediction": "planted-by-construction",
        "config_delta": {"conditioning_variable": "rating", "conditioning_lag_months": 1,
                         "interaction_form": form},
        "required_inputs": ["rating"],
        "generation": {"source": "random_eligible", "seed": 0, "model": "synthetic",
                       "prompt_version": "v0", "library_version": "x",
                       "generated_at": "2026-09-06T00:00:00Z"},
    }
    dec = decode_proposal(raw)
    assert dec.ok, dec.error
    return dec.proposal


_CASE = ScientistCase(
    case_id="case_pc", strategy_id="str", corrected_quant_config_ref="qc://c",
    corrected_run_ref="run://c", audit_report_ref="a://c", failed_check_ids=("lib_gap",),
    failed_check_verdicts={"lib_gap": "FAIL"}, applicable_toggles=("lib_gap",),
    development_window=DevelopmentWindow(start="2004-08", end="2021-12"),
    holdout_status=HoldoutStatus(accessible=False),
)
_AVAILABLE = {"rating", "investment_grade", "size", "gamma_illiq"}


def run_positive_control(beta: float = _BETA, sigma: float = _SIGMA, seed: int = _SEED) -> dict:
    """Run the planted-strong + noise proposals through the REAL funnel; return the control result."""
    lib = load_library()
    base_rb = {"score": "score", "groups": 5, "weighting": "equal", "min_bonds": 4,
               "signal_lag": 0, "nw_lags": 0}
    pos = _proposal("pc_planted_strong", "restrict_investment_grade")   # sees the PLANTED signal
    neg = _proposal("pc_noise", "restrict_high_yield")                  # sees NOISE only
    rep = run_experimentalist(
        _CASE, [pos, neg], lib, panel=_panel(beta=beta, sigma=sigma, seed=seed),
        base_rulebook=base_rb, bbw4_factors=_bbw4(),
        holding_period=1, signal_lookback=1, available_variables=_AVAILABLE, m=6, q=0.10,
        cap=None, direction=1, crowding_config=_cfg(), crowding_factors=_crowding_factors(),
        nw_lags=0)
    recs = {r.proposal_id: r for r in rep.records}

    def _row(pid: str) -> dict:
        r = recs[pid]
        g = r.measurements.gross
        cpcv = r.measurements.cpcv.get("median_sharpe") if r.measurements.cpcv else None
        return {"advanced": pid in rep.advanced,
                "mean_return": getattr(g, "mean_return", None), "sharpe": getattr(g, "sharpe", None),
                "p_raw": getattr(g, "p_raw", None), "p_bh": getattr(g, "p_bh", None),
                "cpcv_median_sharpe": cpcv,
                "refusal_code": r.refusal_code.value if r.refusal_code is not None else None}

    planted, noise = _row("pc_planted_strong"), _row("pc_noise")
    return {
        "control": "rq4_funnel_positive_control",
        "synthetic": True, "reads_real_data": False, "touches_holdout": False, "cost_usd": 0.0,
        "planted_strong": planted, "noise": noise,
        "power_planted_advances": planted["advanced"],
        "specificity_noise_rejected": not noise["advanced"],
        "note": ("Synthetic validation instrument: a planted strong extension ADVANCES and a "
                 "no-signal extension is REJECTED, so the real-data 0/24 null is a true null, not a "
                 "dead pipeline. NOT an RQ4 survivor; never advanced to the holdout."),
    }


def run_sweep(ladder_bp=_DEFAULT_LADDER_BP, n_seeds: int = 20, base_seed: int = _SEED,
              sigma: float = _SIGMA) -> dict:
    """Magnitude ladder x seeds through the SAME funnel path as the single positive control.

    For each (magnitude, seed): the identical two-proposal (planted + noise) funnel call, so the
    BH-FDR family structure matches the recorded run; the PLANTED row is what the curve records.
    Seeds are base_seed + i, i in 0..n_seeds-1 (recorded per cell). Deterministic."""
    per_magnitude = []
    for bp in ladder_bp:
        beta = beta_for_bp(bp)
        cells = []
        for i in range(n_seeds):
            seed = base_seed + i
            res = run_positive_control(beta=beta, sigma=sigma, seed=seed)
            p = res["planted_strong"]
            cells.append({"seed": seed, "advanced": bool(p["advanced"]), "p_bh": p["p_bh"],
                          "cpcv_median_sharpe": p["cpcv_median_sharpe"],
                          "mean_return": p["mean_return"]})
        n_adv = sum(c["advanced"] for c in cells)
        sharpes = [c["cpcv_median_sharpe"] for c in cells if c["cpcv_median_sharpe"] is not None]
        per_magnitude.append({
            "bp_per_month": bp, "beta": beta, "n_seeds": n_seeds,
            "advance_rate": n_adv / n_seeds, "n_advanced": n_adv,
            "median_cpcv_sharpe": statistics.median(sharpes) if sharpes else None,
            "n_with_cpcv": len(sharpes), "cells": cells,
        })
    mde = next((m["bp_per_month"] for m in per_magnitude if m["advance_rate"] >= 0.5), None)
    return {"control": "rq4_funnel_power_curve", "synthetic": True, "reads_real_data": False,
            "touches_holdout": False, "cost_usd": 0.0,
            "ladder_bp": list(ladder_bp), "n_seeds": n_seeds, "base_seed": base_seed,
            "sigma": sigma, "beta_bp_mapping": "beta = 0.00020 * bp / 44 (realised anchor)",
            "empirical_mde_bp": mde, "per_magnitude": per_magnitude,
            "note": ("Synthetic power curve for the RQ4 development funnel (validation instrument, "
                     "never a finding); empirical MDE = smallest ladder magnitude with advance "
                     "rate >= 0.5.")}


def _git_state() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", __file__], cwd=_REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{commit}{' (script modified in working tree)' if dirty else ''}"
    except Exception:
        return "unknown"


def _sweep_markdown(result: dict) -> str:
    lines = [
        "# RQ4 funnel power curve (synthetic positive-control sweep)", "",
        f"Ladder (planted bp/mo): {result['ladder_bp']}  |  n_seeds per magnitude: "
        f"{result['n_seeds']}  |  base seed: {result['base_seed']} (seeds = base + i)  |  "
        f"sigma: {result['sigma']}", "",
        f"Script git state: `{_git_state()}` (`scripts/run_rq4_positive_control.py`). "
        f"Mapping: {result['beta_bp_mapping']}.", "",
        "Synthetic validation instrument ($0, in-memory panel, no /data, never a finding). "
        "Analytic counterpart: rank-1 annualised Sharpe ~= 0.57.", "",
        "| planted bp/mo | beta | advance rate | median CPCV Sharpe |",
        "| --- | --- | --- | --- |",
    ]
    for m in result["per_magnitude"]:
        med = "-" if m["median_cpcv_sharpe"] is None else f"{m['median_cpcv_sharpe']:.3f}"
        lines.append(f"| {m['bp_per_month']:g} | {m['beta']:.6g} | "
                     f"{m['advance_rate']:.2f} ({m['n_advanced']}/{m['n_seeds']}) | {med} |")
    mde = result["empirical_mde_bp"]
    lines += ["", (f"**Empirical MDE (advance rate >= 0.5): {mde:g} bp/mo.**" if mde is not None
                   else "**Empirical MDE (advance rate >= 0.5): not reached on this ladder.**"), ""]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--beta", type=float, default=_BETA,
                    help="planted per-score-unit monthly return (default: historical constant)")
    ap.add_argument("--sigma", type=float, default=_SIGMA,
                    help="idiosyncratic noise sd (default: historical constant)")
    ap.add_argument("--seed", type=int, default=_SEED,
                    help="panel seed; in --sweep mode the BASE seed (seeds = base + i)")
    ap.add_argument("--sweep", action="store_true",
                    help="run the magnitude-ladder x seeds power curve instead of the single control")
    ap.add_argument("--ladder-bp", type=float, nargs="+", default=list(_DEFAULT_LADDER_BP),
                    help="sweep ladder of planted mean monthly returns in bp/mo")
    ap.add_argument("--n-seeds", type=int, default=20, help="seeds per ladder magnitude (sweep)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output path override (single: the JSON; sweep: the JSON, md written "
                         "alongside with .md suffix)")
    return ap


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.sweep:
        t0 = time.monotonic()
        result = run_sweep(ladder_bp=tuple(args.ladder_bp), n_seeds=args.n_seeds,
                           base_seed=args.seed, sigma=args.sigma)
        out = args.out or (_REPO_ROOT / "results" / "scientist" / "rq4_power_curve.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        md = out.with_suffix(".md")
        md.write_text(_sweep_markdown(result), encoding="utf-8")
        for m in result["per_magnitude"]:
            med = m["median_cpcv_sharpe"]
            print(f"[power-curve] {m['bp_per_month']:5.1f}bp advance_rate={m['advance_rate']:.2f} "
                  f"median_cpcv_sharpe={med if med is None else round(med, 3)}")
        print(f"[power-curve] empirical MDE = {result['empirical_mde_bp']} bp/mo "
              f"({time.monotonic() - t0:.0f}s) -> {out} + {md}")
        return 0

    result = run_positive_control(beta=args.beta, sigma=args.sigma, seed=args.seed)
    out = args.out or (_REPO_ROOT / "results" / "scientist" / "rq4_positive_control.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    p, n = result["planted_strong"], result["noise"]
    print(f"[positive-control] PLANTED advanced={p['advanced']} p_bh={p['p_bh']:.2e} "
          f"cpcv_median_sharpe={p['cpcv_median_sharpe']}")
    print(f"[positive-control] NOISE   advanced={n['advanced']} p_bh={n['p_bh']:.3f}")
    print(f"[positive-control] POWER={result['power_planted_advances']} "
          f"SPECIFICITY={result['specificity_noise_rejected']}  -> {out}")
    return 0 if (result["power_planted_advances"] and result["specificity_noise_rejected"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
