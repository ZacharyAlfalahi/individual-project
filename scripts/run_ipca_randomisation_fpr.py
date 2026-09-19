"""Run the IPCA §6.2 matched-twin randomisation false-positive-rate procedure at registered scale.

The procedure is registered (`docs/thresholds.yaml` `auditor.ipca_differential.randomisation_fpr`:
Q = 49 label re-randomisations per dataset, R = 50 independently generated datasets, nominal level
0.05, acceptance-band coverage 0.95). This driver runs the registered cell and writes the artefact
that backs the FPR claim.

It records three things the library's own `FprResult.to_dict()` leaves out, each of which the FPR
claim depends on:

  * the SEED — the procedure is deterministic given one (dataset seeds `seed + rr`, permutation
    seeds `seed + 100_000 + rr`), so without it the artefact cannot be reproduced;
  * the P-VALUES — the per-dataset randomisation p-values, so the result can be re-derived rather
    than taken on trust;
  * a UNIFORMITY statistic — the registered claim asserts the p-values are uniform under
    exchangeability, and nothing in the library tests that. Reported DESCRIPTIVELY: a permutation
    p-value is discrete on multiples of 1/(Q+1), so the continuous-uniform reference is approximate
    and conservative. It is a diagnostic, not a gate.

The acceptance band is an exact equal-tailed binomial band on the REJECT COUNT under
X ~ Binomial(R, alpha). One property of it is worth recording beside the verdict: at R = 50 and
alpha = 0.05, P(X = 0) is about 0.077, which exceeds the 0.025 the lower tail may carry, so the
lower limit is 0 and the band cannot be failed from below. "Inside the band" is therefore a
one-sided statement in practice — only an EXCESS of rejections can fail it — and the artefact says
so rather than leaving the reader to infer it.

Synthetic throughout: the matched-twin generator builds its own panels in memory. No project data is
read, and /data/ is not touched at all.

Usage: ./.venv/bin/python scripts/run_ipca_randomisation_fpr.py [--seed 7] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy import stats

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.ipca_differential.randomisation_fpr import randomisation_fpr  # noqa: E402
from agents.auditor.thresholds import (  # noqa: E402
    load_ipca_fpr_config,
    load_ipca_lambda,
    load_ipca_projection_gate,
)


def uniformity(p_values: tuple[float, ...], q: int) -> dict:
    """Descriptive uniformity of the randomisation p-values. Discrete on 1/(q+1), so the
    continuous-uniform reference is approximate — reported, never gated."""
    p = np.asarray(p_values, dtype=float)
    ks = stats.kstest(p, "uniform")
    deciles = np.histogram(p, bins=10, range=(0.0, 1.0))[0]
    return {
        "test": "one-sample Kolmogorov-Smirnov against U(0,1), DESCRIPTIVE",
        "caveat": f"a randomisation p-value is discrete on multiples of 1/(Q+1) = {1 / (q + 1):.6f}, "
                  "so the continuous-uniform reference is approximate and conservative",
        "min_attainable": 1.0 / (q + 1),
        "ks_d": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mean": float(p.mean()),
        "median": float(np.median(p)),
        "min": float(p.min()),
        "deciles": [int(x) for x in deciles],
        "gating": False,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IPCA §6.2 randomisation FPR at registered scale.")
    ap.add_argument("--seed", type=int, default=7, help="master seed (the run is deterministic given it)")
    ap.add_argument("--r-datasets", type=int, default=None, dest="r",
                    help="override R (default: the registered r_datasets)")
    ap.add_argument("--q-permutations", type=int, default=None, dest="q",
                    help="override Q (default: the registered q_permutations)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)

    cfg, lam, gate = load_ipca_fpr_config(), load_ipca_lambda(), load_ipca_projection_gate()
    started = datetime.now(UTC)
    res = randomisation_fpr(cfg, lam, gate, seed=args.seed, r=args.r, q=args.q)
    elapsed = (datetime.now(UTC) - started).total_seconds()

    lo, hi = res.acceptance_band
    at_registered_scale = (res.r_datasets == cfg.r_datasets and res.q_permutations == cfg.q_permutations)
    payload = {
        "procedure": "ipca_differential §6.2 matched-twin randomisation false-positive rate",
        "computed_on": started.date().isoformat(),
        "run_timestamp": started.isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(elapsed, 3),
        "substrate": "synthetic matched-twin DGP; no project data is read",
        "at_registered_scale": at_registered_scale,
        "registered_scale": {"q_permutations": cfg.q_permutations, "r_datasets": cfg.r_datasets},
        "reduced_fallback_registered": {"q_permutations": cfg.reduced_q, "r_datasets": cfg.reduced_r,
                                        "used": not at_registered_scale,
                                        "note": "the registered reduced-scale compute fallback; used "
                                                "only when R or Q is overridden"},
        "seed": args.seed,
        "twin_dgp": {"n_bonds": cfg.twin_dgp.n_bonds, "n_months": cfg.twin_dgp.n_months,
                     "noise_sd": cfg.twin_dgp.noise_sd},
        **res.to_dict(),
        "p_values": list(res.p_values),
        "acceptance_band_is_one_sided_in_practice": lo == 0,
        "band_note": ("exact equal-tailed binomial band on the REJECT COUNT under X ~ Binomial(R, alpha), "
                      f"carrying at most {(1 - cfg.acceptance_band_level) / 2:.3f} in each tail. "
                      + (f"The lower limit is 0 because with R = {res.r_datasets} and alpha = "
                         f"{res.alpha_nominal}, P(X = 0) already exceeds that tail budget, so no lower cut "
                         "exists and only an excess of rejections can fail the band."
                         if lo == 0 else
                         f"Both limits bite here ({lo} to {hi}), so the band is two-sided.")),
        "uniformity": uniformity(res.p_values, res.q_permutations),
    }

    out_dir = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / "ipca_differential" / "randomisation_fpr")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "randomisation_fpr.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(_REPO_ROOT) if out.is_relative_to(_REPO_ROOT) else out}")
    inner = payload["randomisation_fpr"]
    print(json.dumps({"at_registered_scale": at_registered_scale, "wall_seconds": payload["wall_seconds"],
                      **{k: inner[k] for k in ("n_reject", "fpr_hat", "acceptance_band_counts", "within_band")}},
                     indent=2))
    print("uniformity:", json.dumps({k: payload["uniformity"][k]
                                     for k in ("ks_d", "ks_p", "mean", "deciles")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
