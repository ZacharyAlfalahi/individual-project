"""Run the IPCA §6.3 stochastic perturbation robustness calibration, and its secondary zero-coverage check.

§6.3 is the mean-zero-noise experiment, named as a stability calibration rather than an FPR. It is
registered (`docs/thresholds.yaml` `auditor.ipca_differential.perturbation_robustness`: 200 draws,
noise SD 0.02); this driver executes it at the registered scale as a calibration of the shipped
conditional bootstrap intervals.

The frozen registration (`docs/auditor/evaluation_frozen_ipca_differential_v5_FINAL.md` §6.3) fixes
two things that this driver keeps in front of the reader:

  * The centre is EXPECTED to be non-null and must never be read as a false-positive rate. Mean-zero
    return noise does not give a mean-zero bracket through a nonlinear fitted model — that is the
    mean-zero fallacy the project's own errors-in-variables mechanism exists to refute. What the
    draw distribution measures is the variance/attenuation channel. `is_fpr` is False by
    construction and is carried into the artefact.
  * The secondary check runs "within 6.2's samples": for each matched-twin null dataset, the §5.3
    conditional bootstrap interval either covers zero or does not. Over-coverage is expected under
    an exchangeable null, where the bracket is near zero with a tiny magnitude, so a fraction above
    1 − alpha is conservative rather than miscalibrated.

ONE SUBSTRATE CHOICE IS THE DRIVER'S, AND IS RECORDED AS SUCH. The registration pins the draw count
and the noise SD but not the panel the draws perturb. This driver perturbs a §6.2 matched-twin pair
rather than two independently drawn synthetic panels, so that both halves of §6.3 sit on the same
substrate the registration names for the secondary, and so that the draw distribution is not
contaminated by a real between-panel difference. The alternative would measure perturbation and a
genuine panel difference at once.

Synthetic throughout: no project data is read and /data/ is not touched.

Usage: ./.venv/bin/python scripts/run_ipca_perturbation_robustness.py [--seed 7] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from agents.auditor.ipca_differential.perturbation import (  # noqa: E402
    bootstrap_zero_coverage,
    perturbation_robustness,
)

# `_panels_from_label` is the package's own way of turning a matched-twin dataset into the two feeds
# the differential consumes. It is imported rather than reimplemented so this driver perturbs exactly
# the panels §6.2 randomises over; duplicating that construction here would be a second definition of
# the substrate, which is the failure mode §5.2 of the frozen spec warns about.
from agents.auditor.ipca_differential.randomisation_fpr import (  # noqa: E402
    _panels_from_label,
    make_matched_twin_dataset,
)
from agents.auditor.thresholds import (  # noqa: E402
    load_ipca_bootstrap_config,
    load_ipca_fpr_config,
    load_ipca_lambda,
    load_ipca_perturbation_config,
    load_ipca_projection_gate,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IPCA §6.3 perturbation robustness + zero coverage.")
    ap.add_argument("--seed", type=int, default=7, help="master seed (the run is deterministic given it)")
    ap.add_argument("--n-draws", type=int, default=None, dest="n_draws",
                    help="override the registered n_draws (for smoke runs)")
    ap.add_argument("--r-datasets", type=int, default=None, dest="r",
                    help="override R for the zero-coverage check (default: the registered r_datasets)")
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args(argv)

    lam, gate = load_ipca_lambda(), load_ipca_projection_gate()
    fpr_cfg, boot_cfg, pcfg = load_ipca_fpr_config(), load_ipca_bootstrap_config(), load_ipca_perturbation_config()
    if args.n_draws is not None:
        import dataclasses
        pcfg = dataclasses.replace(pcfg, n_draws=args.n_draws)

    started = datetime.now(UTC)
    twin = make_matched_twin_dataset(fpr_cfg.twin_dgp, lam, seed=args.seed)
    feed_n, feed_b = _panels_from_label(twin, np.zeros(twin.n_bonds, dtype=int))
    pert = perturbation_robustness(feed_n, feed_b, twin.anchor, lam, gate, pcfg, seed=args.seed)
    zc = bootstrap_zero_coverage(fpr_cfg, boot_cfg, lam, gate, seed=args.seed, r=args.r)
    elapsed = (datetime.now(UTC) - started).total_seconds()

    p = pert.to_dict()["perturbation_robustness"]
    at_registered_scale = (p["n_draws"] == load_ipca_perturbation_config().n_draws
                           and zc.r_datasets == fpr_cfg.r_datasets)
    payload = {
        "procedure": "ipca_differential §6.3 stochastic perturbation robustness, with the §6.3 secondary "
                     "bootstrap zero-coverage check within §6.2's matched-twin samples",
        "computed_on": started.date().isoformat(),
        "run_timestamp": started.isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(elapsed, 3),
        "substrate": "synthetic §6.2 matched-twin pair; no project data is read",
        "substrate_is_a_driver_choice": {
            "registered": "n_draws and noise_sd only",
            "chosen": "a §6.2 matched-twin pair (both panels perturbed, anchor held fixed)",
            "why": "keeps both halves of §6.3 on the substrate the registration names for the secondary, "
                   "and keeps a real between-panel difference out of the draw distribution",
        },
        "at_registered_scale": at_registered_scale,
        "seed": args.seed,
        "twin_dgp": {"n_bonds": fpr_cfg.twin_dgp.n_bonds, "n_months": fpr_cfg.twin_dgp.n_months,
                     "noise_sd": fpr_cfg.twin_dgp.noise_sd},
        "bootstrap": {"n_replicates": boot_cfg.n_replicates, "block_length_months": boot_cfg.block_length_months,
                      "alpha": boot_cfg.alpha},
        **pert.to_dict(),
        **zc.to_dict(),
        "draws_retained": False,
        "reproduction": "the library returns summary statistics only, not the individual draws; the run is "
                        "deterministic given the recorded seed, so re-running reproduces them exactly",
        "reading_notes": {
            "centre": "a non-null i_mean is EXPECTED and is the variance/attenuation channel, never a "
                      "false-positive rate (is_fpr is False)",
            "zero_coverage": "over-coverage above 1 - alpha is expected under the exchangeable null, where the "
                             "bracket is near zero with tiny magnitude, so the interval sits astride zero "
                             "conservatively; it is not evidence of miscalibration",
        },
    }

    out_dir = Path(args.out) if args.out else (
        _REPO_ROOT / "results" / "ipca_differential" / "perturbation_robustness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "perturbation_robustness.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(_REPO_ROOT) if out.is_relative_to(_REPO_ROOT) else out}")
    print(json.dumps({"at_registered_scale": at_registered_scale, "wall_seconds": payload["wall_seconds"],
                      "perturbation": {k: p[k] for k in ("n_draws", "i_obs", "i_mean", "i_std", "n_usable")},
                      "zero_coverage": zc.to_dict()["bootstrap_zero_coverage"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
