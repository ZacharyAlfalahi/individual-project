#!/usr/bin/env python
"""Compute the registered RQ3 measurements: corpus prevalence pi-tilde_i(vartheta), the audit
completion rate in both readings, and the per-correction OFF-state check.

Reads the audit report artefacts only — no engine re-run, no model call, development data
only.

    ./.venv/bin/python scripts/run_rq3_registered.py --basis clean
    Inputs: the per-anchor <anchor>_report.json + run_log.json under
      results/consistent_basis/<basis>/audit/full_run   (basis_inputs.basis_dir(basis, "audit", "full_run")).
    Neither producer writes there by default, so point --out at it:
      clean        -> scripts/run_full_audit.py --anchor all \
                        --out results/consistent_basis/clean/audit/full_run
      total_return -> scripts/run_full_audit_total_return.py \
                        --out results/consistent_basis/total_return/audit/full_run
    run_full_audit.py has no --basis flag (it always loads the clean-price inputs), so the basis is
    chosen by WHICH driver runs, never by the output path.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.codegen.jsonio import dump_json  # noqa: E402
from evaluation.rq3_registered.completion import (  # noqa: E402
    completion_block,
    ipca_arm,
    lattice_arm,
)
from evaluation.rq3_registered.off_state import off_state_block  # noqa: E402
from evaluation.rq3_registered.prevalence import compute_prevalence  # noqa: E402
from scripts import basis_inputs  # noqa: E402

STRATEGIES = ("drf", "mom6", "str")
RUN_DATE = date.today().isoformat()
_THRESHOLDS = _REPO_ROOT / "docs" / "thresholds.yaml"


def load_materiality(path: Path | None = None) -> tuple[float, tuple[float, ...]]:
    """(vartheta, sensitivity grid) read fail-loud from thresholds — never defaulted."""
    doc = yaml.safe_load((path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        block = doc["auditor"]["practical_significance"]
        return float(block["vartheta"]), tuple(float(v) for v in block["vartheta_sensitivity_grid"])
    except (KeyError, TypeError) as exc:
        raise KeyError("thresholds.yaml has no auditor.practical_significance "
                       "vartheta / vartheta_sensitivity_grid") from exc


def load_reports(audit_dir: Path) -> dict[str, dict]:
    reports = {}
    for strategy in STRATEGIES:
        path = audit_dir / f"{strategy}_report.json"
        if not path.is_file():
            raise SystemExit(f"REFUSED: missing audit report {path}")
        reports[strategy] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def load_ipca_counts(basis: str) -> tuple[int, int, str]:
    """(runnable, refused, refusal code) for the fitted-model differential arm."""
    path = basis_inputs.basis_dir(basis, "ipca_differential", "results_9pairs.json")
    if not path.is_file():
        return 0, 0, ""
    doc = json.loads(path.read_text(encoding="utf-8"))
    runnable = doc.get("runnable")
    refused = doc.get("refused")
    n_runnable = len(runnable) if isinstance(runnable, list) else int(runnable or 0)
    if isinstance(refused, list):
        n_refused, codes = len(refused), {str(r.get("reason") or r.get("code") or "")
                                          for r in refused if isinstance(r, dict)}
    elif isinstance(refused, dict):
        n_refused, codes = int(refused.get("n_refused") or len(refused)), set()
    else:
        n_refused, codes = int(refused or 0), set()
    return n_runnable, n_refused, ", ".join(sorted(c for c in codes if c))


def render(result: dict) -> str:
    out = [f"# RQ3 registered measurements — {result['basis']} basis", ""]

    prev = result["prevalence"]
    out.append("## Corpus prevalence")
    out.append("")
    out.append(f"- estimand: `{prev['estimand']}`")
    out.append(f"- denominator rule: {prev['denominator_rule']}")
    out.append(f"- measurement path: {prev['measurement_path']}")
    out.append(f"- vartheta = {prev['vartheta']}")
    out.append("")
    out.append("| correction | n runnable | n susceptible | prevalence at vartheta | mu (mean) | tau (mean) |")
    out.append("|---|---|---|---|---|---|")
    for coord, block in sorted(prev["coordinates"].items()):
        out.append(f"| {coord} | {block['n_runnable']} | {block['n_susceptible']} | "
                   f"{block['prevalence']:.3f} | {block['mu_mean']:.6f} | {block['tau_mean']:.6f} |")
    out.append("")

    comp = result["completion"]
    out.append("## Audit completion rate")
    out.append("")
    out.append(f"_{comp['pooling']}_")
    out.append("")
    out.append("| arm | unit | completed | refused | not applicable | conservative | permissive |")
    out.append("|---|---|---|---|---|---|---|")
    for arm in comp["arms"]:
        out.append(f"| {arm['arm']} | {arm['unit']} | {arm['completed']} | {arm['refused']} | "
                   f"{arm['not_applicable']} | {arm['conservative_fraction']} "
                   f"({arm['conservative']:.2f}) | {arm['permissive_fraction']} "
                   f"({arm['permissive']:.2f}) |")
    out.append("")

    off = result["off_state"]
    out.append("## Per-correction OFF state")
    out.append("")
    out.append(f"_{off['scope_limit']}_")
    out.append("")
    out.append(f"- checked: {off['n_checked']} | documented: {off['n_documented']} | "
               f"undocumented: {off['n_undocumented']}")
    if off["no_ops_proved_inert"]:
        out.append(f"- declared no-ops proved inert by the invariance gate: "
                   f"{', '.join(off['no_ops_proved_inert'])}")
    out.append("")
    out.append("| strategy | correction | OFF state | source |")
    out.append("|---|---|---|---|")
    for c in off["checks"]:
        out.append(f"| {c['strategy']} | {c['correction']} | `{c['off_state']}` | {c['source']} |")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=basis_inputs.BASES, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    audit_dir = basis_inputs.basis_dir(args.basis, "audit", "full_run")
    reports = load_reports(audit_dir)
    run_log_path = audit_dir / "run_log.json"
    if not run_log_path.is_file():
        raise SystemExit(f"REFUSED: missing {run_log_path} (the baseline signatures live there)")
    signatures = json.loads(run_log_path.read_text(encoding="utf-8")).get("baseline_signatures", {})

    vartheta, grid = load_materiality()
    n_runnable, n_refused, codes = load_ipca_counts(args.basis)

    arms = [lattice_arm(reports)]
    if n_runnable or n_refused:
        arms.append(ipca_arm(n_runnable, n_refused, refusal_code=codes))

    result = {
        "experiment": "rq3_registered_measurements",
        "computed_after_the_fact": True,
        "computed_on": RUN_DATE,
        "note": ("computed from the audit report artefacts with no engine re-run and no model call"),
        "basis": args.basis,
        "audit_dir": str(audit_dir),
        "strategies": list(STRATEGIES),
        "prevalence": compute_prevalence(reports, vartheta=vartheta, vartheta_grid=grid,
                                         seed=args.seed),
        "completion": completion_block(arms),
        "off_state": off_state_block(reports, signatures),
    }

    out_dir = basis_inputs.basis_dir(args.basis, "auditor")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "rq3_registered.json"
    out_md = out_dir / "rq3_registered.md"
    existing = [p for p in (out_json, out_md) if p.exists()]
    if existing:
        print(f"REFUSED: {', '.join(str(p) for p in existing)} already exist(s); move or delete "
              "to re-run.", file=sys.stderr)
        return 2
    out_json.write_text(dump_json(result), encoding="utf-8")
    out_md.write_text(render(result), encoding="utf-8")

    prev = result["prevalence"]["coordinates"]
    print(f"prevalence at vartheta={vartheta} ({args.basis}):")
    for coord, block in sorted(prev.items()):
        print(f"   {coord:24s} pi-tilde={block['prevalence']:.3f} "
              f"(n_runnable={block['n_runnable']}, susceptible={block['n_susceptible']})")
    for arm in result["completion"]["arms"]:
        print(f"completion [{arm['arm']}]: conservative {arm['conservative_fraction']}, "
              f"permissive {arm['permissive_fraction']}")
    off = result["off_state"]
    print(f"off-state: {off['n_documented']}/{off['n_checked']} documented")
    print(f"results written: {out_json}")
    print(f"report written:  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
