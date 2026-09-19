#!/usr/bin/env python
"""Development-window equivalence test (TOST) on the RQ4 survivors' residual alpha —
registered SC-SCI-14.

Reads the survivors' pinned development-window alpha and NW t from the one-shot holdout
development pins (scripts/oneshot_holdout_prepare_dev_pins.py); the standard error is
|alpha / t|. Its OWN BH family at the registered q — never pooled with the proposal families.
Development data only; the holdout is not touched.

    ./.venv/bin/python scripts/run_rq4_equivalence.py --basis clean

The asymmetry travels with every output: failing to establish equivalence is NOT evidence of
a non-zero alpha.
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
from scripts import basis_inputs  # noqa: E402
from shared.evaluation.equivalence import (  # noqa: E402
    ASYMMETRY,
    EquivalenceInput,
    equivalence_family,
    load_margin,
)

RUN_DATE = date.today().isoformat()
_PROTOCOL = _REPO_ROOT / "docs" / "scientist_protocol.yaml"


def load_q(path: Path | None = None) -> float:
    """The equivalence family's OWN BH level, read fail-loud from the protocol."""
    doc = yaml.safe_load((path or _PROTOCOL).read_text(encoding="utf-8"))
    try:
        return float(doc["inference"]["equivalence_test"]["bh_fdr_q"])
    except (KeyError, TypeError):
        pass
    # the block may sit at the document root depending on the protocol's nesting
    for node in doc.values() if isinstance(doc, dict) else []:
        if isinstance(node, dict) and "equivalence_test" in node:
            return float(node["equivalence_test"]["bh_fdr_q"])
    raise KeyError("scientist_protocol.yaml has no equivalence_test.bh_fdr_q")


def inputs_from_pins(pins: dict) -> list[EquivalenceInput]:
    """One EquivalenceInput per survivor. A survivor whose t is zero has no usable standard
    error and is refused rather than silently given an invented one."""
    out = []
    for row in pins.get("survivors", []):
        g3 = row.get("g3") or {}
        alpha, t_stat = g3.get("alpha_bbw4"), g3.get("t_stat")
        if alpha is None or t_stat in (None, 0):
            raise SystemExit(
                f"REFUSED: survivor {row.get('survivor_id')!r} has alpha={alpha}, t={t_stat} — "
                "no standard error can be derived")
        out.append(EquivalenceInput(
            candidate_id=str(row["survivor_id"]),
            alpha=float(alpha),
            se=abs(float(alpha) / float(t_stat)),
            n_months=int(row["n_months_dev"]),
        ))
    return out


def render(result: dict) -> str:
    fam = result["equivalence"]
    out = [f"# RQ4 equivalence test (TOST) — {result['basis']} basis", ""]
    out.append("_Registered as SC-SCI-14 and computed on the development window after proposal "
               "selection. It decides nothing the proposal families decide._")
    out.append("")
    out.append(f"- test: {fam['test']}")
    out.append(f"- margin delta = {fam['margin_delta']:.6f} "
               f"({result['margin']['factor_of_vartheta']} x vartheta = {result['margin']['vartheta']})")
    out.append(f"- family: {fam['family']}, BH q = {fam['bh_q']}")
    out.append(f"- window: {fam['window']}")
    out.append("")
    out.append(f"> **{fam['asymmetry']}**")
    out.append("")
    out.append("| candidate | alpha | se | n months | p_TOST | BH-adjusted | equivalent |")
    out.append("|---|---|---|---|---|---|---|")
    for r in fam["results"]:
        if r["status"] != "tested":
            out.append(f"| {r['candidate_id']} | {r['alpha']:.6f} | {r['se']:.6f} | "
                       f"{r['n_months']} | — | — | _{r['status']}_ |")
            continue
        out.append(f"| {r['candidate_id']} | {r['alpha']:.6f} | {r['se']:.6f} | {r['n_months']} | "
                   f"{r['p_tost']:.4f} | {r['adjusted_p']:.4f} | {r['equivalent']} |")
    out.append("")
    out.append(f"Equivalence established for **{fam['n_equivalent']} of {fam['n_tested']}** "
               "tested candidates.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basis", choices=basis_inputs.BASES, required=True)
    args = ap.parse_args(argv)

    pins_path = basis_inputs.basis_dir(args.basis, "rq4", "oneshot_holdout", "oneshot_dev_pins.json")
    if not pins_path.is_file():
        print(f"REFUSED: {pins_path} does not exist (local pipeline output, not shipped with the "
              "repository); run scripts/oneshot_holdout_prepare_dev_pins.py for this basis first.",
              file=sys.stderr)
        return 2
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    inputs = inputs_from_pins(pins)
    if not inputs:
        print(f"REFUSED: {pins_path} lists no survivors.", file=sys.stderr)
        return 2

    margin = load_margin()
    q = load_q()
    result = {
        "experiment": "rq4_equivalence_tost",
        "computed_after_the_fact": True,
        "computed_on": RUN_DATE,
        "basis": args.basis,
        "source_pins": str(pins_path),
        "parent_dev_mean": pins.get("parent_dev_mean"),
        "margin": margin,
        "asymmetry": ASYMMETRY,
        "equivalence": equivalence_family(inputs, delta=margin["delta"], q=q),
    }

    out_dir = basis_inputs.basis_dir(args.basis, "rq4", "equivalence")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "equivalence.json"
    out_md = out_dir / "equivalence.md"
    existing = [p for p in (out_json, out_md) if p.exists()]
    if existing:
        print(f"REFUSED: {', '.join(str(p) for p in existing)} already exist(s); move or delete "
              "to re-run.", file=sys.stderr)
        return 2
    out_json.write_text(dump_json(result), encoding="utf-8")
    out_md.write_text(render(result), encoding="utf-8")

    fam = result["equivalence"]
    print(f"TOST ({args.basis}): delta={margin['delta']:.6f}, q={q}, "
          f"{fam['n_tested']} tested, {fam['n_equivalent']} equivalent, "
          f"{fam['n_skipped_insufficient_window']} skipped")
    for r in fam["results"]:
        if r["status"] == "tested":
            print(f"   {r['candidate_id']:42s} alpha={r['alpha']:.5f} p_TOST={r['p_tost']:.4f} "
                  f"adj={r['adjusted_p']:.4f} equivalent={r['equivalent']}")
    print(f"results written: {out_json}")
    print(f"report written:  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
