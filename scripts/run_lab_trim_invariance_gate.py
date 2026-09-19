"""
lab_trim no-op invariance gate on the recorded audit lattices (str, drf) — §3.5 gate, declared.

The registered audit declares `expect_no_op` for `stale_price` only
(`run_auditor.default_anchor_facts`), so its invariance block certifies `stale_price` and not
`lab_trim`. This driver declares
`expect_no_op: true` for `lab_trim` (and keeps the registered `stale_price` declaration), re-executes
the deterministic lattice spine for each anchor through the SAME per-anchor wiring as
`run_full_audit.run_anchor_full` (facts, meas_err OFF family, published-trim re-injection), and
runs the §3.5 gate (config hashes differ; return, bond-count and metric hashes identical on every
parallel edge).

Bootstrap/inference are not run — the gate needs only the lattice. The recomputed lattice is
first verified against the recorded audit report: every Harsanyi dividend, Walsh coefficient, DOE effect
and Shapley value must equal the recorded report's value exactly, else the driver refuses (the
gate would otherwise certify a different lattice). The recorded audit reports are not modified.

`--basis` runs the gate on one return basis (inputs via `scripts/basis_inputs.load_basis_inputs`);
the lattice is then verified against `--audit-reports`, which must hold that basis's audit reports (a
`{anchor}` placeholder addresses per-anchor run directories). Without `--basis` the clean development
loader, the default audit reports and the default output path are used.

Dev-only; the holdout is never read. Usage:
  python scripts/run_lab_trim_invariance_gate.py [--anchors str drf]
  python scripts/run_lab_trim_invariance_gate.py --basis total_return --audit-reports '<TR audit dir>'
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.ipca_differential.runner import load_dev_inputs  # noqa: E402
from scripts.run_auditor import (  # noqa: E402
    audit_anchor,
    default_anchor_facts,
    load_anchor_expost_trim_off,
    load_anchor_meas_err_off_family,
    load_anchor_strategy,
)
from scripts import basis_inputs  # noqa: E402
from shared.licensed_inputs import require_licensed_input  # noqa: E402

RECORDED_AUDIT = REPO_ROOT / "results" / "auditor" / "{anchor}_corrected"
DEFAULT_OUT = REPO_ROOT / "results" / "auditor" / "lab_trim_invariance_gate.json"
DECLARED_NO_OP = ("stale_price", "lab_trim")


def load_inputs(basis: str | None):
    """(maximal, signals, registry): the clean development loader without a basis, else the basis loader."""
    return load_dev_inputs() if basis is None else basis_inputs.load_basis_inputs(basis)


def default_out(basis: str | None = None) -> Path:
    """The default artefact path without a basis; the basis-specific location with one."""
    return DEFAULT_OUT if basis is None else basis_inputs.basis_dir(
        basis, "auditor", "lab_trim_invariance_gate.json")


def report_path(audit_dir: Path, anchor: str) -> Path:
    """``<audit_dir>/<anchor>_report.json``; a ``{anchor}`` placeholder in the directory selects
    per-anchor run directories (e.g. ``{anchor}_total_return``)."""
    return Path(str(audit_dir).replace("{anchor}", anchor)) / f"{anchor}_report.json"


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


class LatticeMismatch(RuntimeError):
    """The recomputed lattice does not reproduce the recorded audit report — refuse to gate it."""


def gate_facts(anchor_id: str):
    """The registered anchor facts with `expect_no_op` additionally declared for lab_trim."""
    return [dataclasses.replace(f, expect_no_op=True)
            if f.toggle_id in DECLARED_NO_OP and f.runnable else f
            for f in default_anchor_facts(anchor_id)]


def verify_reproduction(core_dict: dict, report: dict) -> dict:
    """Exact equality of the saturated bases and Shapley values with the recorded report."""
    checked = 0
    for basis in ("harsanyi_dividends", "walsh_coefficients", "doe_effects"):
        got, want = core_dict["saturated_bases"][basis], report["saturated_bases"][basis]
        if set(got) != set(want):
            raise LatticeMismatch(f"{basis}: coordinate sets differ")
        for k in want:
            if got[k] != want[k]:
                raise LatticeMismatch(f"{basis}[{k}]: recomputed {got[k]!r} != recorded {want[k]!r}")
            checked += 1
    got_s, want_s = core_dict["shapley"]["shapley_values"], report["shapley"]["shapley_values"]
    if got_s != want_s:
        raise LatticeMismatch(f"shapley values differ: {got_s} vs {want_s}")
    return {"exact": True, "n_coordinates_checked": checked + len(want_s)}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--anchors", nargs="+", default=["str", "drf"])
    ap.add_argument("--out", type=Path, default=None,
                    help=f"output JSON (default {_rel(DEFAULT_OUT)}; with --basis "
                         "results/consistent_basis/<basis>/auditor/lab_trim_invariance_gate.json)")
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="return basis of the dev inputs (default: the clean development loader)")
    ap.add_argument("--audit-reports", type=Path, default=RECORDED_AUDIT,
                    help=f"audit directory holding <anchor>_report.json (default {_rel(RECORDED_AUDIT)}; "
                         "a {anchor} placeholder selects per-anchor directories)")
    args = ap.parse_args(argv)
    if args.basis == "total_return" and args.audit_reports == RECORDED_AUDIT:
        ap.error(f"--basis total_return needs --audit-reports: the default {_rel(RECORDED_AUDIT)} holds the "
                 "clean-basis audit reports, whose lattice a total-return panel cannot reproduce")
    out = args.out if args.out is not None else default_out(args.basis)

    maximal, signals, _registry = load_inputs(args.basis)   # holdout never read
    per_anchor: dict = {}
    for a in args.anchors:
        rpath = report_path(args.audit_reports, a)
        report = json.loads(require_licensed_input(
            rpath, "auditor report (local pipeline output, not shipped with the repository)",
        ).read_text(encoding="utf-8"))
        facts = gate_facts(a)
        core = audit_anchor(
            load_anchor_strategy(a), maximal, facts, signals,
            expost_trim_off=load_anchor_expost_trim_off(a),
            meas_err_off_family=load_anchor_meas_err_off_family(a),
        )
        cd = core.to_dict()
        reproduction = verify_reproduction(cd, report)
        per_anchor[a] = {
            "audit_report": {"path": _rel(rpath), "sha256": _sha256(rpath)},
            "declared_expect_no_op": [f.toggle_id for f in facts if f.expect_no_op],
            "lattice_reproduction": reproduction,
            "invariance": cd["invariance"],
        }
        for inv in cd["invariance"]:
            print(f"{a}: {inv['toggle_id']:12} is_no_op={inv['is_no_op']}  {inv['note']}")

    artifact = {
        "purpose": ("§3.5 no-op invariance gate with expect_no_op declared for lab_trim on the "
                    "recorded audit lattices; lattice reproduced exactly before gating."),
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_reports": _rel(args.audit_reports),
        "basis": args.basis or "clean",
        "window": "development 2002-2021 (holdout untouched)",
        "anchors": per_anchor,
    }
    if args.basis is not None:
        artifact["basis_provenance"] = basis_inputs.basis_provenance(args.basis).to_dict()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"written: {_rel(out)}")
    return artifact


if __name__ == "__main__":
    main()
