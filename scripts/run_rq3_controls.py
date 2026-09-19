"""RQ3 controls export — surface the two instrument-specificity controls into one
results JSON.

Two controls, one report:

  * **Positive (known-error) control** — REAL dev-data. BBW carries a documented look-ahead/lead
    defect; the integrated pipeline must reproduce it (as-published correlation collapses) and its
    correction must fix it (round-trip correlation restores). This grades the local
    ``data/development/headlines/leadlag_gate.json`` via
    ``known_error_control.evaluate_known_error_control`` against the pre-registered band
    ``validation.gate_thresholds.lead_lag``.

  * **Negative control (specificity)** — REAL dev-data. The ``traded_liquidity`` control maps to LRF
    (gamma-illiquidity sort); ``negative_control_run.run_negative_control_from_dev`` computes LRF's
    corrected-minus-uncorrected premium differential under the ``meas_err`` / DRR price-cleaning
    correction (on the maximal clean panel — the audit-anchor basis) and its pre-registered
    confirmatory block-bootstrap CI, then grades it with the ``separated``-mode gate. Run ONCE: the
    CI + PASS/FAIL is reported as-is (a CI outside ±vartheta is a finding, never re-run). A
    total-return point cross-check is reported as a pre-declared sensitivity.

Deterministic, offline, DEV-ONLY, $0. Emits ``results/auditor/rq3_controls.json`` (with ``--basis``:
``results/consistent_basis/<basis>/auditor/rq3_controls.json``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from shared.licensed_inputs import require_licensed_input  # noqa: E402

from agents.auditor.validation.known_error_control import (  # noqa: E402
    evaluate_known_error_control,
)
from agents.auditor.validation.negative_control_run import (  # noqa: E402
    run_negative_control_from_dev,
)
from scripts import basis_inputs  # noqa: E402

_THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"
_LEADLAG = REPO_ROOT / "data" / "development" / "headlines" / "leadlag_gate.json"
_BBW_FACTORS = REPO_ROOT / "data" / "development" / "factors" / "bbw_factors.parquet"
_RESULTS_DIR = REPO_ROOT / "results" / "auditor"

_NEG_CONTROL_TEST = "tests/unit/test_negative_control_gate.py"


class RQ3ControlsError(RuntimeError):
    """A required control artefact is missing/malformed — surfaced loudly, never a silent pass."""


def load_vartheta(thresholds_path: Path | None = None) -> float:
    """The pre-registered materiality floor ``auditor.practical_significance.vartheta`` — fail loud
    if absent (never default a specificity floor)."""
    doc = yaml.safe_load((thresholds_path or _THRESHOLDS).read_text(encoding="utf-8"))
    try:
        return float(doc["auditor"]["practical_significance"]["vartheta"])
    except (KeyError, TypeError) as exc:
        raise RQ3ControlsError(
            "auditor.practical_significance.vartheta missing from thresholds.yaml") from exc


def positive_control(gate_report: dict, *, thresholds_path: Path | None = None) -> dict:
    """Grade the known-error positive control from a lead/lag gate report (real dev-data). A
    malformed report (e.g. missing the drf arm) is re-surfaced as the module's typed error."""
    try:
        verdict = evaluate_known_error_control(gate_report, thresholds_path=thresholds_path)
    except ValueError as exc:
        raise RQ3ControlsError(f"known-error control could not grade the gate report: {exc}") from exc
    out = verdict.to_dict()
    out["basis"] = "real_dev_data"
    out["source"] = {
        "gate_report_run_timestamp": gate_report.get("run_timestamp"),
    }
    return out


def total_return_sensitivity(factors_path: Path | None = None, *, basis: str = "total_return") -> dict | None:
    """A pre-declared SENSITIVITY cross-check: the point mean of the LRF corrected-minus-uncorrected
    differential on the default TOTAL-RETURN factor series (``bbw_factors.parquet``), beside the
    confirmatory maximal-basis number. Corrected-minus-uncorrected differentials are expected to be
    basis-invariant, so they should agree. Returns ``None`` if the artefact is absent (a cross-check,
    not a gate).

    ``factors_path`` reads another factor file (e.g. a consistent-basis run's), labelled ``basis``
    and recorded with its SHA-256; omitted, the default factors file is read and the output is
    unchanged."""
    path = _BBW_FACTORS if factors_path is None else Path(factors_path)
    if not path.is_file():
        return None
    import pandas as pd
    df = pd.read_parquet(require_licensed_input(path, "BBW factor panel"))
    diff = (df["lrf_corr"] - df["lrf_raw"]).dropna()
    out = {"basis": basis, "point_gap_mean": float(diff.mean()), "n_months": int(len(diff)),
           "note": "point cross-check only (no CI); the confirmatory interval is the maximal-basis run"}
    if factors_path is not None:
        out["factors_file"] = str(path)
        out["factors_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def negative_control(
    *,
    thresholds_path: Path | None = None,
    seed: int = 0,
    basis: str | None = None,
    factors_dir: Path | None = None,
) -> dict:
    """The REAL dev-data negative-control specificity result: LRF corrected-vs-uncorrected differential
    (meas_err / price-cleaning) through the pre-registered confirmatory block bootstrap, graded by the
    separated-mode gate. Run ONCE — the CI + PASS/FAIL is reported as-is. Adds the unit-verified note
    and the total-return sensitivity cross-check.

    ``basis`` runs the control on that return basis's dev panel (``basis_inputs.load_basis_inputs``)
    and records its panel provenance; ``factors_dir`` points the cross-check at that directory's
    ``bbw_factors.parquet`` (labelled with ``basis`` when one is given). Both omitted ⇒ the default run."""
    if basis is None:
        result = run_negative_control_from_dev(seed=seed, thresholds_path=thresholds_path)
    else:
        maximal, signals, _registry = basis_inputs.load_basis_inputs(basis)
        result = run_negative_control_from_dev(seed=seed, thresholds_path=thresholds_path,
                                               maximal=maximal, signals=signals, return_basis=basis)
        result["basis_provenance"] = basis_inputs.basis_provenance(basis).to_dict()
    result["unit_verified"] = {
        "gate_test": _NEG_CONTROL_TEST,                       # the pure separated-mode gate
        "run_path_test": "tests/unit/test_negative_control_run.py",  # the differential/bootstrap/dev wiring
    }
    if factors_dir is None:
        result["sensitivity_total_return"] = total_return_sensitivity()
    else:
        result["sensitivity_total_return"] = total_return_sensitivity(
            Path(factors_dir) / _BBW_FACTORS.name, basis=basis or "total_return")
    return result


def build_controls(
    gate_report: dict,
    *,
    thresholds_path: Path | None = None,
    negative_control_fn=None,
    basis: str | None = None,
    factors_dir: Path | None = None,
) -> dict:
    """Assemble both controls. ``negative_control_fn`` is injectable so tests can supply a fixture
    instead of running the dev-data computation; when omitted it resolves to the module-level
    ``negative_control`` at call time (so a monkeypatch on it is honoured). ``basis``/``factors_dir``
    are forwarded to the negative control only when given."""
    neg_fn = negative_control_fn or negative_control
    neg_kwargs = {k: v for k, v in (("basis", basis), ("factors_dir", factors_dir)) if v is not None}
    return {
        "experiment": "rq3_instrument_controls",
        "positive_control": positive_control(gate_report, thresholds_path=thresholds_path),
        "negative_control": neg_fn(thresholds_path=thresholds_path, **neg_kwargs),
    }


def default_out(basis: str | None = None) -> Path:
    """The default ``results/auditor/rq3_controls.json`` without a basis; the consistent-basis
    location with one (a basis run never overwrites the default export)."""
    name = "rq3_controls.json"
    return _RESULTS_DIR / name if basis is None else basis_inputs.basis_dir(basis, "auditor", name)


def default_leadlag(basis: str | None = None) -> Path:
    """The default lead/lag gate report without a basis; with one, that basis's own gate (built from its factors)."""
    return _LEADLAG if basis is None else basis_inputs.basis_dir(basis, "headlines", "leadlag_gate.json")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leadlag", default=None,
                    help="lead/lag gate report (default data/development/headlines/leadlag_gate.json; with --basis "
                         "results/consistent_basis/<basis>/headlines/leadlag_gate.json)")
    ap.add_argument("--out", default=None,
                    help="output path (default results/auditor/rq3_controls.json; with --basis "
                         "results/consistent_basis/<basis>/auditor/rq3_controls.json)")
    ap.add_argument("--basis", choices=basis_inputs.BASES, default=None,
                    help="return basis of the negative-control dev panel (default: the clean maximal "
                         "panel via load_dev_inputs)")
    ap.add_argument("--factors-dir", type=Path, default=None,
                    help="directory holding bbw_factors.parquet for the point cross-check (default "
                         "data/development/factors; with --basis, that basis's factors dir)")
    args = ap.parse_args(argv)

    leadlag_path = Path(args.leadlag) if args.leadlag else default_leadlag(args.basis)
    if args.basis is not None and leadlag_path.resolve() == Path(_LEADLAG).resolve():
        raise RQ3ControlsError(
            f"--basis {args.basis} with the default lead/lag gate {_LEADLAG}: that gate is built from the "
            "data/development/factors series, not this basis's factors — pass the basis's own gate "
            "(or omit --leadlag)")
    if not leadlag_path.is_file():
        raise RQ3ControlsError(
            f"lead/lag gate report missing: {leadlag_path} — regenerate it "
            "(./.venv/bin/python scripts/run_leadlag_gate.py) before exporting")
    gate_report = json.loads(leadlag_path.read_text(encoding="utf-8"))

    factors = args.factors_dir
    if factors is None and args.basis is not None:
        factors = basis_inputs.factors_dir(args.basis)
    result = build_controls(gate_report, basis=args.basis, factors_dir=factors)
    stamp = datetime.now(timezone.utc)
    result["provenance"] = {
        "exporter": "scripts/run_rq3_controls.py",
        "run_timestamp": stamp.isoformat(),
        "dev_only": True,
        "model_calls": 0,
        "spend_usd": 0.0,
        "leadlag_gate_sha256": hashlib.sha256(leadlag_path.read_bytes()).hexdigest(),
    }
    if args.basis is not None:
        result["provenance"]["basis"] = args.basis
        result["provenance"]["leadlag_gate"] = str(leadlag_path)

    out = Path(args.out) if args.out else default_out(args.basis)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    pc = result["positive_control"]
    nc = result["negative_control"]
    print(f"positive (known-error) control: passed={pc['passed']} — {pc['detail']}")
    print(f"negative (specificity) control [{nc['basis']}]: passed={nc['passed']} — CI="
          f"[{nc['ci_low']:+.6f}, {nc['ci_high']:+.6f}], abs_gap={nc['absolute_gap']:.6f} "
          f"vs vartheta={nc['vartheta']} (n={nc['bootstrap']['n_months_common']} months, "
          f"{nc['bootstrap']['n_replicates']} reps)")
    print(f"results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
