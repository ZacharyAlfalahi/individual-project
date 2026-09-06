"""RQ3 controls export — surface the two instrument-specificity controls into one
citeable results JSON.

Two controls, one honest report:

  * **Positive (known-error) control** — REAL dev-data. BBW carries a documented look-ahead/lead
    defect; the integrated pipeline must reproduce it (as-published correlation collapses) and its
    correction must fix it (round-trip correlation restores). This grades the committed
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

Deterministic, offline, DEV-ONLY, $0. Emits ``results/auditor/rq3_controls_<date>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from agents.auditor.validation.known_error_control import (  # noqa: E402
    evaluate_known_error_control,
)
from agents.auditor.validation.negative_control_run import (  # noqa: E402
    run_negative_control_from_dev,
)

_THRESHOLDS = REPO_ROOT / "docs" / "thresholds.yaml"
_LEADLAG = REPO_ROOT / "data" / "development" / "headlines" / "leadlag_gate.json"
_BBW_FACTORS = REPO_ROOT / "data" / "development" / "factors" / "bbw_factors.parquet"
_RESULTS_DIR = REPO_ROOT / "results" / "auditor"

_NEG_CONTROL_TEST = "tests/unit/test_negative_control_gate.py"


class RQ3ControlsError(RuntimeError):
    """A required control artefact is missing/malformed — surfaced loudly, never a silent pass."""


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


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
        "gate_report_git_commit": gate_report.get("git_commit"),
    }
    return out


def total_return_sensitivity() -> dict | None:
    """A pre-declared SENSITIVITY cross-check: the point mean of the LRF corrected-minus-uncorrected
    differential on the committed TOTAL-RETURN factor series (``bbw_factors.parquet``), beside the
    confirmatory maximal-basis number. Item-10's basis-invariance-of-differentials finding predicts
    they agree. Returns ``None`` if the artefact is absent (a cross-check, not a gate)."""
    if not _BBW_FACTORS.is_file():
        return None
    import pandas as pd
    df = pd.read_parquet(_BBW_FACTORS)
    diff = (df["lrf_corr"] - df["lrf_raw"]).dropna()
    return {"basis": "total_return", "point_gap_mean": float(diff.mean()), "n_months": int(len(diff)),
            "note": "point cross-check only (no CI); the confirmatory interval is the maximal-basis run"}


def negative_control(*, thresholds_path: Path | None = None, seed: int = 0) -> dict:
    """The REAL dev-data negative-control specificity result: LRF corrected-vs-uncorrected differential
    (meas_err / price-cleaning) through the pre-registered confirmatory block bootstrap, graded by the
    separated-mode gate. Run ONCE — the CI + PASS/FAIL is reported as-is. Adds the unit-verified note
    and the total-return sensitivity cross-check."""
    result = run_negative_control_from_dev(seed=seed, thresholds_path=thresholds_path)
    result["unit_verified"] = {
        "gate_test": _NEG_CONTROL_TEST,                       # the pure separated-mode gate
        "run_path_test": "tests/unit/test_negative_control_run.py",  # the differential/bootstrap/dev wiring
    }
    result["sensitivity_total_return"] = total_return_sensitivity()
    return result


def build_controls(
    gate_report: dict,
    *,
    thresholds_path: Path | None = None,
    negative_control_fn=None,
) -> dict:
    """Assemble both controls. ``negative_control_fn`` is injectable so tests can supply a fixture
    instead of running the ~13s dev-data computation; when omitted it resolves to the module-level
    ``negative_control`` at call time (so a monkeypatch on it is honoured)."""
    neg_fn = negative_control_fn or negative_control
    return {
        "experiment": "rq3_instrument_controls",
        "positive_control": positive_control(gate_report, thresholds_path=thresholds_path),
        "negative_control": neg_fn(thresholds_path=thresholds_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--leadlag", default=str(_LEADLAG))
    ap.add_argument("--out", default=None,
                    help="output path (default results/auditor/rq3_controls_<date>.json)")
    args = ap.parse_args(argv)

    leadlag_path = Path(args.leadlag)
    if not leadlag_path.is_file():
        raise RQ3ControlsError(
            f"lead/lag gate report missing: {leadlag_path} — regenerate it "
            "(./.venv/bin/python scripts/run_leadlag_gate.py) before exporting")
    gate_report = json.loads(leadlag_path.read_text(encoding="utf-8"))

    result = build_controls(gate_report)
    stamp = datetime.now(timezone.utc)
    result["provenance"] = {
        "exporter": "scripts/run_rq3_controls.py",
        "run_timestamp": stamp.isoformat(),
        "git_commit": _git_commit(),
        "dev_only": True,
        "model_calls": 0,
        "spend_usd": 0.0,
        "leadlag_gate_sha256": hashlib.sha256(leadlag_path.read_bytes()).hexdigest(),
    }

    out = Path(args.out) if args.out else _RESULTS_DIR / "rq3_controls.json"
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
