"""Basis-sensitivity export (RQ2 §5.2) — consolidate the credit-factor (crf) return-basis
evidence into a single citeable results JSON.

Bond returns can be measured on a *clean-price* basis (price change only) or a *total-return*
basis (price + accrued interest + coupon). This driver does NO new computation: it reads the two
already-recorded development headline artefacts —

  * ``data/development/monthly_panel_total_return_report.json`` → the ZERO-COUPON IDENTITY check
    (a coupon-free bond has AI=C=0, so its total return equals its clean return EXACTLY), and
  * ``data/development/headlines/accrual_validation.json`` → the clean-vs-total factor LEVELS —

and consolidates the basis-sensitivity story:

  1. **Zero-coupon identity** — exact, the sanity anchor (`raw_ok`/`corr_ok` over the checked
     zero-coupon bond-months).
  2. **Level sensitivity** — the crf premium FLIPS SIGN between bases (clean-price negative,
     total-return positive); LEVELS are basis-sensitive.
  3. **Differential invariance** — the RQ3 bias GATES are computed on corrected-minus-uncorrected
     differentials, where accrual cancels, so they are basis-INVARIANT (the RQ3 conclusions do not
     depend on the basis).

Deterministic, offline, DEV-ONLY (both inputs are under ``/data/development/``; the holdout is
never read). Emits ``results/rq2_basis_sensitivity.json``.
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

_TR_REPORT = REPO_ROOT / "data" / "development" / "monthly_panel_total_return_report.json"
_ACCRUAL = REPO_ROOT / "data" / "development" / "headlines" / "accrual_validation.json"
_RESULTS_DIR = REPO_ROOT / "results"


class BasisSensitivityError(RuntimeError):
    """A required source artefact is missing or malformed — surfaced loudly, never a silent
    default that would misdescribe the basis-sensitivity evidence."""


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path, what: str) -> dict:
    if not path.is_file():
        raise BasisSensitivityError(
            f"{what} artefact missing: {path} — regenerate it "
            "(build_total_return_panel.py / run_accrual_validation.py) before exporting")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_provenance(path: Path, doc: dict) -> dict:
    return {
        "path": _rel(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_run_timestamp": doc.get("run_timestamp"),
        "source_git_commit": doc.get("git_commit"),
    }


# A crf LEVEL move beyond this (%/mo) makes the basis materially level-sensitive.
_BASIS_SENSITIVE_LEVEL_TOL = 0.01
# Corrected-minus-uncorrected differentials must agree across bases within this relative gap for
# the RQ3 gates to be judged basis-invariant (accrual cancels in a differential).
_BASIS_INVARIANT_REL_TOL = 0.15


def _differential_pair_gaps(diffs: dict) -> dict:
    """Match ``*_clean(_pct)`` / ``*_total(_pct)`` differential pairs and compute the absolute and
    relative gap across bases for each — the evidence ``basis_invariant`` is DERIVED from."""
    gaps: dict = {}
    for key, val in diffs.items():
        for clean_suf, total_suf in (("_clean_pct", "_total_pct"), ("_clean", "_total")):
            if key.endswith(clean_suf):
                tkey = key[: -len(clean_suf)] + total_suf
                tval = diffs.get(tkey)
                if isinstance(val, (int, float)) and isinstance(tval, (int, float)):
                    c, t = float(val), float(tval)
                    denom = max(abs(c), abs(t), 1e-9)
                    gaps[key[: -len(clean_suf)]] = {
                        "clean": c, "total": t, "abs_gap": abs(c - t), "rel_gap": abs(c - t) / denom}
                break
    return gaps


def _finding(crf_clean: float, crf_total: float, sign_flip: bool, identity_exact: bool,
             basis_invariant: bool | None) -> str:
    level = ("flips sign between bases (clean-price negative, total-return positive)" if sign_flip
             else f"differs in level between bases (clean {crf_clean:+.4f} vs total "
                  f"{crf_total:+.4f} %/mo)")
    inv = ("and the corrected-minus-uncorrected bias gates are basis-invariant (accrual cancels), "
           "so the RQ3 conclusions do not depend on the return basis" if basis_invariant
           else "and the differential basis-invariance could not be confirmed from the surfaced pairs")
    ident = "exactly" if identity_exact else "INEXACTLY (check the panel build)"
    return f"The credit factor {level}. The zero-coupon identity holds {ident}, {inv}."


def build_basis_sensitivity(tr_report: dict, accrual: dict) -> dict:
    """Pure consolidation of the two source dicts into the basis-sensitivity result. Every verdict
    boolean is DERIVED from the surfaced numbers (never a hardcoded conclusion). Fail loud if a
    required block is absent or a required value is null — the export never fabricates a reading."""
    zci = tr_report.get("zero_coupon_invariance")
    if not isinstance(zci, dict) or "raw_ok" not in zci or "corr_ok" not in zci:
        raise BasisSensitivityError(
            "total-return report has no usable zero_coupon_invariance block")
    levels = accrual.get("levels_corrected")
    diffs = accrual.get("differentials_static")
    if not isinstance(levels, dict) or levels.get("crf_clean_pct") is None \
            or levels.get("crf_total_pct") is None:
        raise BasisSensitivityError(
            "accrual_validation has no usable levels_corrected.crf_clean_pct / crf_total_pct")
    if not isinstance(diffs, dict):
        raise BasisSensitivityError("accrual_validation has no differentials_static block")

    crf_clean = float(levels["crf_clean_pct"])
    crf_total = float(levels["crf_total_pct"])
    crf_sign_flip = (crf_clean < 0.0) and (crf_total > 0.0)
    crf_level_gap = abs(crf_total - crf_clean)
    # DERIVED, not asserted: a sign flip OR a material level move ⇒ level-sensitive.
    basis_sensitive = bool(crf_sign_flip or crf_level_gap > _BASIS_SENSITIVE_LEVEL_TOL)

    identity_exact = bool(zci["raw_ok"]) and bool(zci["corr_ok"])

    pair_gaps = _differential_pair_gaps(diffs)
    # DERIVED from the surfaced differential pairs; None when no pair is available to judge.
    basis_invariant: bool | None = (
        all(g["rel_gap"] < _BASIS_INVARIANT_REL_TOL for g in pair_gaps.values())
        if pair_gaps else None)

    return {
        "experiment": "basis_sensitivity_crf",
        "finding": _finding(crf_clean, crf_total, crf_sign_flip, identity_exact, basis_invariant),
        "zero_coupon_identity": {
            "raw_ok": bool(zci["raw_ok"]),
            "corr_ok": bool(zci["corr_ok"]),
            "z_bond_months_checked": zci.get("z_bond_months_checked"),
            "exact": identity_exact,
            "note": ("a coupon-free bond has AI=C=0, so total return == clean return byte-for-byte "
                     "(atol 1e-12) — the basis sanity anchor"),
        },
        "level_sensitivity": {
            "crf_clean_pct": crf_clean,
            "crf_total_pct": crf_total,
            "crf_sign_flip": crf_sign_flip,
            "crf_level_gap_pct": crf_level_gap,
            "lrf_clean_pct": levels.get("lrf_clean_pct"),
            "lrf_total_pct": levels.get("lrf_total_pct"),
            "drf_clean_pct": levels.get("drf_clean_pct"),
            "drf_total_pct": levels.get("drf_total_pct"),
            "mktb_clean_pct": levels.get("mktb_clean_pct"),
            "mktb_total_pct": levels.get("mktb_total_pct"),
            "criterion": levels.get("criterion"),
            "basis_sensitive": basis_sensitive,             # DERIVED
        },
        "differential_invariance": {
            "mom6_ep_minus_ea_clean_pct": diffs.get("mom6_ep_minus_ea_clean_pct"),
            "mom6_ep_minus_ea_total_pct": diffs.get("mom6_ep_minus_ea_total_pct"),
            "leadlag_drf_corr_clean": diffs.get("leadlag_drf_corr_clean"),
            "leadlag_drf_corr_total": diffs.get("leadlag_drf_corr_total"),
            "criterion": diffs.get("criterion"),
            "pair_gaps": pair_gaps,
            "rel_tol": _BASIS_INVARIANT_REL_TOL,
            "basis_invariant": basis_invariant,             # DERIVED (None if unjudgeable)
        },
        "day_count_fallback": accrual.get("fallback_exposure", {}),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tr-report", default=str(_TR_REPORT))
    ap.add_argument("--accrual", default=str(_ACCRUAL))
    ap.add_argument("--out", default=None, help="output path (default results/rq2_basis_sensitivity.json)")
    args = ap.parse_args(argv)

    tr_path, accrual_path = Path(args.tr_report), Path(args.accrual)
    tr_report = _read_json(tr_path, "total-return report")
    accrual = _read_json(accrual_path, "accrual-validation")

    result = build_basis_sensitivity(tr_report, accrual)
    stamp = datetime.now(timezone.utc)
    result["provenance"] = {
        "exporter": "scripts/run_basis_sensitivity.py",
        "run_timestamp": stamp.isoformat(),
        "git_commit": _git_commit(),
        "dev_only": True,
        "model_calls": 0,
        "spend_usd": 0.0,
        "sources": [
            _source_provenance(tr_path, tr_report),
            _source_provenance(accrual_path, accrual),
        ],
    }

    out = Path(args.out) if args.out else _RESULTS_DIR / "rq2_basis_sensitivity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")

    lv = result["level_sensitivity"]
    print(f"basis-sensitivity: crf clean={lv['crf_clean_pct']:+.4f}%/mo → total={lv['crf_total_pct']:+.4f}%/mo "
          f"(sign flip={lv['crf_sign_flip']}); zero-coupon identity exact="
          f"{result['zero_coupon_identity']['exact']}")
    print(f"results written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
