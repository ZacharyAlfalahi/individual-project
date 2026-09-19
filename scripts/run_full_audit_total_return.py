"""
Full RQ3 audit on the TOTAL-RETURN substrate (the default-flat total-return panels).

Identical audit machinery to `scripts/run_full_audit.py` (DSR gate → per-anchor
`run_anchor_full` → deterministic verified prose → `write_results`); the ONLY difference is
the base panel. Instead of `load_dev_inputs()`'s clean-price maximal, it loads a total-return
maximal panel (`--panel`) left-joined with the total-return per-paper profiles (`--profiles`),
exactly as `load_dev_inputs` joins the clean profiles. The audit checks, bootstrap, inference/
FDR/Bayes/compression/economic, and the numeric verifier are reused byte-for-byte — so the
total-return report differs from the clean report only through the panel's return leg.

The default base panel is the default-flat total-return panel, in which defaulted bonds trade flat
(AI=C=0 from the default month — `build_total_return_panel.py`).

Development panel only; the holdout is never read.

Usage:
  python scripts/run_full_audit_total_return.py \
      --panel data/development/monthly_panel_total_return_default_flat.parquet \
      --profiles data/development/monthly_panel_profiles_total_return_default_flat.parquet \
      --out results/auditor/full_run_total_return_default_flat
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor import AuditorConfig  # noqa: E402
from agents.auditor.ipca_differential.runner import load_dev_signals, load_registry  # noqa: E402
from scripts.run_auditor import ANCHORS, AUDITOR_PREREG_TAG  # noqa: E402
from scripts.run_full_audit import (  # noqa: E402
    DsrPreRegistrationAbsent,
    load_dsr_for_anchors,
    run_anchor_full,
    write_results,
)

DEV = REPO_ROOT / "data" / "development"
DEFAULT_PANEL = DEV / "monthly_panel_total_return_default_flat.parquet"
DEFAULT_PROFILES = DEV / "monthly_panel_profiles_total_return_default_flat.parquet"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_total_return_inputs(panel_path: Path, profiles_path: Path | None):
    """(maximal_tr, signals, registry) — the total-return analogue of
    `ipca_differential.runner.load_dev_inputs`. The TR maximal carries the same
    raw/corr/xret columns (on a total-return basis) and is LEFT-JOINED with the
    TR profile families (`*_bbw_2019` / `*_jostova_2013`) exactly as the clean loader
    joins the clean profiles. Holdout is never read."""
    if not panel_path.is_file():
        raise SystemExit(f"ERROR: total-return panel not found: {panel_path}")
    maximal = pd.read_parquet(panel_path)
    if profiles_path is not None:
        if not profiles_path.is_file():
            raise SystemExit(f"ERROR: total-return profiles not found: {profiles_path}")
        maximal = maximal.merge(pd.read_parquet(profiles_path), on=["cusip", "date"], how="left")
    return maximal, load_dev_signals(), load_registry()


def run_all_total_return(
    anchors,
    *,
    panel_path: Path,
    profiles_path: Path | None,
    out_dir: Path | None = None,
    thresholds_path=None,
) -> int:
    """Mirror `run_full_audit.run_all`, differing only in the base panel. Returns 0 iff every
    requested anchor produced a numerically-verified report."""
    anchors = list(anchors)
    config = AuditorConfig.from_thresholds(thresholds_path)
    dsr_map = load_dsr_for_anchors(anchors, thresholds_path)  # gate first, before any read

    maximal, signals, _registry = load_total_return_inputs(panel_path, profiles_path)
    records = [
        run_anchor_full(a, maximal, signals, config, dsr_map[a], thresholds_path=thresholds_path)
        for a in anchors
    ]

    run_log = {
        "auditor_prereg_tag": AUDITOR_PREREG_TAG,
        "window": "development 2002-2021 (holdout untouched)",
        "run": "full-audit (spine + bootstrap + inference/FDR/Bayes/compression/economic)",
        "basis": "total_return",
        "base_panel": str(panel_path.relative_to(REPO_ROOT)),
        "base_panel_sha256": _sha256(panel_path),
        "base_profiles": (str(profiles_path.relative_to(REPO_ROOT)) if profiles_path else None),
        "base_profiles_sha256": (_sha256(profiles_path) if profiles_path else None),
        "note": ("RQ3 audit on the TOTAL-RETURN basis with the DEFAULT-FLAT correction "
                 "(defaulted bonds trade flat: AI=C=0 from the default month). Audit "
                 "machinery unchanged except the base panel."),
        "dsr_inputs": {a: dsr_map[a].__dict__ for a in anchors},
        "baseline_signatures": {r["anchor"]: r["baseline_signature"] for r in records},
        "anchors": [r["anchor"] for r in records],
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = out_dir or (REPO_ROOT / "results" / "auditor" / "full_run_total_return_default_flat")
    write_results(out_dir, records, run_log)

    all_ok = bool(records) and all(r["numeric_verification"]["ok"] for r in records)
    print(json.dumps({
        "out_dir": str(out_dir),
        "basis": "total_return",
        "base_panel_sha256": run_log["base_panel_sha256"],
        "anchors": {r["anchor"]: {"audit_scope": r["audit_scope"],
                                  "numeric_verification": r["numeric_verification"]} for r in records},
        "ALL_VERIFIED": all_ok,
    }, indent=2, default=str))
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--anchor", choices=(*ANCHORS, "all"), default="all")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL,
                    help="total-return maximal panel (default: the default-flat corrected panel)")
    ap.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES,
                    help="total-return per-paper profiles panel (pass 'none' to skip the join)")
    ap.add_argument("--out", type=Path, default=None, help="output dir")
    args = ap.parse_args()
    anchors = ANCHORS if args.anchor == "all" else (args.anchor,)
    profiles = None if str(args.profiles).lower() == "none" else args.profiles

    try:
        return run_all_total_return(anchors, panel_path=args.panel, profiles_path=profiles, out_dir=args.out)
    except DsrPreRegistrationAbsent as exc:
        print("REFUSED (DSR pre-registration gate, O-A4): " + exc.message, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
