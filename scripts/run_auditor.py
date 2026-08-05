"""
scripts/run_auditor.py — real dev-panel driver for the core Auditor (RQ3).

  ./.venv/bin/python scripts/run_auditor.py --anchor mom6 --check core-sync-1
  ./.venv/bin/python scripts/run_auditor.py --anchor all

Runs each anchor's strategy through the deterministic 2^k bias-toggle lattice
(`run_audit` -> `AuditCore`) on `data/development/monthly_panel_maximal.parquet`.
The frozen holdout is NEVER read. No language model appears in the analytical path
(that is a structural property of the core, not a policy).

`--check core-sync-1` asserts the panel-boundary structural zero proven by the
IPCA-differential extension (`docs/auditor/ipca_differential_open_item_O-EXT-1.md`
§5.3): on the WRDS-MMN dev panel the 30-day stale-price mask masks ZERO incremental
bond-months (`view(stale_mask=True)` is bit-identical to `view(stale_mask=False)`
across the full dev view — panel construction subsumes the staleness correction
upstream). Because the lattice consumes the same `panel_view.stale_mask` axis, the
`stale_price` main effect `E_stale` AND the `meas_err × stale_price` interaction
must be EXACTLY zero. A nonzero value means this component's panel view has diverged
from the extension's — a wiring defect to fix before any results circulate.

STATUS — READ BEFORE RUNNING. The anchor -> strategy -> lattice path has never been
exercised on the real panel (the core has only ever run on synthetic panels in
pytest). What must be confirmed/supplied first is documented in
`docs/auditor/core_real_data_run_prerun_notes.md`. This module is import-safe and
unit-tested on synthetic panels; it does not touch real data until `main()` runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor.checks.orchestrator import (  # noqa: E402
    AuditRefused,
    IncompleteLatticeError,
    run_audit,
)
from agents.auditor.ipca_differential.runner import load_dev_inputs  # noqa: E402
from agents.auditor.schemas.audit_core import AuditCore  # noqa: E402
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleFacts  # noqa: E402
from agents.librarian.registries.standing_substitutions import (  # noqa: E402
    STANDING_SUBS_V1_SHA256,
    load_standing_substitutions,
)

# The pre-registration tag stamped on every core AuditCore (README §14; thresholds:auditor).
AUDITOR_PREREG_TAG = "auditor-prereg-2026-07-22"
ANCHORS = ("str", "drf", "mom6")

# The two CORE-SYNC-1 coordinates in the saturated DOE basis.
_STALE = frozenset({"stale_price"})
_MEAS_STALE = frozenset({"meas_err", "stale_price"})

# view(stale_mask=True) is bit-identical to view(stale_mask=False) on this panel, so the
# stale cells are computed from identical return series. The DOE coordinates are therefore
# zero up to float noise (~1e-19 from the Walsh transform's ±1 summation), NOT bit-zero —
# so the check is tolerance-based, not `== 0.0`. Measured on the synthetic no-op panel.
DEFAULT_TOL = 1e-9


# --------------------------------------------------------------------------
# real-data assembly (used by main(); NOT by the synthetic unit tests)
# --------------------------------------------------------------------------

def load_anchor_strategy(anchor_id: str):
    """The runnable strategy the lattice consumes: gold spec -> adapted per-leg calls,
    WITH the pre-registered standing-substitution conventions (contract §6) applied.

    `load_gold_spec` + `adapt_spec` are the same chain the G2 harness uses
    (`evaluation/harness/round_trip.py:81`), and like that harness this passes the real
    standing-subs table. Without it the value-weighted / ex-post-trimmed anchors REFUSE:
    `weighting_base=market_value` -> par via `par_weighting_v1`, and `expost_trim=truncate`
    -> none via `lab_trim_delegation_v1`. The file hash is asserted against the recorded
    pre-registration constant (the §6 temporal bright line) so an absent/edited table is
    caught fail-loud, never silently applied."""
    from agents.librarian.adapter.adapt import adapt_spec
    from evaluation.gold_specs.gold_loader import load_gold_spec

    standing_subs = load_standing_substitutions()
    if not standing_subs.verify_hash(STANDING_SUBS_V1_SHA256):
        raise ValueError(
            "standing-substitutions file hash does not match STANDING_SUBS_V1_SHA256 "
            "(contract §6): refusing to adapt anchors -- the §6 conventions must match the "
            "recorded pre-registration before any real audit run."
        )
    return adapt_spec(load_gold_spec(anchor_id), standing_subs=standing_subs)


def default_anchor_facts(anchor_id: str | None = None) -> list[ToggleFacts]:
    """The ToggleFacts for an anchor: all toggles runnable, with the registered
    `stale_price` no-op hypothesis carried as `expect_no_op` (a hypothesis the
    invariance machinery TESTS post-run — it never prunes the lattice, §3.3/§3.5).

    Per-anchor exception (spec D1 / ADR §5.4): for `str` (DRR-native), `meas_err` is
    `not_applicable` — str's as-published baseline IS DRR's full cleaning (the
    corrected panel), so there is no raw-vs-corr meas_err contrast to measure. The
    coordinate is EXCLUDED from the runnable lattice (str runs 2^4), NOT rendered as
    a zero. `anchor_id=None` keeps every toggle runnable (the synthetic-test default).

    The CORE-SYNC-1 coordinates (`E_stale`, `meas_err × stale_price`) require `meas_err`
    and `stale_price` runnable; for `str` the interaction has no coordinate (see
    `run_anchor`, which reports CORE-SYNC-1 as not-applicable there rather than a fail)."""
    facts: list[ToggleFacts] = []
    for t in TOGGLE_IDS:
        if anchor_id == "str" and t == "meas_err":
            facts.append(ToggleFacts(
                t, runnable=False, not_applicable=True,
                not_applicable_reason=(
                    "str is DRR-native: its as-published baseline IS DRR's full "
                    "cleaning (the corrected panel), so there is no raw-vs-corr "
                    "meas_err contrast. The coordinate is EXCLUDED, not measured "
                    "(ADR §5.4)."
                ),
            ))
        else:
            facts.append(ToggleFacts(t, runnable=True, expect_no_op=(t == "stale_price")))
    return facts


# --------------------------------------------------------------------------
# the audit + the CORE-SYNC-1 verdict (injectable / pure — unit-tested on synthetic panels)
# --------------------------------------------------------------------------

def load_anchor_expost_trim_off(anchor_id: str, thresholds_path: str | Path | None = None):
    """The per-anchor PUBLISHED return trim re-injected on the lab_trim OFF arm
    (spec E). Only mom6 has one — Jostova fn.16's 99.5th right-tail elimination,
    which `lab_trim_delegation_v1` delegated out of the base rulebook. str/drf
    published no return trim, so None (their lab_trim stays a structural zero,
    correctly). Built as jostova_faithful = truncate (delete) at the full-sample
    99.5th right-tail percentile, recomputed per cell; the interpolation method is
    the pre-registered thresholds constant (never a library default — spec E)."""
    if anchor_id != "mom6":
        return None
    import yaml

    from agents.quant.config import TrimRule

    path = Path(thresholds_path) if thresholds_path else (REPO_ROOT / "docs" / "thresholds.yaml")
    lab = yaml.safe_load(path.read_text())["bias_toggles"]["lab_filter"]
    if lab.get("lattice_action") != "truncate":
        raise ValueError(
            f"lab_filter.lattice_action must be 'truncate' (jostova_faithful); "
            f"got {lab.get('lattice_action')!r}"
        )
    if lab.get("loc") != "right":
        raise ValueError(f"lab_filter.loc must be 'right' (one-sided); got {lab.get('loc')!r}")
    level = float(lab["level"]) / 100.0  # 99.5 -> 0.995 (right-tail level)
    return TrimRule(
        method="truncate", hi=level, bounds_type="percentile",
        percentile_method=lab["percentile_interpolation"],
    )


def audit_anchor(
    strategy,
    maximal_panel: pd.DataFrame,
    facts: list[ToggleFacts],
    signals: pd.DataFrame | None,
    *,
    thresholds_path: str | Path | None = None,
    pre_registration_tag: str | None = AUDITOR_PREREG_TAG,
    primary_metric: str | None = None,
    support_gate=None,
    expost_trim_off=None,
) -> AuditCore:
    """Run the deterministic analytical spine for one strategy -> AuditCore. Thin wrapper
    over `run_audit` so the synthetic tests can inject a strategy/panel/facts without the
    real loaders. `primary_metric`/`support_gate` default to None, so a real run reads them
    fail-loud from thresholds; the synthetic tests pass a small gate for short panels.
    `expost_trim_off` re-injects the per-anchor published trim on the lab_trim OFF arm."""
    return run_audit(
        strategy, maximal_panel, facts,
        signals=signals,
        primary_metric=primary_metric,
        support_gate=support_gate,
        expost_trim_off=expost_trim_off,
        pre_registration_tag=pre_registration_tag,
        thresholds_path=thresholds_path,
    )


def core_sync_1_verdict(doe: Mapping[frozenset, float], *, tol: float = DEFAULT_TOL) -> dict:
    """The CORE-SYNC-1 assertion, read off the saturated DOE basis (`core.saturated.doe`).

    Returns the two coordinates, the tolerance, and a boolean `passed`. `passed` requires
    BOTH coordinates present AND within `tol` of zero. A missing coordinate (e.g. the audit
    was refused, or `meas_err`/`stale_price` was not runnable) is a `passed=False` with an
    explicit reason, never a silent skip."""
    e_stale = doe.get(_STALE)
    interaction = doe.get(_MEAS_STALE)
    reasons: list[str] = []
    if e_stale is None:
        reasons.append("stale_price main effect absent (stale_price not runnable?)")
    elif abs(e_stale) > tol:
        reasons.append(f"E_stale={e_stale:.3e} exceeds tol={tol:.1e} (view divergence?)")
    if interaction is None:
        reasons.append("meas_err×stale_price interaction absent (both must be runnable)")
    elif abs(interaction) > tol:
        reasons.append(f"meas_err×stale_price={interaction:.3e} exceeds tol={tol:.1e}")
    passed = not reasons
    return {
        "E_stale": None if e_stale is None else float(e_stale),
        "meas_err×stale_price": None if interaction is None else float(interaction),
        "tol": tol,
        "expected": "structural zero within float tol (panel views bit-identical at stale_mask axis)",
        "passed": passed,
        "failure_reasons": reasons,
    }


def stale_invariance_note(core: AuditCore) -> dict | None:
    """The complementary signal: the `stale_price` invariance verdict, present iff the
    `expect_no_op` hypothesis was carried. `is_no_op=True` corroborates E_stale=0 from the
    membership/returns side. None if the hypothesis was not set for this run."""
    for inv in core.invariance:
        if inv.toggle_id == "stale_price":
            return inv.to_dict()
    return None


# --------------------------------------------------------------------------
# run one anchor end-to-end (real data) + serialise
# --------------------------------------------------------------------------

def run_anchor(
    anchor_id: str,
    maximal_panel: pd.DataFrame,
    signals: pd.DataFrame | None,
    *,
    tol: float = DEFAULT_TOL,
    thresholds_path: str | Path | None = None,
) -> dict:
    """Assemble the anchor, run the audit, and return a serialisable record with the
    AuditCore, the CORE-SYNC-1 verdict, and the stale invariance note. Raises `AuditRefused`
    upward (the caller records it) rather than fabricating a verdict."""
    strategy = load_anchor_strategy(anchor_id)
    facts = default_anchor_facts(anchor_id)
    expost_trim_off = load_anchor_expost_trim_off(anchor_id, thresholds_path)
    core = audit_anchor(
        strategy, maximal_panel, facts, signals,
        thresholds_path=thresholds_path, expost_trim_off=expost_trim_off,
    )
    # CORE-SYNC-1 tests meas_err × stale_price; for an anchor with meas_err
    # EXCLUDED (not_applicable, e.g. str), that interaction has no coordinate on the
    # reduced 2^4 lattice, so the check is NOT-APPLICABLE, never a failure (ADR §5.4).
    meas_err_na = any(f.toggle_id == "meas_err" and f.not_applicable for f in facts)
    if meas_err_na:
        verdict = {
            "applicable": False,
            "passed": True,
            "reason": (
                "meas_err is not_applicable (excluded) for this anchor; CORE-SYNC-1 "
                "(meas_err × stale_price) has no coordinate on the reduced lattice."
            ),
        }
    else:
        verdict = core_sync_1_verdict(core.saturated.doe, tol=tol)
    return {
        "anchor": anchor_id,
        "audit_scope": core.audit_scope,
        "core_sync_1": verdict,
        "stale_invariance": stale_invariance_note(core),
        "core": core.to_dict(),
    }


def _git_short() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _auditor_config_hash() -> str:
    import hashlib

    import yaml

    block = yaml.safe_load((REPO_ROOT / "docs" / "thresholds.yaml").read_text())["auditor"]
    return hashlib.sha256(yaml.safe_dump(block, sort_keys=True).encode()).hexdigest()[:16]


def write_results(out_dir: Path, records: list[dict], run_log: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    (out_dir / "core_sync_1.json").write_text(
        json.dumps(
            {r["anchor"]: r["core_sync_1"] for r in records},
            indent=2, default=str,
        )
    )
    for r in records:
        (out_dir / f"{r['anchor']}_core.json").write_text(
            json.dumps(r["core"], indent=2, default=str)
        )


def run_all(
    anchors: Iterable[str],
    *,
    out_dir: Path | None = None,
    tol: float = DEFAULT_TOL,
) -> int:
    """Load the dev panel once, run each anchor, write results, print the CORE-SYNC-1
    verdicts. Returns a process exit code: 0 iff every requested anchor PASSED CORE-SYNC-1."""
    maximal, signals, _registry = load_dev_inputs()  # holdout never read
    records: list[dict] = []
    refusals: list[dict] = []
    incomplete_lattice: list[dict] = []
    for anchor_id in anchors:
        try:
            records.append(run_anchor(anchor_id, maximal, signals, tol=tol))
        except AuditRefused as exc:
            refusals.append({"anchor": anchor_id, "refused": str(exc)})
        except IncompleteLatticeError as exc:
            # Spec E2b: a non-computable cell -> a TYPED verdict, never an imputed
            # or partial-surface attribution.
            incomplete_lattice.append({
                "anchor": anchor_id,
                "incomplete_lattice": str(exc),
                "non_computable": sorted("|".join(sorted(s)) or "empty"
                                         for s in exc.non_computable),
            })

    run_log = {
        "git_commit": _git_short(),
        "auditor_prereg_tag": AUDITOR_PREREG_TAG,
        "thresholds_auditor_hash": _auditor_config_hash(),
        "standing_subs_sha256": STANDING_SUBS_V1_SHA256,
        "window": "development 2002-2021 (holdout untouched)",
        "check": "core-sync-1",
        "tol": tol,
        "anchors_requested": list(anchors),
        "anchors_run": [r["anchor"] for r in records],
        "refusals": refusals,
        "incomplete_lattice": incomplete_lattice,
    }
    out_dir = out_dir or (REPO_ROOT / "results" / "auditor" / f"run_{_git_short()}")
    write_results(out_dir, records, run_log)

    all_pass = (
        bool(records)
        and all(r["core_sync_1"]["passed"] for r in records)
        and not refusals
        and not incomplete_lattice
    )
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "core_sync_1": {r["anchor"]: r["core_sync_1"] for r in records},
            "refusals": refusals,
            "incomplete_lattice": incomplete_lattice,
            "ALL_PASS": all_pass,
        },
        indent=2, default=str,
    ))
    return 0 if all_pass else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--anchor", choices=(*ANCHORS, "all"), default="all",
        help="which anchor to audit (default: all three)",
    )
    ap.add_argument(
        "--check", choices=("core-sync-1",), default="core-sync-1",
        help="the assertion to apply after the lattice run",
    )
    ap.add_argument("--out", type=Path, default=None, help="output dir (default results/auditor/run_<git>)")
    ap.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"max |DOE coordinate| accepted as the structural zero (default {DEFAULT_TOL:g})",
    )
    args = ap.parse_args()
    anchors = ANCHORS if args.anchor == "all" else (args.anchor,)
    return run_all(anchors, out_dir=args.out, tol=args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
