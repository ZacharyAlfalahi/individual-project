"""
scripts/run_full_audit.py — real dev-panel driver for the FULL per-strategy Auditor
report (RQ3 items 5+7): the full analytical stack (spine + bootstrap + inference/FDR/
Bayes/compression/economic) plus the DETERMINISTIC, LLM-free explainer.

  ./.venv/bin/python scripts/run_full_audit.py --anchor mom6
  ./.venv/bin/python scripts/run_full_audit.py --anchor all

This is the sibling of `scripts/run_auditor.py` (which runs only the core lattice +
CORE-SYNC-1). Where that driver stops at the deterministic spine, this one assembles
the full `AuditReport` via `run_full_audit` and renders a number-faithful prose
report via the deterministic `render_report` — NO language model is called anywhere
in this path (the LLM explainer's `explain()` itself falls back to `render_report`,
so this is the guaranteed verifier-safe path). Every rendered number is asserted
against the typed report by `verify_numbers` before it is written.

  Development panel ONLY (2002-2021). The frozen holdout is NEVER read.

DSR PRE-REGISTRATION GATE (fail-loud, by design). `run_full_audit` needs a per-anchor
`n_trials`/`sr_std` — the deflated-Sharpe inputs (O-A4, design §9.1). Those are a
PRE-REGISTRATION decision that is NOT yet ratified/committed: a proposal sits at
`docs/auditor/confirmatory_prereg_proposal.md`, whose ratified home is
`thresholds.yaml` -> `auditor.dsr`. Until that block is committed and git-tagged,
this driver REFUSES to run (`DsrPreRegistrationAbsent`). Running it TODAY is expected
to refuse — that is the whole point: the researcher-degree-of-freedom the project
exists to expose is enforced in code, not left to discipline.

STRUCTURE. The pure, injectable functions (`audit_anchor_full`, `render_verified_prose`,
`load_anchor_dsr`) are unit-testable on synthetic scenarios with INJECTED DSR inputs
without touching real data — exactly as `run_auditor.py` separates `audit_anchor`/
`core_sync_1_verdict` from `main`. `main()` is the only path that reads the real panel,
and only after the DSR gate has passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.auditor import (  # noqa: E402
    AuditorConfig,
    AuditReport,
    ToggleFacts,
    render_report,
    run_full_audit,
    verify_numbers,
)
from agents.auditor.explainer.numeric_verifier import VerificationResult  # noqa: E402
from agents.auditor.ipca_differential.runner import load_dev_inputs  # noqa: E402
from agents.auditor.thresholds import (  # noqa: E402
    AuditorThresholdError,
    _auditor_block,
    _require,
    _require_int,
    _require_number,
)

# Reused READ-ONLY from run_auditor (do NOT edit that module): the anchor->strategy
# chain, the default all-runnable ToggleFacts, the pre-registration tag, the anchor
# set, the standing-subs hash, and the git/thresholds provenance helpers — so this
# driver's provenance record mirrors the core driver's exactly.
from scripts.run_auditor import (  # noqa: E402
    ANCHORS,
    AUDITOR_PREREG_TAG,
    STANDING_SUBS_V1_SHA256,
    _auditor_config_hash,
    _git_short,
    default_anchor_facts,
    load_anchor_strategy,
)

_DSR_SECTION = "§9.1 (O-A4 deflated-Sharpe inputs)"

# The exact pre-registration instruction surfaced when auditor.dsr is not yet committed. The
# gate refuses rather than defaulting: a default n_trials/sr_std would silently launder
# a post-hoc deflation constant, the precise sin the Auditor exists to expose (§13.2).
_DSR_ABSENT_MESSAGE = (
    "auditor.dsr pre-registration absent — ratify "
    "docs/auditor/confirmatory_prereg_proposal.md, commit auditor.dsr, "
    "git-tag before the confirmatory run"
)


class DsrPreRegistrationAbsent(AuditorThresholdError):
    """The DSR pre-registration gate (O-A4), enforced IN CODE. The per-anchor
    deflated-Sharpe inputs (`n_trials`, `sr_std`; design §9.1) are a pre-registration
    decision that is NOT yet ratified/committed, so the confirmatory full audit
    REFUSES to run. A subclass of `AuditorThresholdError` so it is caught by the same
    fail-loud handling, but it carries the exact ratify/commit/tag instruction instead
    of the generic 'missing constant' template."""

    def __init__(self, message: str = _DSR_ABSENT_MESSAGE) -> None:
        self.message = message
        # Bypass the parent's templated message; store the bare instruction as the
        # single arg (KeyError.__str__ quote-wraps it — main() prints `.message` clean).
        KeyError.__init__(self, message)


# --------------------------------------------------------------------------
# DSR pre-registration gate (fail-loud) — the O-A4 deflated-Sharpe inputs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorDsr:
    """The ratified per-anchor deflated-Sharpe inputs (O-A4). `n_trials` is the
    strategy-level discovery count (identical across the 2^k cells, NEVER 2^k);
    `sr_std` is the PER-PERIOD (monthly) cross-trial Sharpe SD (the unit is
    load-bearing — see `checks.economic.run_economic`)."""

    n_trials: int
    sr_std: float


def _require_dsr_block(block: dict) -> dict:
    """Return `auditor.dsr` or REFUSE loud if the whole block is absent. This is the
    DSR pre-registration gate: without a ratified+committed `auditor.dsr`, no confirmatory
    full audit may proceed."""
    dsr = block.get("dsr")
    if not isinstance(dsr, dict) or not dsr:
        raise DsrPreRegistrationAbsent()
    return dsr


def load_anchor_dsr(anchor_id: str, path: str | Path | None = None) -> AnchorDsr:
    """Read the ratified per-anchor DSR inputs (O-A4, §9.1) fail-loud, in the
    `agents/auditor/thresholds.py` style. Three distinct refusals:
      - the whole `auditor.dsr` block absent -> `DsrPreRegistrationAbsent` (the gate);
      - this anchor (or its `n_trials`/`sr_std`) absent -> `AuditorThresholdError`
        (a partial pre-registration is still a refusal, never a default);
      - present-but-wrong-type -> `AuditorThresholdError`."""
    block = _auditor_block(path)
    _require_dsr_block(block)  # gate first: the whole block must exist
    n_trials = _require(block, ("dsr", anchor_id, "n_trials"), _DSR_SECTION)
    sr_std = _require(block, ("dsr", anchor_id, "sr_std"), _DSR_SECTION)
    return AnchorDsr(
        n_trials=_require_int(n_trials, f"auditor.dsr.{anchor_id}.n_trials", _DSR_SECTION),
        sr_std=_require_number(sr_std, f"auditor.dsr.{anchor_id}.sr_std", _DSR_SECTION),
    )


def load_dsr_for_anchors(
    anchors: Iterable[str], path: str | Path | None = None
) -> dict[str, AnchorDsr]:
    """Load the DSR inputs for every requested anchor up front, so the gate fires ONCE
    before any real data is touched. Refuses (fail-loud) if the block or any anchor's
    inputs are absent."""
    return {a: load_anchor_dsr(a, path) for a in anchors}


# --------------------------------------------------------------------------
# The per-anchor audit + deterministic render (pure / injectable — unit-tested on
# synthetic scenarios with INJECTED DSR inputs; never touches real data)
# --------------------------------------------------------------------------

def audit_anchor_full(
    strategy,
    maximal_panel: pd.DataFrame,
    facts: list[ToggleFacts],
    config: AuditorConfig,
    *,
    signals: pd.DataFrame | None,
    n_trials: int,
    sr_std: float,
    seed: int = 0,
    pre_registration_tag: str | None = AUDITOR_PREREG_TAG,
) -> AuditReport:
    """Run the full per-strategy audit -> AuditReport. Thin wrapper over
    `run_full_audit` so the synthetic tests can inject a strategy/panel/facts and the
    DSR inputs (`n_trials`/`sr_std`) without the real loaders, mirroring
    `test_auditor_report.py`."""
    return run_full_audit(
        strategy, maximal_panel, facts, config,
        signals=signals,
        n_trials=n_trials,
        sr_std=sr_std,
        seed=seed,
        pre_registration_tag=pre_registration_tag,
    )


class RenderVerificationError(RuntimeError):
    """The deterministic renderer emitted a number that does not trace to the typed
    report. This must NEVER happen (`render_report` is verifier-safe by construction),
    so it is fail-loud, not a warning — a divergence would mean the renderer/report
    contract has broken."""


def render_verified_prose(report: AuditReport) -> tuple[str, VerificationResult]:
    """Render the deterministic (LLM-free) prose and ASSERT `verify_numbers` passes.
    This is the guaranteed-safe path: `render_report` cites only typed report numbers,
    so verification holds by construction; a failure here is a broken invariant and is
    raised, never swallowed."""
    prose = render_report(report)
    result = verify_numbers(prose, report.to_dict())
    if not result.ok:
        raise RenderVerificationError(
            "deterministic render emitted untraceable numbers "
            f"(a broken renderer/report invariant): {result.unverified}"
        )
    return prose, result


# --------------------------------------------------------------------------
# Run one anchor end-to-end (real data) + serialise
# --------------------------------------------------------------------------

def run_anchor_full(
    anchor_id: str,
    maximal_panel: pd.DataFrame,
    signals: pd.DataFrame | None,
    config: AuditorConfig,
    dsr: AnchorDsr,
    *,
    seed: int = 0,
    pre_registration_tag: str | None = AUDITOR_PREREG_TAG,
) -> dict:
    """Assemble the anchor, run the full audit with its ratified DSR inputs, render the
    verified prose, and return a serialisable record (the AuditReport dict, the prose,
    and the numeric-verification result)."""
    strategy = load_anchor_strategy(anchor_id)
    facts = default_anchor_facts()
    report = audit_anchor_full(
        strategy, maximal_panel, facts, config,
        signals=signals, n_trials=dsr.n_trials, sr_std=dsr.sr_std,
        seed=seed, pre_registration_tag=pre_registration_tag,
    )
    prose, verification = render_verified_prose(report)
    rd = report.to_dict()
    return {
        "anchor": anchor_id,
        "audit_scope": rd["audit_scope"],
        "dsr_inputs": {"n_trials": dsr.n_trials, "sr_std": dsr.sr_std},
        "report": rd,
        "prose": prose,
        "numeric_verification": {
            "ok": verification.ok,
            "n_checked": verification.n_checked,
            "unverified": list(verification.unverified),
        },
    }


def write_results(out_dir: Path, records: list[dict], run_log: dict) -> None:
    """Mirror `run_auditor.write_results`: run_log + per-anchor report JSON + per-anchor
    deterministic prose + a numeric-verification summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2, default=str))
    (out_dir / "numeric_verification.json").write_text(
        json.dumps(
            {r["anchor"]: r["numeric_verification"] for r in records},
            indent=2, default=str,
        )
    )
    for r in records:
        a = r["anchor"]
        (out_dir / f"{a}_report.json").write_text(
            json.dumps(r["report"], indent=2, default=str)
        )
        (out_dir / f"{a}_explanation.txt").write_text(r["prose"] + "\n")


def run_all(
    anchors: Iterable[str],
    *,
    out_dir: Path | None = None,
    thresholds_path: str | Path | None = None,
) -> int:
    """The confirmatory driver: DSR gate -> load dev panel -> full audit per anchor ->
    deterministic verified prose -> write. Returns 0 iff every requested anchor produced
    a numerically-verified report. Refuses (fail-loud) at the DSR gate before touching
    real data if `auditor.dsr` is not yet ratified/committed."""
    anchors = list(anchors)

    # Config from thresholds (loads fully from YAML today). This does NOT need the DSR
    # block — that is a separate, not-yet-committed pre-registration.
    config = AuditorConfig.from_thresholds(thresholds_path)

    # DSR gate FIRST — before any real data is read. Refuses today (auditor.dsr absent).
    dsr_map = load_dsr_for_anchors(anchors, thresholds_path)

    maximal, signals, _registry = load_dev_inputs()  # holdout never read
    records = [
        run_anchor_full(a, maximal, signals, config, dsr_map[a]) for a in anchors
    ]

    run_log = {
        "git_commit": _git_short(),
        "auditor_prereg_tag": AUDITOR_PREREG_TAG,
        "thresholds_auditor_hash": _auditor_config_hash(),
        "standing_subs_sha256": STANDING_SUBS_V1_SHA256,
        "window": "development 2002-2021 (holdout untouched)",
        "run": "full-audit (spine + bootstrap + inference/FDR/Bayes/compression/economic)",
        "explainer": "deterministic render_report (NO LLM); verify_numbers asserted",
        "dsr_inputs": {a: dsr_map[a].__dict__ for a in anchors},
        "anchors": [r["anchor"] for r in records],
    }
    out_dir = out_dir or (REPO_ROOT / "results" / "auditor" / f"full_run_{_git_short()}")
    write_results(out_dir, records, run_log)

    all_ok = bool(records) and all(r["numeric_verification"]["ok"] for r in records)
    print(json.dumps(
        {
            "out_dir": str(out_dir),
            "anchors": {
                r["anchor"]: {
                    "audit_scope": r["audit_scope"],
                    "numeric_verification": r["numeric_verification"],
                }
                for r in records
            },
            "ALL_VERIFIED": all_ok,
        },
        indent=2, default=str,
    ))
    return 0 if all_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        "--anchor", choices=(*ANCHORS, "all"), default="all",
        help="which anchor to fully audit (default: all three)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="output dir (default results/auditor/full_run_<git>)",
    )
    args = ap.parse_args()
    anchors = ANCHORS if args.anchor == "all" else (args.anchor,)

    try:
        return run_all(anchors, out_dir=args.out)
    except DsrPreRegistrationAbsent as exc:
        # The DSR pre-registration gate. Refuse loud with the exact instruction, nonzero
        # exit, and NO real-data read having occurred (the gate fires before load).
        print(
            "REFUSED (DSR pre-registration gate, O-A4): " + exc.message,
            file=sys.stderr,
        )
        return 2
    except AuditorThresholdError as exc:
        # A partial/malformed auditor.dsr (or any other missing auditor constant).
        print("REFUSED (auditor pre-registration incomplete): " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
