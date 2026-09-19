"""
G3 phase gate and harness config (evaluation contract §1, D33, D34/D37).

Two jobs, both of the same kind: **config the G3 harness must read and must never
silently default**.

**1. Reportability.** Contract §1 splits the model policy into two phases:
Phase D (free development pair, "**No Phase-D number enters the project**") and
Phase F (paid anchor, the only source of reported RQ1 figures). It also states
that calibration is pair-specific -- ``P(correct|agree)`` and every selective
metric measured on Phase D do **not** transfer.

Without a code-level check that firewall is documentary: `--phase report` differs from
`--phase dev` by a single YAML key, the D33 authorization is a process step with no
code behind it, and a banner at the top of a baseline document is not enforcement.
This module makes it mechanical: a run's phase is DERIVED
from the model ids its trace header records, compared against the pinned stack,
and the resulting stamp is a mandatory field that renderers refuse to drop.

Deriving it from the header (what actually answered) rather than from the CLI
flag (what was requested) is the point -- the header carries the provider-returned
version string, so a run whose SKUs silently drifted is not reportable no matter
what flag produced it.

**2. Thresholds.** ``librarian.g3`` carries D34's ``min_cell_n`` bar and the
Wilson ``z``. No threshold is hard-coded in agent code; the loader
raises when the block is absent rather than defaulting, mirroring
``load_verified_standing_subs`` on the G2 side. A calibration run against
silently-defaulted thresholds is a number nobody chose.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# A provider version stamp: a separator then digits only (e.g. "-20260215").
# Deliberately NOT a general suffix -- see _sku_matches.
_SNAPSHOT_SUFFIX_RE = re.compile(r"^[-_@][0-9]+$")

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_THRESHOLDS_PATH = _REPO_ROOT / "docs" / "thresholds.yaml"

# Header keys that must be present and non-empty for a run to be reportable at
# all. Contract §1: "provider-returned model version logged per call; raw
# responses archived; prompt/schema/decoding hashes stamped." A run missing its
# provenance cannot be reproduced, whatever pair produced it.
_REQUIRED_PROVENANCE = (
    "paper_id", "registry_hash", "canonical_text_hash",
    "silence_table_version", "prompt_template_hashes", "timestamp",
)


class ReportabilityError(RuntimeError):
    """The phase gate could not be evaluated, or a reportable-only operation was
    attempted on a non-reportable run."""


@dataclass(frozen=True)
class G3Thresholds:
    min_cell_n: int
    wilson_z: float
    interval_label: str


def load_g3_thresholds(path: str | Path | None = None) -> G3Thresholds:
    """Load ``librarian.g3``. Raises if the block is absent -- never defaults."""
    p = Path(path) if path is not None else _THRESHOLDS_PATH
    if not p.exists():
        raise ReportabilityError(f"thresholds file not found at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    block = (raw.get("librarian") or {}).get("g3")
    if not isinstance(block, dict):
        raise ReportabilityError(
            f"{p} carries no librarian.g3 block -- refusing to run G3 against defaulted "
            "thresholds (D34's min_cell_n bar and the Wilson z are pre-registered values, "
            "not code constants)"
        )
    for key in ("min_cell_n", "wilson_z", "interval_label"):
        if key not in block:
            raise ReportabilityError(f"librarian.g3 is missing {key!r}")
    return G3Thresholds(
        min_cell_n=int(block["min_cell_n"]),
        wilson_z=float(block["wilson_z"]),
        interval_label=str(block["interval_label"]),
    )


def load_model_stack(path: str | Path | None = None) -> dict:
    """The ``librarian.model_stack`` block (the pinned Phase-D / Phase-F pairs)."""
    p = Path(path) if path is not None else _THRESHOLDS_PATH
    if not p.exists():
        raise ReportabilityError(f"thresholds file not found at {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    stack = (raw.get("librarian") or {}).get("model_stack")
    if not isinstance(stack, dict):
        raise ReportabilityError(f"{p} carries no librarian.model_stack block")
    return stack


@dataclass(frozen=True)
class Reportability:
    """A run's phase classification. Carried as a MANDATORY field on anything
    downstream that renders a number -- there is no default, so a bundle cannot be
    constructed without deciding this."""

    phase: str | None            # "phase_d" | "phase_f" | None (unrecognised)
    reportable: bool
    reason: str
    model_a_id: str
    model_b_id: str
    run_id: str | None = None
    git_dirty: bool | None = None

    @property
    def banner(self) -> str:
        """The line every renderer prefixes when this run is not reportable."""
        if self.reportable:
            return ""
        phase = self.phase or "unrecognised pair"
        return f"*** NON-REPORTABLE ({phase}, contract §1) -- {self.reason} ***"


def _sku_matches(header_id: str, configured_id: str) -> bool:
    """Exact, or the header id is a dated SNAPSHOT of the configured alias.

    Contract §1 anticipates alias drift ("aliases drift"), and a provider may
    return ``claude-sonnet-4-6-20260215`` for a configured ``claude-sonnet-4-6``.

    A bare ``startswith`` is NOT good enough, and getting this wrong defeats the
    whole check: ``gemini-3.5-flash-lite`` starts with ``gemini-3.5-flash``, so a
    prefix rule would silently accept a different SKU -- with a different free-tier
    budget and different behaviour -- as a snapshot of the pinned one. The suffix
    must therefore look like a version stamp: a separator followed by digits only.
    ``-lite``, ``-preview`` and ``-exp`` are rejected."""
    if not header_id or not configured_id:
        return False
    if header_id == configured_id:
        return True
    if not header_id.startswith(configured_id):
        return False
    suffix = header_id[len(configured_id):]
    return bool(_SNAPSHOT_SUFFIX_RE.match(suffix))


def _phase_pair(stack: dict, phase: str) -> tuple[str, str]:
    block = stack.get(phase) or {}
    return (
        str((block.get("model_a") or {}).get("model_id", "")),
        str((block.get("model_b") or {}).get("model_id", "")),
    )


def classify_phase(header: dict, stack: dict | None = None) -> Reportability:
    """Derive a run's phase from the model ids its trace header records.

    The comparison is an UNORDERED pair: which vendor happens to be model_a is a
    configuration detail, not an identity. An unrecognised pair FAILS CLOSED --
    a run produced by SKUs nobody pinned is not reportable, because its
    calibration transfers from nothing."""
    stack = stack if stack is not None else load_model_stack()
    a = str(header.get("model_a_id") or "")
    b = str(header.get("model_b_id") or "")
    run_id = header.get("run_id")

    missing = [k for k in _REQUIRED_PROVENANCE if not header.get(k)]

    phase = None
    for candidate in ("phase_d", "phase_f"):
        ca, cb = _phase_pair(stack, candidate)
        if not ca or not cb:
            continue
        if ((_sku_matches(a, ca) and _sku_matches(b, cb))
                or (_sku_matches(a, cb) and _sku_matches(b, ca))):
            phase = candidate
            break

    if phase is None:
        return Reportability(
            phase=None, reportable=False,
            reason=(f"model pair ({a or '?'}, {b or '?'}) matches neither the pinned phase_d "
                    "nor phase_f pair; a run from unpinned SKUs has no calibration to transfer"),
            model_a_id=a, model_b_id=b, run_id=run_id,
        )

    if phase == "phase_d":
        return Reportability(
            phase="phase_d", reportable=False,
            reason=("Phase-D free development pair; contract §1: no Phase-D number enters the "
                    "project, and pair-specific calibration does not transfer to Phase F"),
            model_a_id=a, model_b_id=b, run_id=run_id,
        )

    if missing:
        return Reportability(
            phase="phase_f", reportable=False,
            reason=f"Phase-F pair but the trace header is missing provenance: {missing}",
            model_a_id=a, model_b_id=b, run_id=run_id,
        )

    # git_dirty is a contract §1 Phase-F requirement ("One tagged commit;
    # git_dirty = false") that NEITHER the spec nor the trace header records. It
    # is reported as unverifiable rather than assumed clean -- a gap G3 surfaces
    # instead of papering over.
    return Reportability(
        phase="phase_f", reportable=True,
        reason="Phase-F pair with complete provenance",
        model_a_id=a, model_b_id=b, run_id=run_id, git_dirty=None,
    )


def require_reportable(rep: Reportability, *, allow_non_reportable: bool = False) -> None:
    """Gate a reportable-only operation.

    Callers that render a headline number must pass through here. The default is
    to REFUSE: a Phase-D figure reaching a project table is exactly the hazard
    contract §1 exists to prevent, and an opt-in flag makes the exception
    explicit and greppable rather than accidental."""
    if rep.reportable or allow_non_reportable:
        return
    raise ReportabilityError(
        f"{rep.banner} Pass allow_non_reportable=True to render it anyway "
        "(dev-iteration reference only)."
    )
