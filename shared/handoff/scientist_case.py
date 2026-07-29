"""
scientist_case.py — the Auditor->Scientist SEAM (R2). The ONE place an audit magnitude
becomes a verdict.

Every module under `agents/scientist/` is import-isolated from the magnitude-bearing Auditor
schemas (enforced statically by `tests/unit/test_scientist_wall_import.py`). This seam lives in
`shared/handoff/` precisely so it MAY import them: it reads the `AuditReport`, applies the
pre-registered entry rule (D14 / R3), and emits a magnitude-free `ScientistCase`. Nothing
downstream of here ever sees a Sharpe gap, effect size or p-value again — that is the wall.

The entry-rule numbers (theta, q) are INJECTED, never hardcoded (prohibition 1). They belong
to the `scientist:` block of `docs/thresholds.yaml`, which references `auditor.theta` /
`auditor.fdr.q` (R3, stated once in the auditor block, never restated). That block does not
exist yet, so `load_entry_rule_params` is FAIL-LOUD (mirrors `agents/auditor/thresholds.py`):
it raises rather than defaulting a post-hoc constant. `apply_entry_rule` itself takes theta/q
as plain parameters and knows nothing about where they came from — it is a pure function, fully
unit-testable with synthetic stats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

# --- SEAM PRIVILEGE (R2) -----------------------------------------------------------------
# This module — and ONLY this module — may import the magnitude-bearing Auditor schemas.
# `agents/scientist/` may not (test_wall_import_invariant). Toggle identities are shared.
from agents.auditor.schemas.audit_report import AuditReport
from agents.auditor.schemas.toggle import TOGGLE_IDS, ToggleId

from agents.scientist.schemas.case import (
    DevelopmentWindow,
    HoldoutStatus,
    ScientistCase,
)
from agents.scientist.schemas.outcomes import Verdict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


# =========================================================================================
# ToggleStat — the seam's magnitude-bearing input (deliberately NOT under agents/scientist/)
# =========================================================================================

@dataclass(frozen=True)
class ToggleStat:
    """One toggle's first-order audit statistics, as read from the `AuditReport`. It carries
    MAGNITUDES — a signed effect and a BH-adjusted p — which is exactly why it lives in the
    seam and never crosses the wall into `agents/scientist/`.

      bh_adjusted_p : within-strategy BH-adjusted p for the first-order DOE effect
                      (`FdrReport.decisions[frozenset({toggle})].adjusted_p`).
      effect_signed : the signed first-order DOE effect E_i = 2*gamma_i = the average
                      ON(corrected) - OFF(as-published) difference on the primary metric
                      (`SaturatedBasis.doe[frozenset({toggle})]`). Negative => the correction
                      LOWERS the metric (performance-reducing).
    """

    bh_adjusted_p: float
    effect_signed: float

    def __post_init__(self) -> None:
        for name in ("bh_adjusted_p", "effect_signed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"ToggleStat.{name} must be a real number; got {value!r}")

    @property
    def effect_magnitude(self) -> float:
        """|E_i| — the materiality magnitude compared against theta."""
        return abs(self.effect_signed)


# =========================================================================================
# The entry rule (D14 / R3) — a PURE function of injected (theta, q)
# =========================================================================================

def apply_entry_rule(
    per_toggle_stats: Mapping[ToggleId, ToggleStat],
    *,
    theta: float,
    q: float,
    refused_toggles: frozenset[ToggleId] = frozenset(),
) -> dict[ToggleId, Verdict]:
    """The pre-registered entry rule (D14 / R3). A toggle is `FAIL` (correction-sensitive) iff
    its first-order effect

      (i)   clears the within-strategy BH-adjusted-p tier:  ``bh_adjusted_p <= q``,  AND
      (ii)  exceeds the materiality threshold:               ``|effect| > theta``,   AND
      (iii) is performance-reducing (the correction lowers the metric): ``effect_signed < 0``.

    Otherwise `PASS`. A toggle in `refused_toggles` (not runnable / provenance not STATED) is
    `REFUSED` regardless of any statistic — it has no defensible endpoint to difference, so no
    magnitude is trusted for it; refusal DOMINATES.

    theta and q are INJECTED here (never hardcoded — prohibition 1); see
    `load_entry_rule_params` for the fail-loud loader that supplies them from the pre-registered
    `scientist:` block. This function is pure and side-effect-free."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"q must be in (0,1); got {q!r}")
    if isinstance(theta, bool) or not isinstance(theta, (int, float)) or theta < 0:
        raise ValueError(f"theta must be a non-negative number; got {theta!r}")

    verdicts: dict[ToggleId, Verdict] = {}
    for tid in refused_toggles:
        verdicts[tid] = "REFUSED"
    for tid, stat in per_toggle_stats.items():
        if tid in refused_toggles:
            continue  # refusal dominates — no magnitude is trusted for a refused toggle
        clears_tier = stat.bh_adjusted_p <= q
        material = stat.effect_magnitude > theta
        performance_reducing = stat.effect_signed < 0.0
        verdicts[tid] = "FAIL" if (clears_tier and material and performance_reducing) else "PASS"
    return verdicts


# =========================================================================================
# build_scientist_case — read AuditReport, apply the rule, emit a magnitude-free case
# =========================================================================================

def _first_order_key(tid: ToggleId) -> frozenset:
    """The saturated-basis / FDR coordinate for a toggle's first-order effect."""
    return frozenset({tid})


def build_scientist_case(
    audit_report: AuditReport,
    *,
    strategy_id: str,
    case_id: str,
    corrected_quant_config_ref: str,
    corrected_run_ref: str,
    audit_report_ref: str,
    development_window: DevelopmentWindow,
    holdout_status: HoldoutStatus,
    theta: float,
    q: float,
    refused_toggles: frozenset[ToggleId] = frozenset(),
) -> ScientistCase:
    """Read the `AuditReport`, apply the entry rule, and emit a magnitude-free `ScientistCase`.

    INTEGRATION POINT — the ONLY Auditor-field extraction in the Scientist stack. Verified
    against `agents/auditor/confirmatory.py` and `agents/auditor/checks/report.py`:

      * signed first-order effect   <- ``audit_report.core.saturated.doe[frozenset({toggle})]``
            E_i = 2*gamma_i = average ON(corrected) - OFF(as-published) on the primary metric.
      * within-strategy BH-adj. p   <- ``audit_report.fdr.decisions[frozenset({toggle})].adjusted_p``
            The within-strategy diagnostic `FdrReport` (report.py builds it via `run_fdr` over
            `confirmatory_coordinates`); first-order coordinates ARE `frozenset({toggle})`
            (confirmatory.py:37).

    Both maps are keyed by frozenset coordinates. A runnable toggle missing EITHER coordinate is
    a fail-loud `KeyError` — we never silently PASS a toggle we could not evaluate (that would
    hide a bias). Refused toggles are excluded from extraction and mapped to REFUSED by the rule.
    """
    core = audit_report.core
    doe = core.saturated.doe            # SaturatedBasis.doe: {frozenset: float}
    decisions = audit_report.fdr.decisions  # FdrReport.decisions: {frozenset: FdrDecision}
    runnable_set = set(core.runnable_toggles)

    per_toggle: dict[ToggleId, ToggleStat] = {}
    for tid in TOGGLE_IDS:  # canonical order
        if tid not in runnable_set or tid in refused_toggles:
            continue
        key = _first_order_key(tid)
        if key not in doe:
            raise KeyError(
                f"build_scientist_case: no first-order DOE effect for {tid!r} "
                f"(saturated.doe key {set(key)!r} absent) — cannot evaluate the entry rule"
            )
        if key not in decisions:
            raise KeyError(
                f"build_scientist_case: no within-strategy BH-adjusted p for {tid!r} "
                f"(fdr.decisions key {set(key)!r} absent) — cannot evaluate the entry rule"
            )
        per_toggle[tid] = ToggleStat(
            bh_adjusted_p=float(decisions[key].adjusted_p),
            effect_signed=float(doe[key]),
        )

    verdicts = apply_entry_rule(per_toggle, theta=theta, q=q, refused_toggles=refused_toggles)

    # applicable_toggles = runnable ∪ refused, canonical order (§5.1 lists all applicable).
    applicable = tuple(
        t for t in TOGGLE_IDS if t in (runnable_set | set(refused_toggles))
    )
    # failed_check_ids = the FAIL (correction-sensitive) toggles — the entry trigger (§5.1).
    failed_ids = tuple(t for t in TOGGLE_IDS if verdicts.get(t) == "FAIL")
    failed_verdicts: dict[ToggleId, Verdict] = {t: "FAIL" for t in failed_ids}

    return ScientistCase(
        case_id=case_id,
        strategy_id=strategy_id,
        corrected_quant_config_ref=corrected_quant_config_ref,
        corrected_run_ref=corrected_run_ref,
        audit_report_ref=audit_report_ref,
        failed_check_ids=failed_ids,
        failed_check_verdicts=failed_verdicts,
        applicable_toggles=applicable,
        development_window=development_window,
        holdout_status=holdout_status,
    )


# =========================================================================================
# Fail-loud loader for the injected entry-rule params (mirrors auditor/thresholds.py)
# =========================================================================================

class ScientistThresholdError(KeyError):
    """A required Scientist entry-rule constant is absent from the `scientist:` block of
    `docs/thresholds.yaml`. Raised, NEVER defaulted — a silent default would launder a post-hoc
    constant, the exact sin the instrument exists to expose (mirrors
    `agents/auditor/thresholds.py::AuditorThresholdError`, §13.2)."""

    def __init__(self, dotted_key: str) -> None:
        self._dotted_key = dotted_key
        super().__init__(
            f"scientist entry-rule constant '{dotted_key}' is missing from "
            f"{THRESHOLDS_FILE.name}. The `scientist:` block references auditor.theta / "
            f"auditor.fdr.q (R3) and must be pre-registered and git-tagged before the first "
            f"generation run — the Scientist never defaults it. (Until the block is committed, "
            f"this loader raises by design; inject theta/q explicitly in tests.)"
        )


@dataclass(frozen=True)
class EntryRuleParams:
    """The two injected entry-rule constants (theta, q), resolved from the `scientist:` block."""

    theta: float
    q: float


def _resolve_dotted(data: dict, dotted: str) -> object:
    """Walk a dotted key path (e.g. 'auditor.practical_significance.vartheta') into `data`.
    Raises `ScientistThresholdError` at the first missing level — so a dangling reference in the
    `scientist:` block fails loud rather than resolving to None."""
    node: object = data
    walked: list[str] = []
    for key in dotted.split("."):
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise ScientistThresholdError(".".join(walked) + " (referenced target missing)")
        node = node[key]
    return node


def _require_number(value: object, dotted_key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScientistThresholdError(f"{dotted_key} (present but not a number: {value!r})")
    return float(value)


def load_entry_rule_params(path: str | Path | None = None) -> EntryRuleParams:
    """Fail-loud loader for the injected (theta, q). Reads the `scientist:` block of
    `thresholds.yaml` and raises `ScientistThresholdError` if the block or either constant is
    absent.

    R3 says the number is stated ONCE in the auditor block and never restated, so the block is
    expected to carry *references* — `scientist.entry_rule.materiality_threshold_ref` and
    `.fdr_q_ref` (dotted paths, e.g. 'auditor.practical_significance.vartheta',
    'auditor.fdr.q'), which this loader resolves against the same file. If a literal `theta`/`q`
    is provided instead of a ref, it is accepted as a fallback. Until the `scientist:` block is
    committed this ALWAYS raises — by design. NOT the runtime path (tests inject theta/q into
    `apply_entry_rule` directly); this exists so a caller that asks for the numbers gets a loud
    failure, never a silent default."""
    p = Path(path) if path is not None else THRESHOLDS_FILE
    with open(p) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ScientistThresholdError("scientist (thresholds.yaml empty/malformed)")
    block = data.get("scientist")
    if not isinstance(block, dict):
        raise ScientistThresholdError("scientist")
    entry = block.get("entry_rule")
    if not isinstance(entry, dict):
        raise ScientistThresholdError("scientist.entry_rule")

    theta = _load_ref_or_literal(
        data, entry, ref_key="materiality_threshold_ref", literal_key="theta",
        dotted="scientist.entry_rule",
    )
    q = _load_ref_or_literal(
        data, entry, ref_key="fdr_q_ref", literal_key="q",
        dotted="scientist.entry_rule",
    )
    if not (0.0 < q < 1.0):
        raise ScientistThresholdError(f"scientist.entry_rule.q (out of (0,1): {q!r})")
    if theta < 0:
        raise ScientistThresholdError(f"scientist.entry_rule.theta (negative: {theta!r})")
    return EntryRuleParams(theta=theta, q=q)


def _load_ref_or_literal(
    data: dict, entry: dict, *, ref_key: str, literal_key: str, dotted: str
) -> float:
    """Resolve `<dotted>.<ref_key>` (a dotted reference) if present, else `<dotted>.<literal_key>`
    (a literal number). Raises if neither is present — the number must come from SOMEWHERE
    pre-registered."""
    if ref_key in entry:
        target = entry[ref_key]
        if not isinstance(target, str) or not target.strip():
            raise ScientistThresholdError(f"{dotted}.{ref_key} (not a non-empty dotted path)")
        return _require_number(_resolve_dotted(data, target), f"{dotted}.{ref_key}->{target}")
    if literal_key in entry:
        return _require_number(entry[literal_key], f"{dotted}.{literal_key}")
    raise ScientistThresholdError(f"{dotted}.{ref_key} (or .{literal_key})")
