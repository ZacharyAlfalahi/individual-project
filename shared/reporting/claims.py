"""claims.py — the claim-ledger and evidence types (docs/reporter/reporter_spec_v0.2.md §8.1, §8.2, §8.4).

Every quantitative statement the Reporter emits is declared as a `ClaimSpec` and recorded
as a `ClaimRecord` binding the displayed number to its typed source (artefact, locator,
formatter, unit, conditioning). This is what stops a real number being attached to the
wrong label (INV-1, INV-10). Upstream-authored prose (paper quotes, mechanism claims, the
Auditor fragment, refusal details) is carried verbatim as an `EvidenceBlock` (INV-14).

These are typed carriers only — stdlib frozen dataclasses with `__post_init__` validation,
NO Pydantic (§1.4). Serialisation of a ledger routes through
`shared/reporting/canonical.py`, which is the single place NaN / -0.0 / NFC are handled;
`to_dict` here therefore returns near-plain structures and may embed a `NanValue`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from .canonical import NanValue

# The two locator kinds. `json_pointer` addresses the artefact's own `to_dict()` output;
# `projection_pointer` addresses a Reporter-owned canonical projection of a type that has
# no usable serialisation (QuantConfig, StrategyResult) — see §8.4.
LocatorKind = Literal["json_pointer", "projection_pointer"]
_LOCATOR_KINDS: tuple[LocatorKind, ...] = ("json_pointer", "projection_pointer")


class ArtefactType(str, Enum):
    """The typed upstream artefacts a claim or evidence block may be sourced from. The
    value is the on-disk / registry label."""

    STRATEGY_SPEC = "strategy_spec"
    LIBRARIAN_TRACE = "librarian_trace"
    ADAPT_RESULT = "adapt_result"
    QUANT_CONFIG = "quant_config"
    CONFIG_REFUSAL = "config_refusal"
    STRATEGY_RESULT = "strategy_result"
    QUANT_COVERAGE = "quant_coverage"
    QUANT_RUN_LOG = "quant_run_log"
    AUDIT_REPORT = "audit_report"
    AUDIT_RUN_LOG = "audit_run_log"
    EVALUATION_RECORD = "evaluation_record"
    GATE_OUTCOME = "gate_outcome"
    EXTENSION_PROPOSAL = "extension_proposal"
    CROWDING_DIAGNOSTIC = "crowding_diagnostic"
    HOLDOUT_VIEW = "holdout_view"
    MECHANISM_LIBRARY = "mechanism_library"
    THRESHOLDS = "thresholds"


class Unit(str, Enum):
    """Display units. The unit rides on the `ClaimSpec` so a percent/decimal mismatch is a
    typed error, not a silent one (INV-1)."""

    DECIMAL = "decimal"
    PERCENT = "percent"
    BPS = "bps"
    COUNT = "count"
    MONTHS = "months"
    T_STAT = "t_stat"
    P_VALUE = "p_value"
    SHARPE = "sharpe"
    DIMENSIONLESS = "dimensionless"


@dataclass(frozen=True)
class SourceLocator:
    """Addresses a value inside a serialised artefact (`json_pointer`) or a Reporter
    projection (`projection_pointer`). `pointer` is an RFC-6901 JSON Pointer."""

    kind: LocatorKind
    pointer: str
    projection_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _LOCATOR_KINDS:
            raise ValueError(f"SourceLocator.kind must be one of {_LOCATOR_KINDS}")
        if not isinstance(self.pointer, str):
            raise TypeError("SourceLocator.pointer must be a str")
        if self.pointer != "" and not self.pointer.startswith("/"):
            raise ValueError(
                f"pointer must be '' or start with '/' (RFC-6901); got {self.pointer!r}"
            )
        if self.kind == "projection_pointer":
            if not self.projection_id:
                raise ValueError(
                    "projection_pointer requires a non-empty projection_id"
                )
        elif self.projection_id is not None:
            raise ValueError("json_pointer must not carry a projection_id")

    def to_dict(self) -> dict:
        out: dict = {"kind": self.kind, "pointer": self.pointer}
        if self.projection_id is not None:
            out["projection_id"] = self.projection_id
        return out


@dataclass(frozen=True)
class EvidenceVerification:
    """A machine verification attached to an `EvidenceBlock` whose numeric tokens are
    licensed by a verifier run (e.g. the Auditor fragment via `verify_numbers`)."""

    verifier: str
    ok: bool
    n_checked: int
    unverified: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("EvidenceVerification.ok must be bool")
        if not isinstance(self.n_checked, int) or isinstance(self.n_checked, bool):
            raise TypeError("EvidenceVerification.n_checked must be int")

    def to_dict(self) -> dict:
        return {
            "verifier": self.verifier,
            "ok": self.ok,
            "n_checked": self.n_checked,
            "unverified": list(self.unverified),
        }


@dataclass(frozen=True)
class EvidenceBlock:
    """Verbatim upstream text with its source hash and locator (INV-14). Numeric tokens
    inside `verbatim_text` are licensed by `verification`, not by a generic exemption."""

    evidence_id: str
    source_artifact: ArtefactType
    source_artifact_sha256: str
    source_locator: SourceLocator
    verbatim_text: str
    verification: EvidenceVerification | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_artifact, ArtefactType):
            raise TypeError("EvidenceBlock.source_artifact must be an ArtefactType")
        if not isinstance(self.source_locator, SourceLocator):
            raise TypeError("EvidenceBlock.source_locator must be a SourceLocator")
        if not isinstance(self.verbatim_text, str):
            raise TypeError("EvidenceBlock.verbatim_text must be a str")

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "source_artifact": self.source_artifact.value,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_locator": self.source_locator.to_dict(),
            "verbatim_text": self.verbatim_text,
            "verification": (
                self.verification.to_dict() if self.verification is not None else None
            ),
        }


@dataclass(frozen=True)
class ClaimSpec:
    """A declared quantitative slot: where the number comes from, how it is formatted, and
    the conditioning it must carry. The renderer resolves and emits it — it exposes no API
    to interpolate a raw number directly (INV-1)."""

    claim_id: str
    slot_id: str
    source_artifact: ArtefactType
    source_locator: SourceLocator
    formatter_id: str
    unit: Unit
    conditioning_pointer: SourceLocator | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_artifact, ArtefactType):
            raise TypeError("ClaimSpec.source_artifact must be an ArtefactType")
        if not isinstance(self.source_locator, SourceLocator):
            raise TypeError("ClaimSpec.source_locator must be a SourceLocator")
        if not isinstance(self.unit, Unit):
            raise TypeError("ClaimSpec.unit must be a Unit")
        if self.conditioning_pointer is not None and not isinstance(
            self.conditioning_pointer, SourceLocator
        ):
            raise TypeError(
                "ClaimSpec.conditioning_pointer must be a SourceLocator or None"
            )

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "slot_id": self.slot_id,
            "source_artifact": self.source_artifact.value,
            "source_locator": self.source_locator.to_dict(),
            "formatter_id": self.formatter_id,
            "unit": self.unit.value,
            "conditioning_pointer": (
                self.conditioning_pointer.to_dict()
                if self.conditioning_pointer is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ClaimRecord:
    """The emitted ledger record for one claim: the resolved raw value, the displayed
    string, and the full provenance needed to re-verify it (INV-1, INV-10). `raw_value`
    may be an int, float, str or `NanValue`; canonicalisation handles the encoding."""

    claim_id: str
    slot_id: str
    source_artifact: ArtefactType
    source_artifact_sha256: str
    source_schema_version: str
    source_locator: SourceLocator
    raw_value: object
    displayed_value: str
    unit: str
    precision: int | None
    conditioning_text: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_artifact, ArtefactType):
            raise TypeError("ClaimRecord.source_artifact must be an ArtefactType")
        if not isinstance(self.source_locator, SourceLocator):
            raise TypeError("ClaimRecord.source_locator must be a SourceLocator")
        if not isinstance(self.displayed_value, str):
            raise TypeError("ClaimRecord.displayed_value must be a str")
        if not isinstance(
            self.raw_value, (int, float, str, NanValue)
        ) or isinstance(self.raw_value, bool):
            raise TypeError(
                "ClaimRecord.raw_value must be int|float|str|NanValue "
                f"(not bool); got {self.raw_value!r}"
            )
        if self.precision is not None and (
            not isinstance(self.precision, int) or isinstance(self.precision, bool)
        ):
            raise TypeError("ClaimRecord.precision must be int or None")

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "slot_id": self.slot_id,
            "source_artifact": self.source_artifact.value,
            "source_artifact_sha256": self.source_artifact_sha256,
            "source_schema_version": self.source_schema_version,
            "source_locator": self.source_locator.to_dict(),
            "raw_value": self.raw_value,
            "displayed_value": self.displayed_value,
            "unit": self.unit,
            "precision": self.precision,
            "conditioning_text": self.conditioning_text,
        }
