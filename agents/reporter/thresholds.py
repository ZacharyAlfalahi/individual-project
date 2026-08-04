"""thresholds.py — fail-loud loader for the Reporter's `reporter:` block (docs/reporter/reporter_spec_v0.2.md §7).

Mirrors `shared/handoff/scientist_case.py` (and, through it, `agents/auditor/thresholds.py`):
a `KeyError` subclass raised rather than defaulted, and `_resolve_dotted` for cross-block
references. The Reporter states no numerical constant of its own that another agent already
owns; instead the `reporter:` block REFERENCES the existing constants (`auditor.fdr.q`,
`auditor.bootstrap.n_replicates`, `auditor.practical_significance.vartheta`,
`mom6.formation_months`), which the verifier resolves as structural constants (§9). The one
genuinely Reporter-owned literal is `reporter.verify.rel_tol`, passed to
`numeric_verifier.verify_numbers`.

Adding the `reporter:` block is additive and safe: no loader enumerates the allowed top-level
keys, and the frozen auditor / IPCA hashes are computed over their own sub-blocks, so a new
sibling key is invisible to them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"

# The structural-constant references the Reporter's verifier resolves against sibling blocks.
_STRUCTURAL_REFS: dict[str, str] = {
    "momentum_horizon": "momentum_horizon_ref",
    "fdr_q": "fdr_q_ref",
    "bootstrap_replicates": "bootstrap_replicates_ref",
    "materiality_threshold": "materiality_threshold_ref",
}


class ReporterThresholdError(KeyError):
    """A required Reporter constant is absent from (or malformed in) the `reporter:` block of
    `docs/thresholds.yaml`. Raised, NEVER defaulted — a silent default would launder a
    post-hoc constant (mirrors `AuditorThresholdError` / `ScientistThresholdError`, §13.2)."""

    def __init__(self, dotted_key: str) -> None:
        self._dotted_key = dotted_key
        super().__init__(
            f"reporter constant '{dotted_key}' is missing or malformed in "
            f"{THRESHOLDS_FILE.name}; the Reporter never defaults a threshold."
        )


@dataclass(frozen=True)
class ReporterParams:
    """The resolved Reporter configuration.

      schema_version        : the `reporter:` block schema version.
      rel_tol               : relative tolerance passed to `verify_numbers` (§9).
      structural_constants  : {name: value} resolved from the referenced sibling blocks,
                              e.g. {"fdr_q": 0.10, "momentum_horizon": 6.0, ...}.
    """

    schema_version: int
    rel_tol: float
    structural_constants: dict[str, float]


def _load_yaml(path: str | Path | None) -> dict:
    p = Path(path) if path is not None else THRESHOLDS_FILE
    try:
        with open(p) as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        raise ReporterThresholdError(f"reporter (thresholds file not found: {p})") from None
    if not isinstance(data, dict):
        raise ReporterThresholdError("reporter (thresholds.yaml empty/malformed)")
    return data


def _resolve_dotted(data: dict, dotted: str) -> object:
    """Walk a dotted key path into `data`, raising at the first missing level so a dangling
    reference fails loud rather than resolving to None."""
    node: object = data
    walked: list[str] = []
    for key in dotted.split("."):
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise ReporterThresholdError(".".join(walked) + " (referenced target missing)")
        node = node[key]
    return node


def _require_number(value: object, dotted_key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReporterThresholdError(f"{dotted_key} (present but not a number: {value!r})")
    return float(value)


def _require_int(value: object, dotted_key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReporterThresholdError(f"{dotted_key} (present but not an int: {value!r})")
    return value


def load_reporter_params(path: str | Path | None = None) -> ReporterParams:
    """Fail-loud loader for the `reporter:` block. Resolves every structural-constant
    reference against the same file (R3-style), so if any referenced sibling constant is
    removed the Reporter fails loudly rather than reporting a stale number."""
    data = _load_yaml(path)
    block = data.get("reporter")
    if not isinstance(block, dict):
        raise ReporterThresholdError("reporter")

    schema_version = _require_int(
        block.get("schema_version"), "reporter.schema_version"
    )

    verify = block.get("verify")
    if not isinstance(verify, dict) or "rel_tol" not in verify:
        raise ReporterThresholdError("reporter.verify.rel_tol")
    rel_tol = _require_number(verify["rel_tol"], "reporter.verify.rel_tol")
    if rel_tol < 0:
        raise ReporterThresholdError(f"reporter.verify.rel_tol (negative: {rel_tol!r})")

    consts_block = block.get("structural_constants")
    if not isinstance(consts_block, dict):
        raise ReporterThresholdError("reporter.structural_constants")
    structural: dict[str, float] = {}
    for name, ref_key in _STRUCTURAL_REFS.items():
        if ref_key not in consts_block:
            raise ReporterThresholdError(f"reporter.structural_constants.{ref_key}")
        target = consts_block[ref_key]
        if not isinstance(target, str) or not target.strip():
            raise ReporterThresholdError(
                f"reporter.structural_constants.{ref_key} (not a dotted path)"
            )
        structural[name] = _require_number(
            _resolve_dotted(data, target),
            f"reporter.structural_constants.{ref_key}->{target}",
        )

    return ReporterParams(
        schema_version=schema_version,
        rel_tol=rel_tol,
        structural_constants=structural,
    )
