"""manifest.py — R1 run index (docs/reporter/reporter_spec_v0.2.md §4, Path B).

There is no unified run identity across the pipeline agents (each keys on a different id), so
a reported run is assembled from a HAND-MAINTAINED pointer file `reporter/runs/<id>.yaml` that
declares every source artefact's repo-relative path, sha256 and schema version, plus a
mandatory `join_rationale` recording that the cross-agent join is a human assertion (§4.2).

This module parses and validates that pointer file into a `RunManifest`:
  * `reporter_run_id` is sanitised (``[a-z0-9_-]+``) and is the ONLY id used in output paths.
  * every artefact path is normalised and rejected if absolute, containing ``..``, or resolving
    (through symlinks) outside the repo root — path-traversal defence.
  * duplicate artefact keys are rejected (custom no-duplicate YAML loader).
  * every declared sha256 is verified against the file bytes on load (INV-15) — the only
    defence against two runs at the same git short SHA overwriting each other.
An unknown artefact key or an absent/empty `join_rationale` is a typed error (INV-13).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from shared.reporting.claims import ArtefactType

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_RUN_ID_RE = re.compile(r"[a-z0-9_-]+")
_VALID_PHASES = ("D", "F")
_VALID_STAGE_STATUSES = frozenset(
    {"not_applicable", "unobserved", "succeeded", "refused", "failed"}
)

# key -> (ArtefactType, producing component). `spec` is the only required artefact; the rest
# are present only where that stage ran. An unknown key is rejected (fail-closed).
_KNOWN_ARTEFACTS: dict[str, tuple[ArtefactType, str]] = {
    "spec": (ArtefactType.STRATEGY_SPEC, "librarian"),
    "librarian_trace": (ArtefactType.LIBRARIAN_TRACE, "librarian"),
    "adapt_result": (ArtefactType.ADAPT_RESULT, "quant"),
    "quant_config": (ArtefactType.QUANT_CONFIG, "quant"),
    "config_refusal": (ArtefactType.CONFIG_REFUSAL, "quant"),
    "quant_result": (ArtefactType.STRATEGY_RESULT, "quant"),
    "quant_coverage": (ArtefactType.QUANT_COVERAGE, "quant"),
    "quant_run_log": (ArtefactType.QUANT_RUN_LOG, "quant"),
    "audit_report": (ArtefactType.AUDIT_REPORT, "auditor"),
    "audit_run_log": (ArtefactType.AUDIT_RUN_LOG, "auditor"),
}
_REQUIRED_ARTEFACTS = ("spec",)


class ManifestError(ValueError):
    """A pointer file is malformed, references an unknown artefact, declares an escaping or
    absolute path, or fails a hash check. Raised, never defaulted (INV-13, INV-15)."""


class _NoDuplicatesLoader(yaml.SafeLoader):
    """A SafeLoader that rejects duplicate mapping keys instead of silently keeping the last —
    so a repeated artefact key in the pointer file is a loud error (§4.2)."""


def _no_duplicate_mapping(loader: _NoDuplicatesLoader, node, deep: bool = False) -> dict:
    mapping: dict = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ManifestError(f"duplicate key {key!r} in pointer file")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_NoDuplicatesLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_mapping
)


@dataclass(frozen=True)
class ArtefactRef:
    """A validated reference to one source artefact: its pointer key, repo-relative path,
    declared sha256, schema version, resolved type and producing component."""

    key: str
    path: str
    sha256: str
    schema_version: str | None
    artefact_type: ArtefactType
    producing_component: str

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "path": self.path,
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "artefact_type": self.artefact_type.value,
            "producing_component": self.producing_component,
        }


@dataclass(frozen=True)
class DeclaredStage:
    """A stage record explicitly declared in the pointer file (e.g. scientist not_applicable,
    with the evidence that licenses it). Interpreted by the R2 loader."""

    status: str
    evidence: str


@dataclass(frozen=True)
class RunManifest:
    """The validated pointer file: metadata, the join assertion, the artefact table (with
    verified hashes), and any explicitly declared stage records."""

    reporter_run_id: str
    schema_version: int
    paper_id: str
    strategy_id: str
    phase: str
    join_rationale: str
    artefacts: dict[str, ArtefactRef]
    declared_stages: dict[str, DeclaredStage]


def _require_str(block: dict, key: str) -> str:
    value = block.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"pointer field {key!r} must be a non-empty string")
    return value


def _validate_repo_relative_path(path_str: object, repo_root: Path) -> Path:
    if not isinstance(path_str, str) or not path_str:
        raise ManifestError("artefact path must be a non-empty string")
    pure = PurePosixPath(path_str)
    if pure.is_absolute():
        raise ManifestError(f"artefact path must be repo-relative: {path_str!r}")
    if ".." in pure.parts:
        raise ManifestError(f"artefact path must not contain '..': {path_str!r}")
    root_resolved = repo_root.resolve()
    abs_path = (repo_root / path_str).resolve()
    try:
        abs_path.relative_to(root_resolved)
    except ValueError:
        raise ManifestError(
            f"artefact path escapes the repo root (symlink?): {path_str!r}"
        ) from None
    return abs_path


def compute_file_sha256(abs_path: Path) -> str:
    """sha256 hex digest over the raw file bytes (the identity the scaffold helper stamps)."""
    return hashlib.sha256(abs_path.read_bytes()).hexdigest()


def _build_artefact_ref(
    key: str, entry: object, repo_root: Path, *, verify_hashes: bool
) -> ArtefactRef:
    if key not in _KNOWN_ARTEFACTS:
        raise ManifestError(
            f"unknown artefact key {key!r} (known: {sorted(_KNOWN_ARTEFACTS)})"
        )
    if not isinstance(entry, dict):
        raise ManifestError(f"artefact {key!r} must be a mapping")
    path_str = _require_str(entry, "path")
    sha_declared = _require_str(entry, "sha256")
    schema_version = entry.get("schema_version")
    if schema_version is not None and not isinstance(schema_version, str):
        raise ManifestError(f"artefact {key!r} schema_version must be a string if present")

    abs_path = _validate_repo_relative_path(path_str, repo_root)
    if verify_hashes:
        if not abs_path.exists():
            raise ManifestError(f"artefact {key!r} not found at {path_str!r}")
        actual = compute_file_sha256(abs_path)
        if actual != sha_declared:
            raise ManifestError(
                f"artefact {key!r} hash mismatch (INV-15): declared {sha_declared!r}, "
                f"actual {actual!r} — the artefact changed since the pointer was written"
            )
    artefact_type, component = _KNOWN_ARTEFACTS[key]
    return ArtefactRef(
        key=key,
        path=path_str,
        sha256=sha_declared,
        schema_version=schema_version,
        artefact_type=artefact_type,
        producing_component=component,
    )


def _parse_stages(raw: object) -> dict[str, DeclaredStage]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ManifestError("pointer 'stages' must be a mapping if present")
    out: dict[str, DeclaredStage] = {}
    for stage, entry in raw.items():
        if not isinstance(entry, dict):
            raise ManifestError(f"stage {stage!r} must be a mapping")
        status = _require_str(entry, "status")
        if status not in _VALID_STAGE_STATUSES:
            raise ManifestError(
                f"stage {stage!r} status {status!r} not in {sorted(_VALID_STAGE_STATUSES)}"
            )
        evidence = _require_str(entry, "evidence")
        out[str(stage)] = DeclaredStage(status=status, evidence=evidence)
    return out


def parse_manifest_text(text: str, *, repo_root: Path, verify_hashes: bool) -> RunManifest:
    """Parse and validate pointer-file YAML text into a `RunManifest`."""
    data = yaml.load(text, Loader=_NoDuplicatesLoader)
    if not isinstance(data, dict):
        raise ManifestError("pointer file must be a mapping")

    run_id = _require_str(data, "reporter_run_id")
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ManifestError(
            f"reporter_run_id {run_id!r} must match [a-z0-9_-]+ (path-safe)"
        )
    schema_version = data.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ManifestError("pointer 'schema_version' must be an int")

    phase = _require_str(data, "phase")
    if phase not in _VALID_PHASES:
        raise ManifestError(f"phase {phase!r} must be one of {_VALID_PHASES}")

    join_rationale = _require_str(data, "join_rationale")

    artefacts_raw = data.get("artefacts")
    if not isinstance(artefacts_raw, dict) or not artefacts_raw:
        raise ManifestError("pointer 'artefacts' must be a non-empty mapping")
    artefacts: dict[str, ArtefactRef] = {}
    for key, entry in artefacts_raw.items():
        artefacts[str(key)] = _build_artefact_ref(
            str(key), entry, repo_root, verify_hashes=verify_hashes
        )
    for required in _REQUIRED_ARTEFACTS:
        if required not in artefacts:
            raise ManifestError(f"pointer file must declare the {required!r} artefact")

    return RunManifest(
        reporter_run_id=run_id,
        schema_version=schema_version,
        paper_id=_require_str(data, "paper_id"),
        strategy_id=_require_str(data, "strategy_id"),
        phase=phase,
        join_rationale=join_rationale,
        artefacts=artefacts,
        declared_stages=_parse_stages(data.get("stages")),
    )


def load_manifest(
    pointer_path: str | Path, *, repo_root: Path = REPO_ROOT, verify_hashes: bool = True
) -> RunManifest:
    """Load, parse and validate a `reporter/runs/<id>.yaml` pointer file."""
    text = Path(pointer_path).read_text()
    return parse_manifest_text(text, repo_root=repo_root, verify_hashes=verify_hashes)


def scaffold_pointer(
    *,
    reporter_run_id: str,
    paper_id: str,
    strategy_id: str,
    phase: str,
    artefacts: dict[str, str],
    repo_root: Path = REPO_ROOT,
    schema_version: int = 1,
) -> str:
    """Return a filled pointer-file YAML string with sha256 hashes computed from the given
    repo-relative artefact paths. Writes NOTHING (the caller prints it). `join_rationale` is
    emitted as a TODO placeholder to be filled in manually — the join is a human assertion."""
    if not _RUN_ID_RE.fullmatch(reporter_run_id):
        raise ManifestError(f"reporter_run_id {reporter_run_id!r} must match [a-z0-9_-]+")
    lines: list[str] = [
        f"reporter_run_id: {reporter_run_id}",
        f"schema_version: {schema_version}",
        f"paper_id: {paper_id}",
        f"strategy_id: {strategy_id}",
        f"phase: {phase}",
        "join_rationale: >",
        "  TODO: state why these runs describe the same strategy execution, and who "
        "asserts it.",
        "  No mechanical cross-agent join exists (Path B).",
        "artefacts:",
    ]
    for key, path_str in artefacts.items():
        if key not in _KNOWN_ARTEFACTS:
            raise ManifestError(f"unknown artefact key {key!r}")
        abs_path = _validate_repo_relative_path(path_str, repo_root)
        if not abs_path.exists():
            raise ManifestError(f"artefact {key!r} not found at {path_str!r}")
        digest = compute_file_sha256(abs_path)
        lines.append(f"  {key}:")
        lines.append(f"    path: {path_str}")
        lines.append(f'    sha256: "{digest}"')
    return "\n".join(lines) + "\n"
