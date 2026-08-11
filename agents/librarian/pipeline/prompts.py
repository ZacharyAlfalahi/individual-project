"""
Prompt/schema/decoding manifest loader (build brief §5.3).

Loads ``data/prompts/manifest.yaml`` -- the field-type -> (prompt, schema,
decoding, sha256) bindings and the field -> field-type map -- into a typed
``PromptManifest``. Two jobs:

  * **Freeze check.** Each field-type records the sha256 of its prompt template;
    the loader recomputes the byte-hash of the on-disk template and asserts it
    equals the recorded value (same discipline as the silence-policy byte-hash).
    A silent edit to a frozen prompt is a build error.
  * **Query construction.** ``query_for(field)`` returns a ``FieldQuery`` carrying
    the field's kind + the template/schema/decoding hashes, so the trace + spec
    header can pin exactly which prompt produced each value (build brief §2).

Data-only: it never calls a model. The pipeline injects the resolved
``FieldQuery`` into the form-filler.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..errors import LibrarianSchemaError
from .model_client import FIELD_KINDS, FieldQuery

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_MANIFEST_PATH = _DATA_ROOT / "prompts" / "manifest.yaml"


@dataclass(frozen=True)
class FieldTypeTemplate:
    """One field-type's bound artefacts + the recorded (and verified) template
    hash. Hashes double as the reproducibility stamp on every trace/header."""

    field_type: str
    prompt: str
    schema: str
    decoding: str
    prompt_sha256: str
    schema_sha256: str
    decoding_sha256: str


@dataclass(frozen=True)
class PromptManifest:
    """The loaded manifest: field-type templates + the field -> field-type map."""

    version: str
    templates: dict  # {field_type: FieldTypeTemplate}
    field_types: dict  # {field: field_type}
    run_templates: dict = field(default_factory=dict)  # {name: FieldTypeTemplate}

    def kind_for(self, field: str) -> str:
        if field not in self.field_types:
            raise LibrarianSchemaError(
                f"no prompt-manifest binding for field {field!r} (scope = Part 1 + "
                "ALREADY_FINAL_PART2; S3-adjudicated fields land post-S3)"
            )
        return self.field_types[field]

    def template_for(self, field: str) -> FieldTypeTemplate:
        return self.templates[self.kind_for(field)]

    def query_for(self, field: str, k: int = 1) -> FieldQuery:
        """A ``FieldQuery`` for ``field``, stamped with its template/schema/
        decoding hashes for the trace + header."""
        tpl = self.template_for(field)
        return FieldQuery(
            field=field,
            kind=tpl.field_type,
            template_hash=tpl.prompt_sha256,
            schema_hash=tpl.schema_sha256,
            decoding_hash=tpl.decoding_sha256,
            k=k,
        )

    @property
    def combined_prompt_hash(self) -> str:
        """A single sha256 over all field-type prompt hashes (sorted) -- the one
        ``prompt_template_hashes`` string stamped into a spec header."""
        joined = ";".join(
            f"{ft}:{self.templates[ft].prompt_sha256}" for ft in sorted(self.templates)
        )
        # Guarded fold: an empty run_templates yields a byte-identical hash to the
        # pre-run-template manifest; a populated one appends the sorted run-template
        # prompt hashes so the header pins the enumeration prompt too (WS-3).
        if self.run_templates:
            joined += ";" + ";".join(
                f"{rt}:{self.run_templates[rt].prompt_sha256}" for rt in sorted(self.run_templates)
            )
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _sha256_of(path: Path) -> str:
    if not path.exists():
        raise LibrarianSchemaError(f"prompt-manifest artefact not found at {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompt_manifest(path: str | Path | None = None) -> PromptManifest:
    """Load + verify the prompt manifest. Recomputes each prompt template's
    byte-hash and asserts it equals the recorded ``sha256`` (freeze check)."""
    p = Path(path) if path is not None else _DEFAULT_MANIFEST_PATH
    if not p.exists():
        raise LibrarianSchemaError(f"prompt manifest not found at {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise LibrarianSchemaError("prompt manifest must be a mapping at top level")

    version = raw.get("version")
    if not isinstance(version, str) or version.strip() == "":
        raise LibrarianSchemaError("prompt manifest must declare a non-empty 'version'")

    raw_ft = raw.get("field_types")
    if not isinstance(raw_ft, dict) or len(raw_ft) == 0:
        raise LibrarianSchemaError("prompt manifest 'field_types' must be a non-empty mapping")

    root = p.resolve().parent.parent  # .../data
    templates: dict[str, FieldTypeTemplate] = {}
    for ft, spec in raw_ft.items():
        if ft not in FIELD_KINDS:
            raise LibrarianSchemaError(
                f"prompt manifest field-type {ft!r} not one of {sorted(FIELD_KINDS)}"
            )
        if not isinstance(spec, dict):
            raise LibrarianSchemaError(f"field-type {ft!r} binding must be a mapping")
        prompt_rel = spec.get("prompt")
        schema_rel = spec.get("schema")
        decoding_rel = spec.get("decoding")
        recorded = spec.get("sha256")
        if not all(isinstance(x, str) and x for x in (prompt_rel, schema_rel, decoding_rel)):
            raise LibrarianSchemaError(f"field-type {ft!r} must bind prompt/schema/decoding paths")
        prompt_path = root / prompt_rel
        computed = _sha256_of(prompt_path)
        if recorded is not None and computed != recorded:
            raise LibrarianSchemaError(
                f"prompt template {prompt_rel!r} sha256 mismatch: recorded {recorded!r}, "
                f"computed {computed!r} -- a frozen prompt was edited without re-stamping"
            )
        templates[ft] = FieldTypeTemplate(
            field_type=ft,
            prompt=prompt_rel,
            schema=schema_rel,
            decoding=decoding_rel,
            prompt_sha256=computed,
            schema_sha256=_sha256_of(root / schema_rel),
            decoding_sha256=_sha256_of(root / decoding_rel),
        )

    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, dict) or len(raw_fields) == 0:
        raise LibrarianSchemaError("prompt manifest 'fields' must be a non-empty mapping")
    field_types: dict[str, str] = {}
    for fld, ft in raw_fields.items():
        if ft not in templates:
            raise LibrarianSchemaError(
                f"field {fld!r} maps to unknown field-type {ft!r} (no template bound)"
            )
        field_types[str(fld)] = str(ft)

    # --- run-templates (WS-3): whole-paper structured calls (e.g. enumeration),
    # bound by NAME, not by field. The FIELD_KINDS gate deliberately does NOT apply
    # here -- a run-template is not a per-field kind. Same freeze discipline: each
    # records its prompt sha256 and the loader recomputes + asserts equality.
    run_templates: dict[str, FieldTypeTemplate] = {}
    raw_run = raw.get("run_templates")
    if raw_run is not None:
        if not isinstance(raw_run, dict):
            raise LibrarianSchemaError("prompt manifest 'run_templates' must be a mapping")
        for name, spec in raw_run.items():
            if not isinstance(spec, dict):
                raise LibrarianSchemaError(f"run-template {name!r} binding must be a mapping")
            prompt_rel = spec.get("prompt")
            schema_rel = spec.get("schema")
            decoding_rel = spec.get("decoding")
            recorded = spec.get("sha256")
            if not all(isinstance(x, str) and x for x in (prompt_rel, schema_rel, decoding_rel)):
                raise LibrarianSchemaError(
                    f"run-template {name!r} must bind prompt/schema/decoding paths"
                )
            computed = _sha256_of(root / prompt_rel)
            if recorded is not None and computed != recorded:
                raise LibrarianSchemaError(
                    f"run-template prompt {prompt_rel!r} sha256 mismatch: recorded {recorded!r}, "
                    f"computed {computed!r} -- a frozen prompt was edited without re-stamping"
                )
            run_templates[str(name)] = FieldTypeTemplate(
                field_type=str(name),
                prompt=prompt_rel,
                schema=schema_rel,
                decoding=decoding_rel,
                prompt_sha256=computed,
                schema_sha256=_sha256_of(root / schema_rel),
                decoding_sha256=_sha256_of(root / decoding_rel),
            )

    return PromptManifest(
        version=str(version),
        templates=templates,
        field_types=field_types,
        run_templates=run_templates,
    )
