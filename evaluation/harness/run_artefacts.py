"""
G3 run-artefact loader (evaluation contract §3, D12, D34/D37).

Loads one extraction run's ``spec_0.json`` + ``trace_0.json`` into a shape the
gold-calibration scorer can walk. D12 makes the ``ExtractionTrace`` "the RQ1
dataset"; this module is the reader for it.

Raw dicts, NOT typed deserialisation -- deliberate, for two reasons:

  * a ``from_dict`` would have to live in ``agents/librarian/pipeline/trace.py``,
    i.e. production code changed purely to serve evaluation; and
  * ``__post_init__`` validation would REJECT the very anomalies G3 exists to
    surface (a truncated run, an append-corrupted archive, future schema drift).
    A scorer that cannot load a broken artefact cannot report that it is broken.

The guarantee is bought more cheaply instead: ``verify_trace_integrity``
recomputes the trace's canonical-JSON sha256 and asserts it equals the spec
header's ``trace_sha256`` (the D12 desync guard). That proves dict-traversal read
exactly the bytes that were hashed -- a stronger claim than type reconstruction,
in six lines.

TRAP, load-bearing: ``FieldTraceRecord.agreement`` is NOT model agreement.
``form_filler`` sets it to ``(final tag == "STATED")``, so a field where both
models agreed but the quote gate failed carries ``agreement=False``. ``RunField``
therefore does not expose it at all; use ``models_agree``, recomputed from
``normalised_a``/``normalised_b``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.harness.reportability import Reportability, classify_phase  # noqa: E402


class ArtefactError(RuntimeError):
    """A run's artefacts could not be loaded cleanly. Surfaced loudly -- a run
    that cannot be read must never be scored as if it were empty."""


class ArtefactIntegrityError(ArtefactError):
    """The artefacts are internally inconsistent (trace hash mismatch, or a raw
    archive that does not correspond 1:1 with the trace). Distinct from a missing
    file: this is corruption, and scoring it would produce a plausible number
    from an artefact nobody can reproduce."""


# The marker `scripts/run_librarian.py` writes for a field the run did NOT ASK
# about, as opposed to one the paper was silent on. The tag-reason registry has no
# `not_extracted` row (its five UNKNOWN reasons are all claims about the paper), so
# these ride `not_stated` and are identified by this note prefix. Kept in step with
# run_librarian.NOT_EXTRACTED_NOTE_PREFIX by a tripwire test.
NOT_EXTRACTED_NOTE_PREFIX = "not extracted"


@dataclass(frozen=True)
class RunField:
    """One trace record, minus the traps.

    ``agreement`` is deliberately ABSENT (see the module docstring). ``conditions``
    is an order-free multi-label view of what happened, so that no reason-ordering
    is baked into the scorer: the D9 merge attributes a field to exactly one reason
    in a fixed branch order (quote gate before single-response), which means the
    shipped ``final_reason`` under-counts one-model-mute fields. Both views are
    carried; the metrics layer reports both.
    """

    field: str
    a_answered: bool
    b_answered: bool
    a_quote: str | None
    b_quote: str | None
    a_located: bool
    b_located: bool
    a_model_id: str | None
    b_model_id: str | None
    normalised_a: object
    normalised_b: object
    final_tag: str
    shipped_reason: str
    ship_choice: str | None
    not_extracted: bool

    @property
    def shipped(self) -> bool:
        return self.final_tag == "STATED"

    @property
    def models_agree(self) -> bool:
        """TRUE model concord: both answered and the normalised values match.

        Not read from the trace's ``agreement`` bit, which means something else."""
        return self.a_answered and self.b_answered and self.normalised_a == self.normalised_b

    @property
    def conditions(self) -> frozenset[str]:
        """Order-free conditions. Sums to more than one per field by design --
        this is an INCIDENCE view, never a partition."""
        out = set()
        if not self.a_answered and not self.b_answered:
            out.add("both_silent")
        if self.a_answered != self.b_answered:
            out.add("one_silent")
        if self.a_answered and self.b_answered:
            out.add("both_answered")
            out.add("values_agree" if self.models_agree else "values_differ")
        if (self.a_answered and not self.a_located) or (self.b_answered and not self.b_located):
            out.add("quote_gate_failed")
        if self.a_answered and self.b_answered and self.models_agree \
                and not (self.a_located and self.b_located):
            # Both models found the same answer and it was lost to the LOCATOR,
            # not to reading. The declared fourth 2x2 cell (D37 ruling 9) and the
            # population contract §3.6's ReAct-vs-k=3 decision turns on.
            out.add("agree_quote_gate_failed")
        if self.not_extracted:
            out.add("not_extracted")
        return frozenset(out)


@dataclass(frozen=True)
class RunArtefacts:
    """One scored run: its header, its fields keyed by flat trace name, its
    reportability stamp, and the raw-archive line counts (kept for the integrity
    check, not for scoring).

    ``reportability`` has NO DEFAULT: a run cannot be loaded without its phase
    being decided, so nothing downstream can render a number from an artefact
    whose reportability nobody established (contract §1)."""

    run_dir: Path
    paper_id: str
    header: dict
    fields: dict[str, RunField]
    reportability: "Reportability"
    raw_line_counts: dict[str, int] = dc_field(default_factory=dict)

    @property
    def model_ids(self) -> tuple[str, str]:
        return (self.header.get("model_a_id", ""), self.header.get("model_b_id", ""))


def canonical_trace_json(trace_dict: dict) -> str:
    """Byte-identical to ``ExtractionTrace._canonical_json`` (sorted keys, ASCII,
    tight separators). Reimplemented rather than imported so the check does not
    depend on constructing the typed object it is meant to verify."""
    return json.dumps(trace_dict, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def verify_trace_integrity(trace_dict: dict, spec_dict: dict) -> None:
    """Raise ``ArtefactIntegrityError`` unless the trace hashes to the spec
    header's ``trace_sha256`` (D12: spec and trace cannot silently desync)."""
    stamped = (spec_dict.get("header") or {}).get("trace_sha256")
    if not stamped:
        raise ArtefactIntegrityError(
            "spec header carries no trace_sha256 -- cannot verify the trace it points at"
        )
    actual = hashlib.sha256(canonical_trace_json(trace_dict).encode("utf-8")).hexdigest()
    if actual != stamped:
        raise ArtefactIntegrityError(
            f"trace hash mismatch: spec header stamps {stamped[:16]}..., trace hashes to "
            f"{actual[:16]}... -- the spec and trace do not correspond (D12)"
        )


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise ArtefactError(f"missing run artefact: {path}")
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtefactError(f"{path} is not valid JSON: {exc}") from exc


def _not_extracted_fields(spec_dict: dict) -> set[str]:
    """Flat field names the run did not ASK about, read off the spec's Evidence
    notes. The trace header carries no ``field_limit``, so the spec's note is the
    only signal distinguishing a never-asked field from a genuinely silent paper
    -- and conflating them would corrupt the §3.6 missed-evidence denominator."""
    found: set[str] = set()

    def walk(node, name=None):
        if isinstance(node, dict):
            ev = node.get("evidence")
            tag = node.get("tag")
            if tag is not None and isinstance(ev, dict) and name:
                note = ev.get("note") or ""
                if note.startswith(NOT_EXTRACTED_NOTE_PREFIX):
                    found.add(name)
            for k, v in node.items():
                walk(v, k)
        elif isinstance(node, list):
            for v in node:
                walk(v, name)

    walk(spec_dict)
    return found


def load_run(run_dir: str | Path, *, strategy_index: int = 0,
             check_raw: bool = True, model_stack: dict | None = None) -> RunArtefacts:
    """Load one strategy's spec + trace from a run directory.

    ``check_raw`` compares the per-model raw JSONL line counts against the trace
    record count. ``_archive`` appends while ``_write_outputs`` overwrites, so a
    directory written twice accumulates raw lines while its spec/trace are
    replaced -- the archive then no longer corresponds to the trace. That has
    already happened once in this project (``runs/bbw_full.``), so it is checked
    rather than assumed."""
    run_dir = Path(run_dir)
    spec = _read_json(run_dir / f"spec_{strategy_index}.json")
    trace = _read_json(run_dir / f"trace_{strategy_index}.json")
    verify_trace_integrity(trace, spec)

    records = trace.get("records") or []
    raw_counts: dict[str, int] = {}
    raw_dir = run_dir / "raw"
    if raw_dir.is_dir():
        for p in sorted(raw_dir.glob("raw_model_*.jsonl")):
            with p.open("r", encoding="utf-8") as fh:
                raw_counts[p.name] = sum(1 for _ in fh)
    if check_raw and raw_counts:
        bad = {k: v for k, v in raw_counts.items() if v != len(records)}
        if bad:
            raise ArtefactIntegrityError(
                f"{run_dir}: raw archive does not correspond to the trace "
                f"({len(records)} records vs {bad}). _archive appends while the spec is "
                "overwritten, so this directory was almost certainly written more than once; "
                "scoring it would mix calls from different runs."
            )

    skipped = _not_extracted_fields(spec)
    fields: dict[str, RunField] = {}
    for rec in records:
        name = rec["field"]
        if name in fields:
            raise ArtefactIntegrityError(
                f"{run_dir}: duplicate trace record for field {name!r} -- the flat trace "
                "namespace collided (a registry parameter sharing a schema field name), "
                "so a record would be silently overwritten"
            )
        a, b = rec["model_a"], rec["model_b"]
        fields[name] = RunField(
            field=name,
            a_answered=bool(a["answered"]), b_answered=bool(b["answered"]),
            a_quote=a.get("quote"), b_quote=b.get("quote"),
            a_located=a.get("locate_result") is not None,
            b_located=b.get("locate_result") is not None,
            a_model_id=a.get("model_id"), b_model_id=b.get("model_id"),
            normalised_a=rec.get("normalised_a"), normalised_b=rec.get("normalised_b"),
            final_tag=rec["final_tag"], shipped_reason=rec["final_reason"],
            ship_choice=rec.get("ship_choice"),
            not_extracted=name in skipped,
        )

    header = trace.get("header") or {}
    return RunArtefacts(
        run_dir=run_dir,
        paper_id=header.get("paper_id", ""),
        header=header,
        fields=fields,
        reportability=classify_phase(header, model_stack),
        raw_line_counts=raw_counts,
    )
