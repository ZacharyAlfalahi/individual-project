"""
Librarian pipeline -- the extraction machinery (build brief §5.2-5.6).

Turns a frozen ``CanonicalText`` + two ``ModelClient``s into ``StrategySpec`` /
``ExtractionTrace`` pairs, plus typed failure/review events. Every component runs
on ANY canonical text (stub or frozen) via ``.locate()``; only the top-level
``run_paper`` orchestrator enforces the frozen-status gate (§5.1), so a stub can
never reach a real run.

  * ``model_client`` -- ``FieldQuery`` / ``ModelAnswer`` value objects, the
                        ``ModelClient`` Protocol, and the fully offline
                        ``FakeModelClient`` (dependency-injected as model_a /
                        model_b; no network, no SDK).
  * ``prompts``      -- the by-field-type prompt/schema/decoding manifest loader
                        (hashed, freeze-checked); ``query_for(field)``.
  * ``lister``       -- enumeration (D20): ``Construction``, ``GridInfo``,
                        ``enumerate_constructions`` (dual-model agreement;
                        disagreement -> ``EnumerationDisagreement`` review event),
                        ``load_gold_list``.
  * ``form_filler``  -- per-field dual-model extraction (D9/D11/D16): ``normalise``,
                        ``fill_field`` (-> ``FieldOutcome``), ``fill_method_summary``.
  * ``signal_filler``-- dual-model ``SignalRef`` extraction (D22): ``fill_signal_ref``
                        (-> ``SignalRefOutcome``), reusing the D9 merge for the
                        concept_id + each registry parameter.
  * ``cross_check``  -- deterministic Part 1 <-> Part 2 reconciliation (D14/D18);
                        ``ReviewFlag``; never auto-repairs.
  * ``trace``        -- the ``ExtractionTrace`` sidecar (D12): per-field records,
                        run header, ``to_dict``, ``sha256``.
  * ``emission``     -- ``emit_spec_and_trace`` (stamp + fail-closed validate) and
                        the ``run_paper`` orchestrator (frozen gate first).
  * ``failures``     -- the four typed outcome events (D31): ``UnparseablePdf`` /
                        ``PartialParse`` (-> paper_failed), ``LocatorSystematicFailure``
                        / ``EnumerationDisagreement`` (-> review).
"""

from __future__ import annotations

from .cross_check import ReviewFlag, cross_check
from .emission import (
    LibrarianEmissionError,
    PaperRunResult,
    RunProvenance,
    emit_spec_and_trace,
    run_paper,
)
from .failures import (
    FAILURE_ROUTING,
    EnumerationDisagreement,
    LocatorSystematicFailure,
    PartialParse,
    UnparseablePdf,
    route_of,
)
from .form_filler import (
    FieldOutcome,
    fill_field,
    fill_method_summary,
    normalise,
)
from .lister import (
    Construction,
    EnumerationResult,
    GridInfo,
    enumerate_constructions,
    load_gold_list,
)
from .model_client import (
    FieldQuery,
    FakeModelClient,
    ModelAnswer,
    ModelClient,
)
from .signal_filler import (
    MAX_DESCRIBED_QUOTES,
    SignalRefOutcome,
    fill_signal_ref,
)
from .prompts import PromptManifest, load_prompt_manifest
from .trace import (
    ExtractionTrace,
    FieldTraceRecord,
    ModelTrace,
    TraceRunHeader,
)

__all__ = [
    # model client
    "FieldQuery",
    "ModelAnswer",
    "ModelClient",
    "FakeModelClient",
    # prompts
    "PromptManifest",
    "load_prompt_manifest",
    # lister
    "Construction",
    "GridInfo",
    "EnumerationResult",
    "enumerate_constructions",
    "load_gold_list",
    # form filler
    "FieldOutcome",
    "fill_field",
    "fill_method_summary",
    "normalise",
    # signal-ref filler
    "SignalRefOutcome",
    "fill_signal_ref",
    "MAX_DESCRIBED_QUOTES",
    # cross check
    "ReviewFlag",
    "cross_check",
    # trace
    "ModelTrace",
    "FieldTraceRecord",
    "TraceRunHeader",
    "ExtractionTrace",
    # emission
    "RunProvenance",
    "PaperRunResult",
    "LibrarianEmissionError",
    "emit_spec_and_trace",
    "run_paper",
    # failures
    "UnparseablePdf",
    "PartialParse",
    "LocatorSystematicFailure",
    "EnumerationDisagreement",
    "FAILURE_ROUTING",
    "route_of",
]
