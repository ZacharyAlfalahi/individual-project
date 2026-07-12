"""
Shared offline builders for the Librarian pipeline tests.

Everything here is synthetic + offline. The canonical-text substrate is the
Cluster-2 stub fixture (``canonical_text.stub.yaml``); no real quote fixtures are
authored (the brief forbids it). A frozen twin of the stub is built in-memory
(same pages, ``status="frozen"``) ONLY for the ``run_paper`` orchestrator tests,
which need a text that passes ``require_frozen``.
"""

from __future__ import annotations

from pathlib import Path

from agents.librarian.config.canonical_text import CanonicalText, load_canonical_text
from agents.librarian.pipeline import (
    Construction,
    RunProvenance,
    ModelAnswer,
)

# The Cluster-2 stub fixture path (status: stub).
STUB_PATH = str(
    Path(__file__).resolve().parent.parent.parent
    / "agents"
    / "librarian"
    / "fixtures"
    / "canonical_text.stub.yaml"
)

# Verbatim substrings that ARE present in the stub pages (so their locate()
# succeeds). Page 0 first, then page 1 -- used to exercise the quote gate.
Q_QUINTILES = "sorted into quintiles"          # page 0
Q_HIGHEST = "highest-signal bonds"             # page 0
Q_RANKED = "ranked each month"                 # page 0
Q_VALUE_WEIGHTED = "value-weighted at formation"  # page 1
Q_MONTHLY = "rebalanced monthly"               # page 1
Q_LONG_SHORT = "long-short factor return series"  # page 1
# A string that is NOT in the stub (forces a quote-gate failure).
Q_NOT_IN_STUB = "this exact phrase never appears in the stub text at all"


def load_stub() -> CanonicalText:
    return load_canonical_text(STUB_PATH)


def frozen_stub() -> CanonicalText:
    """A frozen twin of the stub (same pages) for orchestrator tests -- the only
    way to exercise a text that passes ``require_frozen`` without authoring a
    real fixture."""
    stub = load_stub()
    return CanonicalText(
        source_pdf=stub.source_pdf,
        source_sha256=stub.source_sha256,
        parser=dict(stub.parser),
        normalisation=dict(stub.normalisation),
        pages=stub.pages,
        status="frozen",
    )


def answer(field, raw, quote, model_id=None, quotes=()):
    return ModelAnswer(
        field=field, answered=True, raw=raw, quote=quote, quotes=tuple(quotes), model_id=model_id
    )


def silent(field, model_id=None):
    return ModelAnswer(field=field, answered=False, model_id=model_id)


def construction(name="Momentum Factor", quote=Q_RANKED, cls="strategy", grid=None):
    from agents.librarian.pipeline import GridInfo

    return Construction(name=name, quote=quote, cls=cls, grid=grid or GridInfo())


def provenance(**overrides):
    base = dict(
        paper_id="SYNTH-0001",
        registry_version="sig-v1",
        registry_hash="deadbeef",
        silence_table_version="v1.1",
        canonical_text_hash="cafef00d",
        model_a_id="fake-a",
        model_b_id="fake-b",
        prompt_template_hashes="promhash",
        run_id="run-1",
        timestamp="2026-07-09T00:00:00Z",
    )
    base.update(overrides)
    return RunProvenance(**base)
