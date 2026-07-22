"""
The Auditor — RQ3 bias-attribution instrument.

A deterministic experimental framework (design: docs/auditor/auditor_design.md,
v1.7): it runs each strategy through a 2^k lattice of bias-toggle combinations,
producing 2^k return series that differ ONLY in toggle settings, then decomposes
the endpoint gap three ways (corner marginals, DOE effects, Shapley shares). Zero
language model appears anywhere in the analytical path (§11); an LLM appears only
in a downstream explainer, and a numeric verifier guarantees no number originates
there.

This package implements the instrument
core (preflight -> lattice -> common-support metrics -> algebra -> Shapley ->
invariance) plus the validation gates (Layer A algebraic, Layer B injection) and
the fixed-block bootstrap. Inference, Bayesian and hierarchical layers, the full
AuditReport, the LLM explainer prose, and the recovery sweep are also included.

The public entry point is `run_audit` (see checks.orchestrator).
"""

from __future__ import annotations

from .checks.orchestrator import AuditRefused, run_audit
from .checks.report import AuditorConfig, run_full_audit
from .explainer.explainer import ExplainerOutput, explain
from .explainer.model_client import (
    ExplainerResponse,
    FakeExplainerClient,
    LiveExplainerClient,
    build_live_explainer_client,
)
from .explainer.numeric_verifier import VerificationResult, verify_numbers
from .explainer.renderer import render_report
from .schemas.audit_core import AuditCore
from .schemas.audit_report import AuditReport
from .schemas.toggle import TOGGLE_IDS, ToggleFacts

__all__ = [
    "run_audit",
    "run_full_audit",
    "AuditorConfig",
    "AuditRefused",
    "AuditCore",
    "AuditReport",
    "ToggleFacts",
    "TOGGLE_IDS",
    "verify_numbers",
    "VerificationResult",
    "render_report",
    "explain",
    "ExplainerOutput",
    "FakeExplainerClient",
    "LiveExplainerClient",
    "ExplainerResponse",
    "build_live_explainer_client",
]
