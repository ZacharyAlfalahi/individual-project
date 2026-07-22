"""
explainer.py — the LLM prose explainer (design step 19, §11).

Runs strictly AFTER the deterministic verdict, with ZERO gating power and
originating ZERO numbers. It rewrites the audit's findings as plain-language prose,
then `numeric_verifier` checks EVERY numeric token against the typed report; on any
mismatch (or a malformed response) it retries, and after `max_attempts` it FALLS
BACK to the deterministic `render_report` (which is verifier-safe by construction).
So the returned prose is always number-faithful — the LLM can only improve fluency,
never introduce a number that is not in the report.

The factual source handed to the model IS `render_report(report)` — a compact block
whose numbers are already the report's values in verifier-passing display form — so a
compliant model simply reuses them. This keeps the explainer SEPARATE from
`run_full_audit` (which stays deterministic / LLM-free); the caller attaches the
prose downstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ..schemas.audit_report import AuditReport
from .model_client import ExplainerClient
from .numeric_verifier import verify_numbers
from .renderer import render_report

_PROMPT_TEMPLATE = """\
You are writing a short, plain-language explanation of an automated bias audit of a
corporate-bond factor strategy, for a reader who is not a statistician. Rewrite the
audit findings below as a fluent 3-6 sentence summary.

STRICT RULES (a downstream verifier rejects violations):
- Use ONLY the numbers that appear in the findings below, copied EXACTLY as written.
- Do NOT compute, re-round, combine, or invent ANY number.
- Do NOT introduce any figure (a count, a percentage, a year) that is not below.
- Return ONLY a JSON object of the form: {{"explanation": "<your summary>"}}

FINDINGS:
{findings}
"""


def build_prompt(report: AuditReport) -> str:
    """The explainer prompt: the deterministic findings block + strict instruction."""
    return _PROMPT_TEMPLATE.format(findings=render_report(report))


def _extract_prose(text: str) -> str | None:
    """Parse the model's `{"explanation": "..."}` object. Tolerates surrounding
    whitespace / code fences by falling back to the first `{`..last `}` span.
    Returns None on any failure (a failed attempt → retry → fallback)."""
    for candidate in (text, _brace_span(text)):
        if candidate is None:
            continue
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            prose = obj.get("explanation")
            if isinstance(prose, str) and prose.strip():
                return prose
    return None


def _brace_span(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


@dataclass(frozen=True)
class ExplainerOutput:
    """The explainer result. `prose` is always number-faithful (the fallback
    guarantees it). `used_fallback` records whether the deterministic renderer was
    used because the model never produced verifiable prose."""

    prose: str
    model_version: str | None
    configured_model_id: str | None
    prompt_hash: str
    verified: bool
    used_fallback: bool
    n_attempts: int
    unverified: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "prose": self.prose,
            "model_version": self.model_version,
            "configured_model_id": self.configured_model_id,
            "prompt_hash": self.prompt_hash,
            "verified": self.verified,
            "used_fallback": self.used_fallback,
            "n_attempts": self.n_attempts,
            "unverified": list(self.unverified),
        }


def explain(
    report: AuditReport,
    client: ExplainerClient,
    *,
    configured_model_id: str | None = None,
    max_attempts: int = 3,
) -> ExplainerOutput:
    """Render `report` as verified prose via `client`, falling back to the
    deterministic renderer if the model never produces number-faithful prose."""
    report_dict = report.to_dict()
    prompt = build_prompt(report)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    last_version: str | None = None
    for attempt in range(1, max_attempts + 1):
        resp = client.generate(prompt)
        last_version = resp.model_version
        prose = _extract_prose(resp.text)
        if prose is not None and verify_numbers(prose, report_dict).ok:
            return ExplainerOutput(
                prose=prose, model_version=resp.model_version,
                configured_model_id=configured_model_id, prompt_hash=prompt_hash,
                verified=True, used_fallback=False, n_attempts=attempt,
            )

    # Fallback: the deterministic renderer emits only report numbers (verifier-safe).
    fallback = render_report(report)
    v = verify_numbers(fallback, report_dict)
    return ExplainerOutput(
        prose=fallback, model_version=last_version,
        configured_model_id=configured_model_id, prompt_hash=prompt_hash,
        verified=v.ok, used_fallback=True, n_attempts=max_attempts,
        unverified=v.unverified,
    )
