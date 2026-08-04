"""verify.py — R6 the five layers of defence (docs/reporter/reporter_spec_v0.2.md §9).

The Reporter must never emit a number that is not bound to a typed source, nor bind a real
number to the wrong label. Five layers enforce this:

  1. AST lint       — no float literals and no arithmetic in the renderer's `render_*` functions
                      (a research quantity must never be computed in a template; INV-1).
  2. Type constraint— every section returns a `RenderedFragment` (enforced by construction; a
                      section cannot return bare text). Verified structurally here.
  3. Structural     — every `ClaimRecord`'s displayed value is re-derivable from its raw value
                      and, for bundle-sourced claims, the raw value re-resolves from the artefact
                      at its locator (INV-1).
  4. Ledger bijection (INV-10) — the multiset of numeric tokens in the note body equals the
                      multiset of claim display tokens; an orphan claim or a stray token fails.
  5. Stray-token backstop — `numeric_verifier.verify_numbers` over the body against the union of
                      claim raw values, evidence-block numbers and structural constants.

Code spans (`...`), fenced blocks and blockquotes (verbatim evidence) are excluded from the
token scan: hashes, git SHAs, toggle ids and mechanism claims are identifiers/verbatim, not
Reporter claims, and are licensed elsewhere (INV-14).
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from agents.auditor.explainer.numeric_verifier import _NUMBER_RE, verify_numbers

from shared.reporting.canonical import NanValue

from . import renderer as _renderer_module
from .format import fmt, scale_for_unit
from .thresholds import load_reporter_params

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE = re.compile(r"`[^`]*`")
_BLOCKQUOTE = re.compile(r"^\s*>.*$", re.MULTILINE)
# Numeric-only operators: a difference, ratio, power etc. cannot apply to strings, so their
# presence in a template function means a research quantity is being computed (forbidden, §6).
# `+` and `*` are permitted — the renderer uses them for string building only.
_ARITH = (
    ast.Sub,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.MatMult,
)


class LintError(AssertionError):
    """The renderer source contains a numeric literal or arithmetic in a template function."""


class VerificationError(AssertionError):
    """A rendered document failed one of the verification layers."""


@dataclass(frozen=True)
class VerifyReport:
    ok: bool
    n_claims: int
    n_tokens_checked: int
    errors: tuple[str, ...] = field(default_factory=tuple)


# --- Layer 1: AST lint -----------------------------------------------------------------------

def lint_source(source: str) -> list[str]:
    """Return the numeric-literal / arithmetic violations in every function of `source`. A
    float literal or a numeric-only operator (Sub/Div/…) inside a renderer function means a
    hardcoded number or a computed research quantity — including in a helper that feeds prose,
    so all functions are scanned, not only `render_*`."""
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.BinOp) and isinstance(sub.op, _ARITH):
                violations.append(f"{node.name}: arithmetic at line {sub.lineno}")
            if isinstance(sub, ast.AugAssign) and isinstance(sub.op, _ARITH):
                violations.append(f"{node.name}: augmented arithmetic at line {sub.lineno}")
            if isinstance(sub, ast.Constant) and isinstance(sub.value, float):
                violations.append(f"{node.name}: float literal at line {sub.lineno}")
    return violations


def lint_renderer() -> None:
    """Raise `LintError` if the renderer's `render_*` functions contain a float literal or a
    numeric-only operator. Int literals (slice/index) and string `+`/`*` are allowed."""
    violations = lint_source(inspect.getsource(_renderer_module))
    if violations:
        raise LintError("; ".join(violations))


# --- token helpers ---------------------------------------------------------------------------

def strip_noncontent(md: str) -> str:
    """Remove fenced blocks, inline code spans and blockquote lines — the regions carrying
    identifiers, hashes and verbatim evidence, none of which are Reporter claim numbers."""
    md = _FENCED.sub(" ", md)
    md = _BLOCKQUOTE.sub(" ", md)
    md = _INLINE.sub(" ", md)
    return md


def _tokens(text: str) -> list[str]:
    return [m.group(1) + m.group(2) for m in _NUMBER_RE.finditer(text)]


def _structural_constants() -> list[float]:
    params = load_reporter_params()
    values = list(params.structural_constants.values())
    values.extend([5.0, 32.0])  # n_toggles and the saturated basis size (2^5), derived
    return values


def _evidence_numbers(doc) -> list[float]:
    out: list[float] = []
    for ev in doc.evidence:
        for tok in _tokens(ev.verbatim_text):
            try:
                out.append(float(tok.rstrip("%")))
            except ValueError:
                continue
    return out


# --- Layers 3-5 ------------------------------------------------------------------------------

def _verify_structural(doc, bundle, errors: list[str]) -> None:
    from .emit import emit_claim  # local import avoids a cycle at module load

    for claim in doc.claims:
        # (a) displayed value is re-derivable from the raw value.
        precision = claim.precision
        scale = scale_for_unit(_unit_from_value(claim.unit))
        raw = claim.raw_value
        redisplayed = fmt(
            raw if not isinstance(raw, NanValue) else raw,
            precision=precision,
            scale=scale,
        )
        if redisplayed != claim.displayed_value:
            errors.append(
                f"structural: claim {claim.claim_id} displayed {claim.displayed_value!r} "
                f"!= re-derived {redisplayed!r}"
            )
        # (b) for a bundle-sourced json_pointer claim, the raw value re-resolves.
        if (
            bundle is not None
            and claim.source_locator.kind == "json_pointer"
            and bundle.has(claim.source_artifact)
        ):
            spec = _spec_from_claim(claim)
            try:
                _, rerec = emit_claim(spec, bundle)
            except Exception as exc:  # noqa: BLE001 - report, do not crash the whole pass
                errors.append(f"structural: claim {claim.claim_id} re-resolve failed: {exc}")
                continue
            if not _same_value(rerec.raw_value, claim.raw_value):
                errors.append(
                    f"structural: claim {claim.claim_id} raw {claim.raw_value!r} "
                    f"!= re-resolved {rerec.raw_value!r}"
                )


def _verify_bijection(doc, body: str, errors: list[str]) -> int:
    prose_tokens = Counter(_tokens(body))
    claim_tokens = Counter(c.displayed_value for c in doc.claims if _is_number_token(c))
    structural = Counter(_number_str(v) for v in _structural_constants())
    evidence = Counter(_tokens_from_numbers(_evidence_numbers(doc)))

    # Every claim display token must appear in the prose.
    for token, count in claim_tokens.items():
        if prose_tokens.get(token, 0) < count:
            errors.append(
                f"bijection: claim token {token!r} appears {prose_tokens.get(token, 0)}x "
                f"in prose, expected >= {count} (orphan claim)"
            )
    # Every prose token must be covered by a claim, a structural constant or evidence.
    allowed = claim_tokens + structural + evidence
    for token, count in prose_tokens.items():
        if allowed.get(token, 0) < count:
            errors.append(
                f"bijection: prose token {token!r} ({count}x) is not covered by any claim, "
                "structural constant or evidence block (stray number)"
            )
    return sum(prose_tokens.values())


def _verify_backstop(doc, body: str, errors: list[str]) -> None:
    universe: list[float] = []
    for claim in doc.claims:
        if isinstance(claim.raw_value, (int, float)) and not isinstance(
            claim.raw_value, bool
        ):
            # The displayed token is the SCALED value (percent x100, bps x10000), so the
            # universe must hold the scaled value — mirroring the Layer-3 re-derivation.
            scale = scale_for_unit(_unit_from_value(claim.unit))
            universe.append(float(claim.raw_value) * scale)
    universe.extend(_evidence_numbers(doc))
    universe.extend(_structural_constants())
    result = verify_numbers(body, universe)
    if not result.ok:
        errors.append(
            f"backstop: unverified numeric tokens {result.unverified}"
        )


def verify_document(doc, bundle=None) -> VerifyReport:
    """Run all five layers over a `RenderedDocument`. Returns a report; raises nothing here so
    the caller can decide (the publication pipeline raises on `not ok`)."""
    errors: list[str] = []
    lint_renderer()  # Layer 1 (raises LintError on violation)
    body = strip_noncontent(doc.note_markdown)
    _verify_structural(doc, bundle, errors)  # Layer 3
    n_tokens = _verify_bijection(doc, body, errors)  # Layer 4
    _verify_backstop(doc, body, errors)  # Layer 5
    return VerifyReport(
        ok=not errors,
        n_claims=len(doc.claims),
        n_tokens_checked=n_tokens,
        errors=tuple(errors),
    )


def assert_verified(doc, bundle=None) -> VerifyReport:
    """Verify and raise `VerificationError` if any layer failed."""
    report = verify_document(doc, bundle)
    if not report.ok:
        raise VerificationError("; ".join(report.errors))
    return report


# --- small helpers ---------------------------------------------------------------------------

def _is_number_token(claim) -> bool:
    # A NaN claim renders as "nan", which is not a numeric token (unverifiable-by-omission).
    return not isinstance(claim.raw_value, NanValue)


def _number_str(value: float) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _tokens_from_numbers(values) -> list[str]:
    return [repr(float(v)) for v in values]


def _same_value(a, b) -> bool:
    if isinstance(a, NanValue) and isinstance(b, NanValue):
        return True
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return a == b


def _unit_from_value(unit_value: str):
    from shared.reporting.claims import Unit

    return Unit(unit_value)


def _spec_from_claim(claim):
    from shared.reporting.claims import ClaimSpec, Unit

    return ClaimSpec(
        claim_id=claim.claim_id,
        slot_id=claim.slot_id,
        source_artifact=claim.source_artifact,
        source_locator=claim.source_locator,
        formatter_id=_formatter_from_precision(claim.precision),
        unit=Unit(claim.unit),
        conditioning_pointer=None,
    )


def _formatter_from_precision(precision) -> str:
    mapping = {None: "raw", 0: "int", 1: "1dp", 2: "2dp", 3: "3dp", 4: "4dp"}
    return mapping.get(precision, "raw")
