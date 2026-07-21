"""
numeric_verifier.py — every numeric token in prose must trace to a typed field
(§11, the Reporter discipline applied to the Auditor's explainer).

The LLM explainer runs strictly AFTER the deterministic verdict and originates
zero numbers (§11). This verifier enforces that: it extracts every numeric token
from the explainer's prose and checks each one traces to a value in the
`AuditCore` (its `to_dict()`), accounting for display rounding and the
fraction<->percent convention (a "41%" in prose traces to a stored share of 0.41).
Any token that does not trace is reported as unverified — the prose is rejected
before commit.

This module is deterministic and contains no language model itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A signed decimal / integer / scientific number, optionally followed by '%'.
_NUMBER_RE = re.compile(
    r"(?<![\w.])([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(%?)"
)


def _collect_report_numbers(obj) -> list[float]:
    """Recursively collect every numeric leaf (int/float, not bool) from a nested
    dict/list structure."""
    out: list[float] = []
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_collect_report_numbers(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_report_numbers(v))
    return out


def _decimals(token: str) -> int:
    """Number of fractional digits displayed (0 for an integer). Scientific
    notation is treated as high-precision (return a large value)."""
    if "e" in token or "E" in token:
        return 12
    if "." in token:
        return len(token.split(".", 1)[1])
    return 0


def _half_ulp(decimals: int) -> float:
    return 0.5 * (10.0 ** (-decimals))


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    n_checked: int
    unverified: tuple[str, ...]


def verify_numbers(
    prose: str, report: dict, *, rel_tol: float = 1e-9
) -> VerificationResult:
    """Verify that every numeric token in `prose` traces to a number in `report`
    (an AuditCore.to_dict()). A token matches a report number if the report number
    is within half a display-ulp of the token (so display rounding is allowed), on
    either the literal or the /100 (percent) reading."""
    report_numbers = _collect_report_numbers(report)
    unverified: list[str] = []
    n_checked = 0

    for match in _NUMBER_RE.finditer(prose):
        raw, pct = match.group(1), match.group(2)
        n_checked += 1
        value = float(raw)
        dec = _decimals(raw)
        candidates = [(value, dec)]
        if pct == "%":
            candidates.append((value / 100.0, dec + 2))

        matched = False
        for cand, cdec in candidates:
            tol = _half_ulp(cdec) + rel_tol * (1.0 + abs(cand))
            if any(abs(r - cand) <= tol for r in report_numbers):
                matched = True
                break
        if not matched:
            unverified.append(raw + pct)

    return VerificationResult(
        ok=not unverified, n_checked=n_checked, unverified=tuple(unverified)
    )
