"""Strict-JSON writing for codegen artefacts (WS-C).

``json.dumps`` emits bare ``NaN`` / ``Infinity`` by default, which a conforming JSON parser
rejects — an artefact meant to be read by other tools must not require a lenient one. A
non-computable metric (an undefined correlation, say) is written as ``null``; the rendered
Markdown still shows it as ``NaN``, so nothing is lost to a human reader.
"""

from __future__ import annotations

import json
import math


def json_safe(node: object) -> object:
    """``node`` with every non-finite float replaced by ``None``, recursively."""
    if isinstance(node, float):
        return None if not math.isfinite(node) else node
    if isinstance(node, dict):
        return {k: json_safe(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [json_safe(v) for v in node]
    return node


def dump_json(payload: object) -> str:
    """Deterministic, strict JSON (``allow_nan=False`` is the guard: if a non-finite value
    somehow escapes ``json_safe``, this raises rather than writing an unparseable file)."""
    return json.dumps(json_safe(payload), indent=2, sort_keys=True, default=str,
                      allow_nan=False)
