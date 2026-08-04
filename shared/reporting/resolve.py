"""resolve.py — RFC-6901 JSON Pointer resolution for `SourceLocator` (docs/reporter/reporter_spec_v0.2.md §8.4).

A claim's value is addressed by a `SourceLocator` into a serialised structure: either the
artefact's own `to_dict()` output (`json_pointer`) or a Reporter-owned projection
(`projection_pointer`). Resolution FAILS CLOSED — an unresolved pointer raises
`PointerResolutionError`, never returns a default (INV-13). This matters because upstream
`to_dict()` renames keys (e.g. `saturated` -> `saturated_bases`) and drops `None` fields
(`Evidence.to_dict`), so a stale pointer must surface loudly at verification time.

This module resolves a pointer against an ALREADY-FETCHED dict. The caller (renderer)
decides which dict to pass — the artefact's `to_dict()` for a `json_pointer`, or the
projection for a `projection_pointer` — so this module stays free of artefact knowledge.
"""

from __future__ import annotations

import re

from .claims import SourceLocator

# RFC-6901 array index: ASCII "0" or a non-zero-leading run of ASCII digits. Deliberately
# excludes non-ASCII "digits" (e.g. superscripts, Arabic-Indic) that `str.isdigit()` accepts.
_INDEX_RE = re.compile(r"0|[1-9][0-9]*")


class PointerResolutionError(KeyError):
    """A JSON Pointer failed to resolve against the given document. Fail-closed (INV-13):
    a missing key, a missing list index, or indexing a non-container all raise here."""

    def __init__(self, pointer: str, reason: str) -> None:
        self._pointer = pointer
        super().__init__(f"JSON Pointer {pointer!r} did not resolve: {reason}")


def _unescape(token: str) -> str:
    """RFC-6901 token unescape: ``~1`` -> ``/`` and ``~0`` -> ``~`` (in that order)."""
    return token.replace("~1", "/").replace("~0", "~")


def resolve_json_pointer(doc: object, pointer: str) -> object:
    """Resolve an RFC-6901 JSON Pointer against `doc`. ``""`` returns the whole document.
    Raises `PointerResolutionError` on any missing token — the key/index is absent, or a
    non-container is indexed. Note that a resolved value may legitimately be ``None`` (a
    present JSON null); that is returned, not treated as a miss."""
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise PointerResolutionError(pointer, "must be '' or start with '/'")

    node = doc
    # Split drops the leading '' before the first '/'.
    for raw in pointer.split("/")[1:]:
        token = _unescape(raw)
        if isinstance(node, dict):
            if token not in node:
                raise PointerResolutionError(
                    pointer, f"key {token!r} absent from object"
                )
            node = node[token]
        elif isinstance(node, (list, tuple)):
            if not _is_index(token):
                raise PointerResolutionError(
                    pointer, f"token {token!r} is not a valid array index"
                )
            idx = int(token)
            if idx >= len(node):
                raise PointerResolutionError(
                    pointer, f"index {idx} out of range (len {len(node)})"
                )
            node = node[idx]
        else:
            raise PointerResolutionError(
                pointer,
                f"cannot index {type(node).__name__} with token {token!r}",
            )
    return node


def _is_index(token: str) -> bool:
    """A JSON-Pointer array index is a non-negative ASCII integer with no leading zeros
    (except the single character ``"0"``). ``"-"`` (append) is deliberately not supported,
    and non-ASCII digit forms are rejected so a bad token fails closed rather than either
    leaking a raw ``ValueError`` from ``int()`` or silently aliasing an element."""
    return _INDEX_RE.fullmatch(token) is not None


def resolve_locator(locator: SourceLocator, doc: object) -> object:
    """Resolve a `SourceLocator`'s pointer against the already-fetched `doc` (the artefact's
    `to_dict()` output for a json_pointer, or the projection for a projection_pointer)."""
    return resolve_json_pointer(doc, locator.pointer)
