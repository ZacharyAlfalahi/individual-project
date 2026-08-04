"""JSON Pointer resolution (docs/reporter/reporter_spec_v0.2.md §8.4): RFC-6901 semantics, fail-closed (INV-13)."""

from __future__ import annotations

import pytest

from shared.reporting.claims import SourceLocator
from shared.reporting.resolve import (
    PointerResolutionError,
    resolve_json_pointer,
    resolve_locator,
)

DOC = {
    "summary": {"sharpe": 0.42, "t_stat": 2.1},
    "legs": [{"n_groups": 5}, {"n_groups": 10}],
    "a/b": {"~x": 1},
    "maybe": None,
}


def test_empty_pointer_returns_whole_doc():
    assert resolve_json_pointer(DOC, "") is DOC


def test_nested_object_pointer():
    assert resolve_json_pointer(DOC, "/summary/sharpe") == 0.42


def test_array_index_pointer():
    assert resolve_json_pointer(DOC, "/legs/0/n_groups") == 5
    assert resolve_json_pointer(DOC, "/legs/1/n_groups") == 10


def test_escaping_tilde_and_slash():
    # "a/b" as a key is encoded /a~1b ; "~x" is encoded /~0x
    assert resolve_json_pointer(DOC, "/a~1b/~0x") == 1


def test_present_null_is_returned_not_a_miss():
    assert resolve_json_pointer(DOC, "/maybe") is None


def test_missing_key_fails_closed():
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer(DOC, "/summary/missing")


def test_index_out_of_range_fails_closed():
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer(DOC, "/legs/9")


def test_non_index_token_on_array_fails_closed():
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer(DOC, "/legs/first")


def test_indexing_a_scalar_fails_closed():
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer(DOC, "/summary/sharpe/deeper")


def test_pointer_without_leading_slash_fails_closed():
    with pytest.raises(PointerResolutionError):
        resolve_json_pointer(DOC, "summary")


def test_resolve_locator_uses_pointer():
    loc = SourceLocator(kind="json_pointer", pointer="/summary/t_stat")
    assert resolve_locator(loc, DOC) == 2.1
