"""WS-C — codegen artefacts must be STRICT JSON: `json.dumps` writes bare NaN by default,
which a conforming parser rejects."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation.codegen.jsonio import dump_json, json_safe  # noqa: E402


def test_non_finite_values_become_null_and_the_file_parses_strictly():
    payload = {"correlation": float("nan"), "bound": float("inf"),
               "rows": [1.5, float("-inf")], "n": 3, "name": "drf"}

    text = dump_json(payload)
    back = json.loads(text)          # strict by default: would raise on bare NaN

    assert back["correlation"] is None
    assert back["bound"] is None
    assert back["rows"] == [1.5, None]
    assert back["n"] == 3 and back["name"] == "drf"


def test_finite_values_are_untouched():
    assert json_safe({"a": 0.5, "b": [1, 2.25]}) == {"a": 0.5, "b": [1, 2.25]}


def test_nested_structures_are_sanitised():
    nested = {"metrics": {"per_member": [{"corr": float("nan")}]}}
    assert json.loads(dump_json(nested))["metrics"]["per_member"][0]["corr"] is None


def test_a_non_finite_value_that_escapes_sanitising_raises_rather_than_writing():
    class Sneaky(dict):
        """Escapes json_safe (it is dict-like only at dump time)."""

    with pytest.raises(ValueError):
        json.dumps({"x": math.nan}, allow_nan=False)
