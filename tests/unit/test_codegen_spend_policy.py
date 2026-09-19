"""The registered spend policy (`p1_codegen.budget`) read by `preflight.load_spend_policy`.

Validation happens BEFORE conversion on purpose: `float(True)` is `1.0` and `int(0.5)` is `0`, so a
coercing loader turns a malformed entry into a plausible-looking policy — an unbounded spend ceiling,
or a token floor that silently is not there. These pin each shape as a loud failure instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.codegen.preflight import load_spend_policy

_BLOCK = "p1_codegen:\n  budget:\n    per_vendor_usd_ceiling: {ceiling}\n    input_tokens_floor: {floor}\n"


def _write(tmp_path: Path, name: str, *, ceiling: str = "10.0", floor: str = "8000") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(_BLOCK.format(ceiling=ceiling, floor=floor), encoding="utf-8")
    return path


def test_the_shipped_policy_loads():
    ceiling, floor = load_spend_policy()
    assert ceiling > 0 and isinstance(floor, int) and floor > 0


@pytest.mark.parametrize("ceiling,label", [("true", "boolean"), (".inf", "infinite"),
                                           ("-5.0", "negative"), ("'10'", "string")])
def test_ceiling_rejects_malformed_values(tmp_path, ceiling, label):
    """An infinite or boolean ceiling would remove (or silently shrink) the live-run spend guard."""
    with pytest.raises(ValueError, match="per_vendor_usd_ceiling"):
        load_spend_policy(_write(tmp_path, f"ceiling_{label}", ceiling=ceiling))


@pytest.mark.parametrize("floor,label", [("0.5", "fractional"), ("0", "zero"),
                                         ("-1", "negative"), ("8000.0", "float"), ("true", "boolean")])
def test_floor_rejects_non_positive_integers(tmp_path, floor, label):
    """`int(0.5)` is 0 — a coercing loader would drop the floor entirely rather than complain."""
    with pytest.raises(ValueError, match="input_tokens_floor"):
        load_spend_policy(_write(tmp_path, f"floor_{label}", floor=floor))


def test_missing_block_is_a_loud_key_error(tmp_path):
    path = tmp_path / "bare.yaml"
    path.write_text("p1_codegen: {}\n", encoding="utf-8")
    with pytest.raises(KeyError, match="p1_codegen.budget"):
        load_spend_policy(path)
