"""one-shot holdout window derivation (spec §1.2)."""

from __future__ import annotations

import pytest

from agents.scientist.experimentalist.oneshot_holdout.windows import (
    Window,
    add_months,
    months_inclusive,
    registered_window,
)


def test_registered_window_validates_span():
    w = registered_window(("2022-01", "2025-09", 45))
    assert (w.start, w.end, w.n_months) == ("2022-01", "2025-09", 45)


def test_registered_window_rejects_wrong_span():
    with pytest.raises(ValueError):
        Window("2022-01", "2025-09", 44).validate()


def test_add_months_arithmetic():
    assert add_months("2022-01", 35) == "2024-12"
    assert add_months("2022-01", -39) == "2018-10"
    assert months_inclusive("2022-01", "2025-09") == 45
