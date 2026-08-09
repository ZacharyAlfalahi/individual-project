"""one-shot holdout window + sensitivity sub-window derivation (spec §1.2)."""

from __future__ import annotations

import pytest

from agents.scientist.experimentalist.oneshot_holdout.windows import (
    Window,
    add_months,
    derive_sensitivity_subwindow,
    months_inclusive,
    registered_window,
)


def test_registered_window_validates_span():
    w = registered_window(("2022-01", "2025-09", 45))
    assert (w.start, w.end, w.n_months) == ("2022-01", "2025-09", 45)


def test_registered_window_rejects_wrong_span():
    with pytest.raises(ValueError):
        Window("2022-01", "2025-09", 44).validate()


def test_sensitivity_subwindow_is_derived_not_configured():
    # From the registered 45-month window the tagged 36-month sub-window is derived: shared
    # start, ends 35 months later -> 2022-01..2024-12.
    reg = registered_window(("2022-01", "2025-09", 45))
    sub = derive_sensitivity_subwindow(reg)
    assert (sub.start, sub.end, sub.n_months) == ("2022-01", "2024-12", 36)


def test_sensitivity_subwindow_lies_inside_registered():
    reg = registered_window(("2022-01", "2025-09", 45))
    sub = derive_sensitivity_subwindow(reg)
    assert months_inclusive(reg.start, sub.end) <= reg.n_months


def test_sensitivity_subwindow_refuses_shorter_registered_window():
    # A mis-registered window shorter than the sub-window must fail loud, not silently shrink.
    short = Window("2022-01", "2024-11", 35)
    with pytest.raises(ValueError):
        derive_sensitivity_subwindow(short)


def test_add_months_arithmetic():
    assert add_months("2022-01", 35) == "2024-12"
    assert add_months("2022-01", -39) == "2018-10"
    assert months_inclusive("2022-01", "2025-09") == 45
