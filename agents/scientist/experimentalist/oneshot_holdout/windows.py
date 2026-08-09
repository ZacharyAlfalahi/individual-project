"""Window and sensitivity sub-window derivation (one-shot holdout §1.2).

The registered evaluation window comes from ``holdout.assert_evaluation_window()``
(SC-SCI-12, the two-source guard). The tagged 36-month sensitivity sub-window is
**derived** from it here — never separately configured — so the SC-SCI-12 window move
stays "one value everywhere, mechanically" (SC-SCI-13 clause 3). Month-grain arithmetic
on ``YYYY-MM`` strings; no dates, no run-time choices.
"""

from __future__ import annotations

from dataclasses import dataclass

# The tagged sensitivity sub-window length (2022-01..2024-12), SC-SCI-13. Derived — the
# sub-window shares the registered start and ends this many months later.
SENSITIVITY_N_MONTHS = 36


def _parse_ym(ym: str) -> tuple[int, int]:
    parts = str(ym).split("-")
    if len(parts) != 2:
        raise ValueError(f"expected 'YYYY-MM'; got {ym!r}")
    return int(parts[0]), int(parts[1])


def _month_index(ym: str) -> int:
    year, month = _parse_ym(ym)
    if not (1 <= month <= 12):
        raise ValueError(f"month out of range in {ym!r}")
    return year * 12 + (month - 1)


def _from_index(idx: int) -> str:
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def add_months(ym: str, months: int) -> str:
    return _from_index(_month_index(ym) + months)


def months_inclusive(start: str, end: str) -> int:
    """Inclusive count of months in [start, end] (both 'YYYY-MM')."""
    return _month_index(end) - _month_index(start) + 1


@dataclass(frozen=True)
class Window:
    start: str
    end: str
    n_months: int

    def validate(self) -> "Window":
        got = months_inclusive(self.start, self.end)
        if got != self.n_months:
            raise ValueError(
                f"window {self.start}..{self.end} spans {got} months, not n_months={self.n_months}"
            )
        return self


def registered_window(triple: tuple[str, str, int]) -> Window:
    """Wrap the ``(start, end, n_months)`` returned by ``assert_evaluation_window()``."""
    start, end, n = triple
    return Window(start=str(start), end=str(end), n_months=int(n)).validate()


def derive_sensitivity_subwindow(registered: Window) -> Window:
    """The tagged 36-month sub-window: shares the registered start, ends 35 months later.

    Fails loud if the registered window is shorter than the sub-window (the sub-window
    must lie inside it) — a guard against a mis-registered window silently shrinking.
    """
    registered.validate()
    if registered.n_months < SENSITIVITY_N_MONTHS:
        raise ValueError(
            f"registered window ({registered.n_months} mo) is shorter than the "
            f"{SENSITIVITY_N_MONTHS}-month sensitivity sub-window"
        )
    end = add_months(registered.start, SENSITIVITY_N_MONTHS - 1)
    return Window(start=registered.start, end=end, n_months=SENSITIVITY_N_MONTHS).validate()
