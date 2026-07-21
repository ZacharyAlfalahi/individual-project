"""
G3 statistics (D34 mechanics, contract §3.3).

Wilson score intervals for proportions, and nothing else. Deliberately tiny: the
project declares no third-party dependencies (``pyproject.toml`` carries a ruff
config and no ``dependencies`` list), numpy is present only because the frozen
quant engine uses it, and scipy is absent. Wilson is ten lines of arithmetic, so
it is written rather than depended upon.

**Wilson, not Wald.** D34 names it, and the reason bites here: the counts are
small and frequently sit at 0 or n. The Wald interval ``p ± z·sqrt(p(1-p)/n)``
collapses to zero width at both ends -- it would report 0/18 as "0.0% [0.0%,
0.0%]", a false certainty from an interval that has simply degenerated. Wilson
stays bounded and asymmetric near the ends.

**n = 0 returns None, never NaN.** An empty cell is a real and expected outcome
on a three-anchor gold set (contract §3.2's DISAGREE arm is empty on the current
artefacts). ``None`` forces the renderer to say "n=0" rather than propagate a NaN
into a table, and makes a division-by-zero impossible rather than merely unlikely.

**The interval label rides in the type.** Contract §3.3: field-level intervals
are *descriptive*, the unit of generalisation is the observed gold corpus, and no
population-level claim is made. Carrying the label on ``Proportion`` rather than
adding it at render time means a caller cannot render an interval and forget to
say what kind of interval it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def wilson(k: int, n: int, z: float) -> tuple[float, float] | None:
    """Two-sided Wilson score interval for k successes in n trials.

    Returns ``None`` when ``n == 0`` -- an empty cell has no interval, and
    inventing one would be a number nobody observed."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"wilson: need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Proportion:
    """One proportion with its interval, its n, and its calibration status.

    ``numerator``/``denominator`` are kept as raw counts, and the renderer always
    shows them: contract §5.3's house style is "raw counts always shown ('1/18
    audited, 5.6%') -- never the percentage alone". A rate without its n is not
    interpretable on a corpus this size."""

    label: str
    numerator: int
    denominator: int
    z: float
    min_cell_n: int
    interval_label: str = "descriptive"
    note: str = ""

    @property
    def value(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.numerator / self.denominator

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.numerator, self.denominator, self.z)

    @property
    def calibrated(self) -> bool:
        """D34: cells below the pre-registered n bar report 'uncalibrated'.

        On a three-anchor gold set most per-field-type cells are expected to sit
        below it. That is the honest output, not an error state."""
        return self.denominator >= self.min_cell_n

    def render(self) -> str:
        if self.denominator == 0:
            return f"{self.label}: n=0"
        frac = f"{self.numerator}/{self.denominator}"
        pct = f"{self.value:.1%}"
        if not self.calibrated:
            return f"{self.label}: {frac} = {pct} (uncalibrated, n<{self.min_cell_n})"
        lo, hi = self.interval
        return (f"{self.label}: {frac} = {pct} "
                f"[{lo:.1%}, {hi:.1%}] ({self.interval_label})")
