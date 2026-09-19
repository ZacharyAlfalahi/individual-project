"""Benchmark-integrity regression (Workstream B — B2's regression target, re-homed).

Pins the four **BBW-4 benchmark** components against silent builder drift:
``drf_corr``, ``crf_corr``, ``lrf_corr`` (``data/development/factors/bbw_factors.parquet``)
and ``mktb_corr`` (``data/development/factors/mktb.parquet``). These factors are
NOT protected here as the P1 oracle — they are the **RQ4 primary benchmark**:
``mktb.parquet -> factors_mktb`` and ``bbw_factors.parquet -> factors_bbw`` both
feed the BBW-4 frame (``agents/scientist/experimentalist/oneshot_holdout/panel_builder.py``),
which reaches ``stage2_evaluate`` ("BBW-4 primary") and
``regress_on_benchmark(cand, bbw4_factors)`` at G3
(``agents/scientist/experimentalist/{inference,orchestrator}.py``). A silent edit
to a builder that moved any component would make **every RQ4 alpha wrong** — a
failure reaching further than the P1 oracle.

The frozen expected levels are read from Workstream-D's recorded descriptive run
(``results/quant/descriptive/anchors/anchor_descriptive.json``, the
corr / total_return means), a separately-recorded artefact that a builder
rebuild does not regenerate — so a drifted rebuild of the parquet fails against
it here. Skips gracefully when the factor parquets or the recorded run are absent
(``data/`` not built in this checkout), mirroring ``test_codegen_oracle_sanity.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACTORS = _REPO_ROOT / "data" / "development" / "factors"
_BBW = _FACTORS / "bbw_factors.parquet"
_MKTB = _FACTORS / "mktb.parquet"
_DESCRIPTIVE = (
    _REPO_ROOT / "results" / "quant" / "descriptive" / "anchors" / "anchor_descriptive.json"
)

# Tight tolerance in %/mo: the rebuilt parquet is compared to the recorded descriptive run at
# 1e-4 %/mo (= 0.01 bp/mo) — immune to float-repr noise, yet failing on any material builder
# drift. The assertion is this tolerance, not exact equality.
_TOL_PCT_PER_MONTH = 1e-4

# BBW-4 component -> (parquet path, corr-family column).
_COMPONENTS: dict[str, tuple[Path, str]] = {
    "drf": (_BBW, "drf_corr"),
    "crf": (_BBW, "crf_corr"),
    "lrf": (_BBW, "lrf_corr"),
    "mktb": (_MKTB, "mktb_corr"),
}

pytestmark = pytest.mark.skipif(
    not (_BBW.exists() and _MKTB.exists() and _DESCRIPTIVE.exists()),
    reason="BBW-4 factor parquets or the recorded descriptive run are absent "
    "(data/ not built in this checkout)",
)


def _expected_level_pct(factor: str) -> float:
    """The frozen corr / total_return mean (%/mo) from Workstream-D's recorded run."""
    doc = json.loads(_DESCRIPTIVE.read_text(encoding="utf-8"))
    return float(doc["table"]["total_return"][factor]["corr"]["mean_pct_per_month"])


def _built_level_pct(path: Path, column: str) -> float:
    """The committed factor's own mean level (%/mo), warm-up NaNs dropped."""
    frame = pd.read_parquet(path)
    assert column in frame.columns, f"{path.name} has no BBW-4 column {column!r}"
    return float(frame[column].dropna().mean() * 100.0)


@pytest.mark.parametrize("factor", sorted(_COMPONENTS))
def test_bbw4_component_level_matches_recorded_run(factor):
    """A builder edit that silently moved this RQ4 benchmark factor fails here."""
    path, column = _COMPONENTS[factor]
    built = _built_level_pct(path, column)
    expected = _expected_level_pct(factor)
    assert abs(built - expected) <= _TOL_PCT_PER_MONTH, (
        f"BBW-4 benchmark component {factor} ({path.name}:{column}) drifted: "
        f"built {built:.6f} %/mo vs committed {expected:.6f} %/mo "
        f"(delta {abs(built - expected):.2e} > {_TOL_PCT_PER_MONTH:.0e}) — "
        "a builder edit silently moved an RQ4 benchmark factor."
    )
