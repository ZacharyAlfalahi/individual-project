"""
Ledger enumeration backstop (WS-4) — the pre-corpus integrity gate.

Two independent nets over the D29 assumptions-ledger enumeration
(``agents/quant/config/ledger_check.py`` + ``data/ledger_check_table.yaml``):

  (1) STRUCTURAL completeness — every checkable ledger row names a field that
      actually resolves on a real ``StrategySpec``, and the three supported anchors
      pass the D28/D29 gate with ZERO mismatches. This proves that wiring
      ``check_assumptions`` into ``run_quant.run_all`` cannot spuriously refuse a
      supported anchor.

  (2) REACHED-vs-LOAD-BEARING — a committed generalisation of the throwaway
      ``docs/quant/validation/ledger_validation_artifacts/flip_checks.py``: the
      overlap engine's cohort-averaging choice (equal-weight vs bond-count-weighted
      concurrent cohorts) is LOAD-BEARING (flipping it moves the headline), yet it
      is an ENGINE-INTERNAL choice — not a Part 2 field — so the Part2-scoped ledger
      structurally cannot enumerate it. This pins the enumeration's scope boundary
      (``assumptions_ledger_v2.md`` §6, "what this does NOT close") as a tripwire a
      corpus run must respect.

Reads ``data/development/`` only (the autouse holdout guard enforces it). No repo
files are modified; the flip is a faithful re-aggregation of the SAME cohort
contributions the engine already computed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.quant.config.ledger_check import (
    _field_for,
    check_assumptions,
    load_ledger_check_table,
)
from evaluation.gold_specs.gold_loader import load_gold_spec

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "development"
ANCHORS = ("str", "drf", "mom6")


# --- (1) structural completeness (offline) --------------------------------------

def test_anchors_produce_zero_ledger_refusals():
    # The D28/D29 gate now runs inside run_quant.run_all; a supported anchor must pass it with
    # zero mismatches, else wiring it would flip a running anchor to a spurious refusal.
    table = load_ledger_check_table()
    for a in ANCHORS:
        refusals = check_assumptions(load_gold_spec(a), table)
        assert refusals == (), (
            f"{a} unexpectedly hit the ledger gate: {[r.to_dict() for r in refusals]}"
        )


def test_every_ledger_row_field_resolves_on_a_real_spec():
    # The 'reached' side: a ledger row naming a field absent from the schema would be a silent
    # no-fire. _field_for raises on an unknown field, so resolving every row's field against a
    # real spec proves the enumeration references live fields only.
    table = load_ledger_check_table()
    spec = load_gold_spec("drf")
    for row in table.rows:
        if row.block == "common":
            _field_for(spec.part2, row.part2_field, "Part2")            # raises if dead
        else:
            for i, leg in enumerate(spec.part2.legs):
                _field_for(leg, row.part2_field, f"legs[{i}]")


# --- (2) reached-vs-load-bearing: the mom6 cohort-averaging flip -----------------

@pytest.fixture(scope="module")
def _mom6_overlap_panel():
    from agents.quant.library.run_config import (
        ConstructionConfig,
        EvaluationConfig,
        PanelViewConfig,
        RunConfig,
    )
    from agents.quant.library.views import view

    if not (DEV / "monthly_panel_total_return.parquet").exists():
        pytest.skip("requires the licensed dev panel (data/development/) — not shipped; see README")
    maximal = pd.read_parquet(DEV / "monthly_panel_total_return.parquet")
    mom6_signal = pd.read_parquet(DEV / "signals" / "mom6.parquet")
    vcfg = RunConfig(
        panel_view=PanelViewConfig(price_family="corr", stale_mask=False, include_terminal_rows=False),
        construction=ConstructionConfig(signal_lag=1, expost_trim="none"),
        evaluation=EvaluationConfig(),
    )
    mp = (
        view(maximal, vcfg, signals=mom6_signal)
        .drop_duplicates(subset=["cusip", "date"])
        .reset_index(drop=True)
    )
    return mp[["cusip", "date", "ret", "size", "mom6"]]


def test_mom6_cohort_averaging_is_load_bearing_but_out_of_ledger_scope(_mom6_overlap_panel):
    from agents.quant.library.characteristic_sort import (
        _apply_defaults,
        extract_monthly_selections,
    )
    from agents.quant.library.overlap import _leg_return_at, run_with_holding_period

    mp = _mom6_overlap_panel
    rb = {"score": "mom6", "groups": 10, "weighting": "equal",
          "long_group": 9, "short_group": 0, "signal_lag": 1, "nw_lags": None}
    orig = run_with_holding_period(mp, rb, holding_period=6)
    mean_orig = orig["strategy_ret"].mean() * 100

    # Faithful re-aggregation of the SAME cohort contributions (mirrors the engine body).
    weighting = _apply_defaults(rb)["weighting"]
    selections = extract_monthly_selections(mp, rb)
    panel_lookup = dict(zip(zip(mp["cusip"].values, mp["date"].values), mp["ret"].values))
    contributions: dict = {}
    for formation_t, sel in selections.items():
        for h in range(1, 7):
            m = formation_t + pd.offsets.MonthEnd(h)
            long_r, n_long = _leg_return_at(sel["long"], m, panel_lookup, weighting)
            short_r, n_short = _leg_return_at(sel["short"], m, panel_lookup, weighting)
            if long_r is None or short_r is None:
                continue
            contributions.setdefault(m, []).append((long_r, short_r, n_long + n_short))

    eq, cw = [], []
    for m in sorted(contributions):
        tr = contributions[m]
        longs = np.array([t[0] for t in tr])
        shorts = np.array([t[1] for t in tr])
        w = np.array([t[2] for t in tr], dtype=float)
        w = w / w.sum()
        eq.append(float(longs.mean() - shorts.mean()))            # equal-weight (== engine)
        cw.append(float((w * longs).sum() - (w * shorts).sum()))  # bond-count-weighted cohorts
    mean_eq = np.mean(eq) * 100
    mean_cw = np.mean(cw) * 100

    # Sanity: the equal-weight reconstruction reproduces the engine (the flip is faithful).
    assert mean_eq == pytest.approx(mean_orig, abs=1e-9)
    # Load-bearing: flipping cohort-averaging to bond-count-weighted MOVES the headline. This
    # engine-internal choice (overlap.py) is NOT a Part 2 field, so the Part2-scoped D29 ledger
    # structurally cannot enumerate it — the documented §6 scope boundary. A non-trivial delta
    # pins it as load-bearing (directional, never a pinned level).
    assert abs(mean_cw - mean_orig) > 1e-3, (
        f"cohort-averaging flip moved the headline by only {mean_cw - mean_orig:+.6f} %/mo; "
        "expected a load-bearing move"
    )
