"""Projections (docs/reporter/reporter_spec_v0.2.md §8.4, C6): field selection, no arithmetic, DataFrame excluded."""

from __future__ import annotations

import ast
import inspect

import pytest

import agents.reporter.projections as proj
from agents.reporter.projections import (
    STRATEGY_RESULT_PROJECTION_ID,
    UnknownProjectionError,
    project_quant_config,
    project_strategy_result,
    resolve_projection,
)


def test_strategy_result_projection_from_dict_selects_summary():
    persisted = {
        "strategy_label": "drf",
        "summary": {"sharpe": 0.4, "t_stat": 2.1},
        "variant": False,
        "n_legs": 2,
    }
    out = project_strategy_result(persisted)
    assert out["summary"]["sharpe"] == 0.4
    assert out["strategy_label"] == "drf"


def test_strategy_result_projection_excludes_monthly_returns():
    pd = pytest.importorskip("pandas")
    from agents.quant.config.runner import StrategyResult

    sr = StrategyResult(
        strategy_label="drf",
        monthly_returns=pd.DataFrame({"r": [0.01, 0.02]}),
        summary={"sharpe": 0.4},
        relationship_to_benchmark={},
        settings_used={},
        bookkeeping={},
        variant=False,
        n_legs=2,
        combiner={},
    )
    out = project_strategy_result(sr)
    assert "monthly_returns" not in out
    assert out["summary"]["sharpe"] == 0.4


def test_quant_config_projection_unwraps_inherited():
    from types import SimpleNamespace

    class _Inh:
        def __init__(self, value):
            self.value = value

    cfg = SimpleNamespace(
        strategy_id="drf",
        groups=_Inh(5),
        weighting=_Inh("size"),
        signal_lag=_Inh(1),
        min_bonds=_Inh(20),
        long_group=_Inh(5),
        short_group=_Inh(1),
        control_groups=None,
        holding_period=_Inh(1),
    )
    out = project_quant_config(cfg)
    assert out["groups"] == 5
    assert out["weighting"] == "size"
    assert "control_groups" not in out  # None dropped


def test_resolve_projection_unknown_id_fails_closed():
    with pytest.raises(UnknownProjectionError):
        resolve_projection("no_such_projection_v9", {})


def test_resolve_projection_dispatches():
    out = resolve_projection(STRATEGY_RESULT_PROJECTION_ID, {"strategy_label": "x"})
    assert out["strategy_label"] == "x"


def test_projections_contain_no_arithmetic():
    # A projection SELECTS and SERIALISES; it must compute no research quantity (§8.4).
    tree = ast.parse(inspect.getsource(proj))
    arith = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.MatMult,
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, arith):
            raise AssertionError(f"arithmetic BinOp in projections at line {node.lineno}")
        if isinstance(node, ast.AugAssign) and isinstance(node.op, arith):
            raise AssertionError(f"augmented arithmetic in projections at line {node.lineno}")
