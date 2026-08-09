"""SC-SCI-13 holdout bootstrap — provably inert + no-holdout (mirrors P3 / G4-PBO patterns).

Four structural proofs: (1) no Scientist gate module references the diagnostic, so no gate
decision can depend on it; (2) the module performs NO I/O — so it cannot, by construction,
read `/data/holdout/` (the guarantee is a test, not a promise); (3) no back-edge import into
the Scientist; (4) the wrapper never mutates its inputs; (5) a below-floor cell is LABELLED,
not raised (it deliberately does not route through the Auditor driver's refusal path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.stats.holdout_bootstrap import (
    holdout_inference_window_sensitivity,
    paired_difference_bootstrap,
)

_ROOT = Path(__file__).resolve().parents[2]
_GATE_DIR = _ROOT / "agents" / "scientist" / "experimentalist"
_MODULE = _ROOT / "shared" / "stats" / "holdout_bootstrap.py"

# I/O-CALL tokens. A pure module that contains none of these cannot open any file — so it
# cannot read the holdout, whatever its docstring says. (The docstring intentionally names
# `/data/holdout/` to state it does NOT read it, so we scan for I/O *calls*, not path text.)
_IO_TOKENS = (
    "open(", "read_parquet", "read_csv", "scan_parquet", "scan_csv",
    "ParquetFile", ".to_parquet", ".to_csv", "pd.read", "np.load",
    "import os", "import pathlib", "from pathlib",
)

FACTORS = ("mktb", "drf", "crf", "lrf")


def _synthetic():
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-31", periods=45, freq="ME")
    fr = pd.DataFrame({"date": dates, **{f: rng.normal(scale=0.02, size=45) for f in FACTORS}})
    surv = pd.Series(0.003 + rng.normal(scale=0.001, size=45) + 0.5 * fr["mktb"].to_numpy(), index=dates)
    par = pd.Series(0.001 + rng.normal(scale=0.001, size=45) + 0.5 * fr["mktb"].to_numpy(), index=dates)
    return surv, par, fr


def test_no_gate_module_imports_holdout_bootstrap():
    for path in sorted(_GATE_DIR.glob("*.py")):
        src = path.read_text(encoding="utf-8")
        assert "holdout_bootstrap" not in src, f"{path.name} references holdout_bootstrap"
        assert "holdout_inference_window_sensitivity" not in src, path.name


def test_module_does_no_io():
    src = _MODULE.read_text(encoding="utf-8")
    for token in _IO_TOKENS:
        assert token not in src, f"holdout_bootstrap.py contains an I/O token {token!r}"


def test_module_does_not_import_the_scientist():
    assert "agents.scientist" not in _MODULE.read_text(encoding="utf-8")


def test_inputs_are_not_mutated():
    surv, par, fr = _synthetic()
    surv_b, par_b, fr_b = surv.copy(), par.copy(), fr.copy()
    holdout_inference_window_sensitivity(
        surv, par, fr, subwindow_cutoff=pd.Timestamp("2024-12-31"),
        n_replicates=50, min_effective_blocks=10, seed=1,
    )
    pd.testing.assert_series_equal(surv, surv_b)
    pd.testing.assert_series_equal(par, par_b)
    pd.testing.assert_frame_equal(fr, fr_b)


def test_below_floor_is_labelled_not_raised():
    diff = pd.Series(np.linspace(-0.01, 0.01, 45), index=pd.date_range("2022-01-31", periods=45, freq="ME"))
    (ci,) = paired_difference_bootstrap(
        diff, block_lengths=(6,), n_replicates=50, min_effective_blocks=10, seed=1
    )
    assert ci.effective_blocks == 7 and ci.floor_label == "below_floor"
    assert ci.ci_low is not None and ci.ci_high is not None    # a CI object, never an exception
