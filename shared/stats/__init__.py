"""shared/stats — R1: WRAP, DON'T MOVE.

Re-export the Auditor's frozen, pre-registered statistics (tag `auditor-prereg-2026-07-22`)
UNCHANGED, so the Scientist reuses one implementation and one convention. Only CPCV is net-new
(see cpcv.py). Zero edits under agents/auditor/ — that is the whole point of wrap-not-move: the
Auditor suite stays trivially green.

The Scientist package (agents/scientist/**) is import-walled from agents.auditor.checks.economic
(a forbidden module, R2/INVARIANT 1). Routing deflated_sharpe_ratio et al. through this shared
surface is exactly how the deterministic Experimentalist obtains them without breaching the wall.
"""

from __future__ import annotations

from agents.auditor.checks.bootstrap import (
    BootstrapResult,
    block_length,
    circular_block_indices,
    effective_blocks,
    run_bootstrap,
)
from agents.auditor.checks.economic import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
)
from agents.auditor.checks.fdr import (
    FdrDecision,
    FdrReport,
    benjamini_hochberg,
    corpus_confirmatory_fdr,
    run_fdr,
)

from shared.stats.cpcv import (
    CPCVResult,
    cpcv_evaluate,
    cpcv_folds,
    n_backtest_paths,
    partition_groups,
)
from shared.stats.posterior import (
    PosteriorSummary,
    PriorPosterior,
    alpha_se_from_t,
    posterior_summary,
)

__all__ = [
    # FDR (fdr.py)
    "FdrDecision",
    "FdrReport",
    "benjamini_hochberg",
    "corpus_confirmatory_fdr",
    "run_fdr",
    # Deflated / probabilistic Sharpe (economic.py)
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "probabilistic_sharpe_ratio",
    # Block bootstrap primitives (bootstrap.py)
    "BootstrapResult",
    "block_length",
    "circular_block_indices",
    "effective_blocks",
    "run_bootstrap",
    # CPCV (net-new, cpcv.py)
    "CPCVResult",
    "cpcv_evaluate",
    "cpcv_folds",
    "n_backtest_paths",
    "partition_groups",
    # Scalar normal–normal posterior layer (net-new, posterior.py — WS-A/P3)
    "PosteriorSummary",
    "PriorPosterior",
    "alpha_se_from_t",
    "posterior_summary",
]
