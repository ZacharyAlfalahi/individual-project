"""
thresholds.py — typed, FAIL-LOUD loader for the Auditor's pre-registration
constants (design §13.1; open item O-A10 / O-A6).

The project's contribution is a critique of undisclosed researcher degrees of
freedom, and every numerical threshold the Auditor uses is a place where a nudge
in the expected direction would be individually defensible and collectively fatal
(§13.2). So these constants are NOT invented in code: they are pre-registered by
the researcher — with a cited source or an ex-ante justification — committed to
`docs/thresholds.yaml` under the `auditor:` block, and git-tagged before the first
confirmatory run.

This module therefore has ONE job: read those constants and **raise loudly if any
is absent**. A default would silently launder a post-hoc constant, which is the
exact sin the instrument exists to expose. Every loader raises
`AuditorThresholdError` (a subclass of KeyError) naming the missing key and the
design section that governs it.

For testability, every loader accepts an explicit `path` (like
`views._load_stale_theta_days` hardcodes the repo file); and every downstream
gate accepts an explicit override value so unit tests never need the real,
not-yet-pre-registered numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLDS_FILE = REPO_ROOT / "docs" / "thresholds.yaml"


class AuditorThresholdError(KeyError):
    """A required Auditor pre-registration constant is absent from
    docs/thresholds.yaml. Raised rather than defaulted, because a silent default
    would launder a post-hoc constant (design §13.2)."""

    def __init__(self, dotted_key: str, section: str) -> None:
        self._dotted_key = dotted_key
        # KeyError str-reprs its arg with quotes; pass a full sentence so the
        # message reads cleanly at the console.
        super().__init__(
            f"auditor pre-registration constant '{dotted_key}' is missing from "
            f"{THRESHOLDS_FILE.name} (governs design {section}). It must be "
            f"pre-registered with a cited source and git-tagged before the "
            f"confirmatory run — the Auditor never defaults it (§13.2)."
        )


# ---------------------------------------------------------------------------
# Raw block access
# ---------------------------------------------------------------------------

def _load_yaml(path: str | Path | None) -> dict:
    p = Path(path) if path is not None else THRESHOLDS_FILE
    with open(p) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise AuditorThresholdError("auditor", "§13.1 (thresholds.yaml is empty/malformed)")
    return data


def _require(block: dict, keys: tuple[str, ...], section: str) -> object:
    """Walk `keys` into nested dicts; raise AuditorThresholdError at the first
    missing level. `keys` is the dotted path under the top-level `auditor:` block
    (which is prepended for the error message)."""
    node: object = block
    walked: list[str] = ["auditor"]
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise AuditorThresholdError(".".join(walked + [key]), section)
        node = node[key]
        walked.append(key)
    return node


def _auditor_block(path: str | Path | None) -> dict:
    data = _load_yaml(path)
    block = data.get("auditor")
    if not isinstance(block, dict):
        raise AuditorThresholdError("auditor", "§13.1 (the whole auditor: block)")
    return block


def _require_number(value: object, dotted_key: str, section: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditorThresholdError(
            f"{dotted_key} (present but not a number: {value!r})", section
        )
    return float(value)


def _require_int(value: object, dotted_key: str, section: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditorThresholdError(
            f"{dotted_key} (present but not an int: {value!r})", section
        )
    return value


# ---------------------------------------------------------------------------
# Typed views
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SupportGate:
    """Minimum-common-support gate (§4.2, D-A40). A lattice can be mathematically
    complete and statistically unusable; below either threshold cross-cell
    nonlinear inference is refused or downgraded to descriptive."""

    min_common_months: int
    min_common_fraction_of_reference: float


@dataclass(frozen=True)
class BootstrapConfig:
    """Fixed-block synchronised bootstrap parameters (§6.2, D-A29). Block length is
    l = max(H_max, block_length_months); the bootstrap must satisfy l < T_common
    with at least `min_effective_blocks` (B_min) effective blocks."""

    n_replicates: int
    min_effective_blocks: int      # B_min
    block_length_months: int       # l_data_driven — the pre-committed floor


def load_support_gate(path: str | Path | None = None) -> SupportGate:
    """Read the §4.2 minimum-common-support gate. Raises if unregistered."""
    block = _auditor_block(path)
    mcm = _require(block, ("support_gate", "min_common_months"), "§4.2")
    mcf = _require(
        block, ("support_gate", "min_common_fraction_of_reference"), "§4.2"
    )
    return SupportGate(
        min_common_months=_require_int(
            mcm, "auditor.support_gate.min_common_months", "§4.2"
        ),
        min_common_fraction_of_reference=_require_number(
            mcf, "auditor.support_gate.min_common_fraction_of_reference", "§4.2"
        ),
    )


def load_bootstrap_config(path: str | Path | None = None) -> BootstrapConfig:
    """Read the §6.2 fixed-block bootstrap parameters. Raises if unregistered."""
    block = _auditor_block(path)
    n_rep = _require(block, ("bootstrap", "n_replicates"), "§6.2")
    b_min = _require(block, ("bootstrap", "min_effective_blocks"), "§6.2")
    blk = _require(block, ("bootstrap", "block_length_months"), "§6.2")
    return BootstrapConfig(
        n_replicates=_require_int(n_rep, "auditor.bootstrap.n_replicates", "§6.2"),
        min_effective_blocks=_require_int(
            b_min, "auditor.bootstrap.min_effective_blocks", "§6.2"
        ),
        block_length_months=_require_int(
            blk, "auditor.bootstrap.block_length_months", "§6.2"
        ),
    )


def load_primary_metric(path: str | Path | None = None) -> str:
    """The primary metric name (§9.1). One shared contract; the metric every
    cross-cell attribution and the FDR family are computed on."""
    block = _auditor_block(path)
    value = _require(block, ("primary_metric",), "§9.1")
    if not isinstance(value, str) or not value.strip():
        raise AuditorThresholdError(
            "auditor.primary_metric (present but not a non-empty string)", "§9.1"
        )
    return value


def load_shapley_pct_denominator_min(path: str | Path | None = None) -> float:
    """The Shapley percentage-denominator guard (§5.3, D-A20): shares are emitted
    only when |Delta_correction| exceeds this pre-registered minimum; otherwise
    the report emits null, not a number."""
    block = _auditor_block(path)
    value = _require(block, ("shapley", "percentage_denominator_min"), "§5.3")
    return _require_number(
        value, "auditor.shapley.percentage_denominator_min", "§5.3"
    )


def load_vartheta(path: str | Path | None = None) -> float:
    """ϑ — the practical-significance threshold. Load-bearing TWICE: it classifies
    economic significance (§9) AND defines material-effect prevalence (§8.2.3). Must
    be pre-registered with a cited source (O-A6)."""
    block = _auditor_block(path)
    value = _require(block, ("practical_significance", "vartheta"), "§9 / §8.2.3")
    return _require_number(value, "auditor.practical_significance.vartheta", "§9 / §8.2.3")


def load_compression_dmax(path: str | Path | None = None) -> float:
    """D_max — the compression-adequacy materiality threshold (§8.1, O-A10)."""
    block = _auditor_block(path)
    value = _require(block, ("compression", "d_max"), "§8.1")
    return _require_number(value, "auditor.compression.d_max", "§8.1")


def load_fdr_q(path: str | Path | None = None) -> float:
    """The BH-FDR level q over the confirmatory family (§7.2)."""
    block = _auditor_block(path)
    value = _require(block, ("fdr", "q"), "§7.2")
    return _require_number(value, "auditor.fdr.q", "§7.2")


@dataclass(frozen=True)
class BayesParams:
    """Bayesian normal-approximation constants (§7.3-7.4): the weakly-informative
    prior scale and the eigenvalue floor for V̂_boot regularisation."""

    prior_scale: float
    epsilon: float


def load_bayes_params(path: str | Path | None = None) -> BayesParams:
    block = _auditor_block(path)
    ps = _require(block, ("bayes", "prior_scale"), "§7.4")
    eps = _require(block, ("bayes", "epsilon"), "§7.3.3")
    prior_scale = _require_number(ps, "auditor.bayes.prior_scale", "§7.4")
    epsilon = _require_number(eps, "auditor.bayes.epsilon", "§7.3.3")
    # ε must be strictly positive: the eigenvalue floor guarantees a positive-definite
    # covariance only when ε > 0 (a zero floor would leave a singular V̂). prior_scale
    # likewise must be positive (it enters as 1/prior_scale²).
    if epsilon <= 0:
        raise AuditorThresholdError(
            "auditor.bayes.epsilon (must be strictly positive; the eigenvalue floor "
            f"needs ε > 0, got {epsilon})", "§7.3.3"
        )
    if prior_scale <= 0:
        raise AuditorThresholdError(
            f"auditor.bayes.prior_scale (must be strictly positive; got {prior_scale})",
            "§7.4",
        )
    return BayesParams(prior_scale=prior_scale, epsilon=epsilon)


@dataclass(frozen=True)
class EconomicGapBands:
    small: float
    moderate: float
    large: float


def load_economic_gap_bands(path: str | Path | None = None) -> EconomicGapBands:
    """Practical-significance bands for the endpoint gap (§9), each with a source."""
    block = _auditor_block(path)
    node = _require(block, ("economic", "gap_bands"), "§9")
    if not isinstance(node, dict):
        raise AuditorThresholdError("auditor.economic.gap_bands (not a mapping)", "§9")
    return EconomicGapBands(
        small=_require_number(node.get("small"), "auditor.economic.gap_bands.small", "§9"),
        moderate=_require_number(node.get("moderate"), "auditor.economic.gap_bands.moderate", "§9"),
        large=_require_number(node.get("large"), "auditor.economic.gap_bands.large", "§9"),
    )


@dataclass(frozen=True)
class ExplainerModelConfig:
    """The pre-registered LLM explainer model for one phase (§11). `dev` uses the
    free `phase_d` model (non-reportable prose); `reported` uses `phase_f`."""

    phase: str                       # "dev" | "reported"
    vendor: str
    model_id: str
    api_key_env: str
    temperature: float
    max_output_tokens: int
    max_retries: int
    min_interval_s: float


def load_explainer_config(
    phase: str, path: str | Path | None = None
) -> ExplainerModelConfig:
    """Read the step-19 explainer model for `phase` ∈ {'dev','reported'} fail-loud
    from `auditor.explainer` (§11)."""
    if phase not in ("dev", "reported"):
        raise ValueError(f"phase must be 'dev' or 'reported'; got {phase!r}")
    phase_key = "phase_d" if phase == "dev" else "phase_f"
    block = _auditor_block(path)
    temperature = _require_number(
        _require(block, ("explainer", "temperature"), "§11"),
        "auditor.explainer.temperature", "§11",
    )
    max_tokens = _require_int(
        _require(block, ("explainer", "max_output_tokens"), "§11"),
        "auditor.explainer.max_output_tokens", "§11",
    )
    max_retries = _require_int(
        _require(block, ("explainer", "max_retries"), "§11"),
        "auditor.explainer.max_retries", "§11",
    )
    pspec = _require(block, ("explainer", phase_key), "§11")
    if not isinstance(pspec, dict):
        raise AuditorThresholdError(f"auditor.explainer.{phase_key} (not a mapping)", "§11")

    def _str(key: str) -> str:
        value = pspec.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AuditorThresholdError(
                f"auditor.explainer.{phase_key}.{key} (missing or not a non-empty string)", "§11"
            )
        return value

    min_interval = pspec.get("min_interval_s", 0.0)
    return ExplainerModelConfig(
        phase=phase,
        vendor=_str("vendor"),
        model_id=_str("model_id"),
        api_key_env=_str("api_key_env"),
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=max_retries,
        min_interval_s=_require_number(
            min_interval, f"auditor.explainer.{phase_key}.min_interval_s", "§11"
        ),
    )
