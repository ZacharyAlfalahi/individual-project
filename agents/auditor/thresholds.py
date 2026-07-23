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


def _require_str(value: object, dotted_key: str, section: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AuditorThresholdError(
            f"{dotted_key} (present but not a non-empty string: {value!r})", section
        )
    return value


def _require_bool(value: object, dotted_key: str, section: str) -> bool:
    if not isinstance(value, bool):
        raise AuditorThresholdError(
            f"{dotted_key} (present but not a bool: {value!r})", section
        )
    return value


def _require_int_list(value: object, dotted_key: str, section: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise AuditorThresholdError(
            f"{dotted_key} (present but not a non-empty list: {value!r})", section
        )
    return tuple(_require_int(v, f"{dotted_key}[{i}]", section) for i, v in enumerate(value))


def _require_str_list(value: object, dotted_key: str, section: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise AuditorThresholdError(
            f"{dotted_key} (present but not a non-empty list: {value!r})", section
        )
    return tuple(_require_str(v, f"{dotted_key}[{i}]", section) for i, v in enumerate(value))


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


# ---------------------------------------------------------------------------
# Frozen-Loadings IPCA Differential (RQ3 extension) — spec §9.
#
# The extension is a pre-registered exploratory stretch. Its `status` field gates
# execution: `development_contract_only` (this build) registers only the differential and inference
# constants; `load_ipca_execution_config` refuses any reportable / real-data run until
# `status == 'preregistered_complete'` AND the deferred bootstrap/FPR/stability blocks
# exist — so a partial pre-registration can never masquerade as final (§9 requires ALL
# constants registered before any extension run).
# ---------------------------------------------------------------------------

_IPCA_SECTION = "§9 (IPCA differential v5)"


def _ipca_block(path: str | Path | None) -> dict:
    """The `auditor.ipca_differential` sub-block; raise loudly if absent."""
    block = _auditor_block(path)
    node = block.get("ipca_differential")
    if not isinstance(node, dict):
        raise AuditorThresholdError("auditor.ipca_differential", _IPCA_SECTION)
    return block  # return the auditor block so _require builds full dotted paths


@dataclass(frozen=True)
class IPCALambda:
    """Category-4 registered constants λ (spec §4.1 / §9.5). Fixed ex ante, identical
    across every panel, fit, cell, placebo and diagnostic replicate."""

    factor_count: int                       # K
    k_sensitivity: tuple[int, ...]          # K±1 sensitivity set
    instrument_count: int                   # L = len(characteristic_order) + 1
    characteristic_order: tuple[str, ...]
    als_tolerance: float
    als_max_iter: int
    month_weighting: str
    scaling_lane: str
    vol_floor: float
    normalisation_rule: str
    n_initialisations: int                  # M (production rule §5.2)
    initialisation_seeds: tuple[int, ...]
    tie_break: str

    def __post_init__(self) -> None:
        if self.instrument_count != len(self.characteristic_order) + 1:
            raise AuditorThresholdError(
                f"auditor.ipca_differential.lambda.instrument_count "
                f"({self.instrument_count}) must equal len(characteristic_order)+1 "
                f"({len(self.characteristic_order) + 1}; +1 for the constant column)",
                _IPCA_SECTION,
            )


def load_ipca_lambda(path: str | Path | None = None) -> IPCALambda:
    """Read the λ registered constants (§4.1 / §9.5). Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "lambda", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "lambda", *keys))

    return IPCALambda(
        factor_count=_require_int(req("factor_count"), dk("factor_count"), _IPCA_SECTION),
        k_sensitivity=_require_int_list(req("k_sensitivity"), dk("k_sensitivity"), _IPCA_SECTION),
        instrument_count=_require_int(req("instrument_count"), dk("instrument_count"), _IPCA_SECTION),
        characteristic_order=_require_str_list(
            req("characteristic_order"), dk("characteristic_order"), _IPCA_SECTION
        ),
        als_tolerance=_require_number(req("als_tolerance"), dk("als_tolerance"), _IPCA_SECTION),
        als_max_iter=_require_int(req("als_max_iter"), dk("als_max_iter"), _IPCA_SECTION),
        month_weighting=_require_str(req("month_weighting"), dk("month_weighting"), _IPCA_SECTION),
        scaling_lane=_require_str(req("scaling_lane"), dk("scaling_lane"), _IPCA_SECTION),
        vol_floor=_require_number(req("vol_floor"), dk("vol_floor"), _IPCA_SECTION),
        normalisation_rule=_require_str(
            req("normalisation_rule"), dk("normalisation_rule"), _IPCA_SECTION
        ),
        n_initialisations=_require_int(
            req("n_initialisations"), dk("n_initialisations"), _IPCA_SECTION
        ),
        initialisation_seeds=_require_int_list(
            req("initialisation_seeds"), dk("initialisation_seeds"), _IPCA_SECTION
        ),
        tie_break=_require_str(req("tie_break"), dk("tie_break"), _IPCA_SECTION),
    )


@dataclass(frozen=True)
class IPCAProjectionGate:
    """Projection identifiability gate (§6.4). Gated on B_t = Z_t Γ directly."""

    min_cross_section_n: int
    require_rank: int                       # = K
    max_condition_number: float
    pseudoinverse_permitted: bool
    pseudoinverse_tolerance: float | None
    max_failed_month_fraction: float
    failed_month_handling: str              # "exclude" | "cell_refusal"

    def __post_init__(self) -> None:
        if self.failed_month_handling not in ("exclude", "cell_refusal"):
            raise AuditorThresholdError(
                "auditor.ipca_differential.projection_gate.failed_month_handling "
                f"(must be 'exclude' or 'cell_refusal'; got {self.failed_month_handling!r})",
                _IPCA_SECTION,
            )
        if self.pseudoinverse_permitted and self.pseudoinverse_tolerance is None:
            raise AuditorThresholdError(
                "auditor.ipca_differential.projection_gate.pseudoinverse_tolerance "
                "(must be a number when pseudoinverse_permitted is true)",
                _IPCA_SECTION,
            )


def load_ipca_projection_gate(path: str | Path | None = None) -> IPCAProjectionGate:
    """Read the §6.4 projection gate constants. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "projection_gate", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "projection_gate", *keys))

    tol_raw = req("pseudoinverse_tolerance")
    tol = None if tol_raw is None else _require_number(
        tol_raw, dk("pseudoinverse_tolerance"), _IPCA_SECTION
    )
    return IPCAProjectionGate(
        min_cross_section_n=_require_int(
            req("min_cross_section_n"), dk("min_cross_section_n"), _IPCA_SECTION
        ),
        require_rank=_require_int(req("require_rank"), dk("require_rank"), _IPCA_SECTION),
        max_condition_number=_require_number(
            req("max_condition_number"), dk("max_condition_number"), _IPCA_SECTION
        ),
        pseudoinverse_permitted=_require_bool(
            req("pseudoinverse_permitted"), dk("pseudoinverse_permitted"), _IPCA_SECTION
        ),
        pseudoinverse_tolerance=tol,
        max_failed_month_fraction=_require_number(
            req("max_failed_month_fraction"), dk("max_failed_month_fraction"), _IPCA_SECTION
        ),
        failed_month_handling=_require_str(
            req("failed_month_handling"), dk("failed_month_handling"), _IPCA_SECTION
        ),
    )


@dataclass(frozen=True)
class IPCAReporting:
    """The frozen reporting set + emphasis rules (§3.2). No aggregate, no GRS."""

    n_pairs: int
    anchors: tuple[str, ...]
    focal_pairs: dict[str, str]             # bias -> anchor, only where a mechanism is registered
    no_focal_pair: tuple[str, ...]
    benchmark_overlay_exemption: bool

    def __post_init__(self) -> None:
        expected = self.n_pairs
        got = len(set(self.focal_pairs) | set(self.no_focal_pair)) * len(self.anchors)
        if got != expected:
            raise AuditorThresholdError(
                f"auditor.ipca_differential.reporting.n_pairs ({expected}) must equal "
                f"len(biases) × len(anchors) ({got}); biases = focal_pairs ∪ no_focal_pair",
                _IPCA_SECTION,
            )


def load_ipca_reporting(path: str | Path | None = None) -> IPCAReporting:
    """Read the §3.2 reporting set. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "reporting", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "reporting", *keys))

    focal_raw = req("focal_pairs")
    if not isinstance(focal_raw, dict) or not focal_raw:
        raise AuditorThresholdError(dk("focal_pairs") + " (not a non-empty mapping)", _IPCA_SECTION)
    focal = {
        _require_str(k, dk("focal_pairs", "<key>"), _IPCA_SECTION):
        _require_str(v, dk("focal_pairs", str(k)), _IPCA_SECTION)
        for k, v in focal_raw.items()
    }
    return IPCAReporting(
        n_pairs=_require_int(req("n_pairs"), dk("n_pairs"), _IPCA_SECTION),
        anchors=_require_str_list(req("anchors"), dk("anchors"), _IPCA_SECTION),
        focal_pairs=focal,
        no_focal_pair=_require_str_list(req("no_focal_pair"), dk("no_focal_pair"), _IPCA_SECTION),
        benchmark_overlay_exemption=_require_bool(
            req("benchmark_overlay_exemption"), dk("benchmark_overlay_exemption"), _IPCA_SECTION
        ),
    )


@dataclass(frozen=True)
class IPCAExecutionConfig:
    """The full pre-registration bundle required before any EXTENSION RUN (§9, §12.7).
    Only obtainable once `status == 'preregistered_complete'` and the deferred
    bootstrap/FPR/stability blocks are registered."""

    lam: IPCALambda
    projection_gate: IPCAProjectionGate
    reporting: IPCAReporting


def load_ipca_execution_config(path: str | Path | None = None) -> IPCAExecutionConfig:
    """Gate for any reportable / real-data extension run (§9 requires ALL constants
    registered first; §12.7 is the run). Raises `AuditorThresholdError` unless the block's
    `status` is `preregistered_complete` AND the deferred bootstrap / randomisation-FPR /
    stability blocks are present. In the development contract this ALWAYS
    raises — by design — so a partial pre-registration can never launch a run."""
    block = _ipca_block(path)
    status = _require_str(
        _require(block, ("ipca_differential", "status"), _IPCA_SECTION),
        "auditor.ipca_differential.status", _IPCA_SECTION,
    )
    if status != "preregistered_complete":
        raise AuditorThresholdError(
            f"auditor.ipca_differential.status is {status!r}, not 'preregistered_complete' "
            "— the IPCA differential is a development contract only. The §9 bootstrap / "
            "randomisation-FPR / stability constants are not yet registered, so no reportable "
            "or real-data extension run may proceed (execution checklist §12.7)",
            _IPCA_SECTION,
        )
    # status claims completeness: the deferred blocks must then actually exist.
    for key in ("bootstrap", "randomisation_fpr", "stability_diagnostic"):
        _require(block, ("ipca_differential", key), _IPCA_SECTION)
    return IPCAExecutionConfig(
        lam=load_ipca_lambda(path),
        projection_gate=load_ipca_projection_gate(path),
        reporting=load_ipca_reporting(path),
    )
