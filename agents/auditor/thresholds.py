"""
thresholds.py — typed, FAIL-LOUD loader for the Auditor's pre-registration
constants (design §13.1; items O-A10 / O-A6).

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

import math
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


def load_inert_relative_tol(path: str | Path | None = None) -> float:
    """The inert-coordinate relative tolerance (§7.1) — registered, never defaulted: it decides
    whether a DOE coordinate is tested at all."""
    block = _auditor_block(path)
    value = _require_number(_require(block, ("inert_relative_tol",), "§7.1"),
                            "auditor.inert_relative_tol", "§7.1")
    if not math.isfinite(value) or value <= 0:
        # A non-finite tolerance would mark EVERY coordinate inert (p = 1, t = 0) and silently
        # suppress the tests this constant exists to route; a non-positive one is not a tolerance.
        raise AuditorThresholdError(
            f"auditor.inert_relative_tol (present but not a finite positive number: {value!r})", "§7.1"
        )
    return value


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


def load_vartheta_grid(path: str | Path | None = None) -> tuple[float, ...]:
    """The §8.2.3/§9 neighbouring-threshold SENSITIVITY grid. Every entry must be a
    positive number and the frozen headline vartheta MUST be one of them, so the sweep
    always brackets — and can never silently drift from — the primary cutoff."""
    dotted = "auditor.practical_significance.vartheta_sensitivity_grid"
    block = _auditor_block(path)
    raw = _require(block, ("practical_significance", "vartheta_sensitivity_grid"), "§8.2.3 / §9")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise AuditorThresholdError(f"{dotted} (present but not a non-empty list: {raw!r})",
                                    "§8.2.3 / §9")
    grid = tuple(
        _require_number(v, f"{dotted}[{i}]", "§8.2.3 / §9") for i, v in enumerate(raw)
    )
    if any(v <= 0 for v in grid):
        raise AuditorThresholdError(f"{dotted} (every threshold must be positive: {grid!r})",
                                    "§8.2.3 / §9")
    headline = load_vartheta(path)
    if not any(abs(v - headline) <= 1e-15 for v in grid):
        raise AuditorThresholdError(
            f"{dotted} (must contain the frozen headline vartheta={headline!r})", "§8.2.3 / §9"
        )
    return grid


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
class MtFlagParams:
    """Check-5 constants (§7.5, additive 2026-09-03): the just-significant |t|
    band (inclusive bounds), the free-parameter ceiling, and the D5-resolved zoo
    label file. Informational only — these gate nothing (D7/A7)."""

    t_band: tuple[float, float]
    max_free_parameters: int
    zoo_names_path: str
    strip_trailing_asterisk: bool
    aliases: dict[str, str]


def load_mt_flag_params(path: str | Path | None = None) -> MtFlagParams:
    block = _auditor_block(path)
    band = _require(block, ("mt_flag", "t_band"), "§7.5")
    if not (isinstance(band, (list, tuple)) and len(band) == 2):
        raise AuditorThresholdError(
            f"auditor.mt_flag.t_band must be a [lo, hi] pair (§7.5); got {band!r}")
    lo = _require_number(band[0], "auditor.mt_flag.t_band[0]", "§7.5")
    hi = _require_number(band[1], "auditor.mt_flag.t_band[1]", "§7.5")
    max_fp = _require(block, ("mt_flag", "max_free_parameters"), "§7.5")
    if not isinstance(max_fp, int) or isinstance(max_fp, bool) or max_fp < 0:
        raise AuditorThresholdError(
            f"auditor.mt_flag.max_free_parameters must be an int >= 0 (§7.5); got {max_fp!r}")
    zoo = _require(block, ("mt_flag", "zoo_names"), "§7.5")
    if not isinstance(zoo, str) or not zoo:
        raise AuditorThresholdError(
            f"auditor.mt_flag.zoo_names must be a repo-relative path (§7.5/D5); got {zoo!r}")
    strip = _require(block, ("mt_flag", "strip_trailing_asterisk"), "§7.5")
    if not isinstance(strip, bool):
        raise AuditorThresholdError(
            f"auditor.mt_flag.strip_trailing_asterisk must be a bool; got {strip!r}")
    aliases = _require(block, ("mt_flag", "aliases"), "§7.5")
    if not isinstance(aliases, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in aliases.items()):
        raise AuditorThresholdError(
            f"auditor.mt_flag.aliases must be a str->str map; got {aliases!r}")
    return MtFlagParams(t_band=(lo, hi), max_free_parameters=max_fp, zoo_names_path=zoo,
                        strip_trailing_asterisk=strip,
                        aliases={k.casefold(): v for k, v in aliases.items()})


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
class IPCABootstrapConfig:
    """Conditional moving-block bootstrap parameters (§5.3). Fixed blocks, ℓ = max(H_max,
    block_length_months); ONE common block sequence across all four cells; conditional on the
    realised fitted states (re-fitting uncertainty not estimated)."""

    n_replicates: int
    block_length_months: int
    min_effective_blocks: int
    holding_period_default: int
    alpha: float
    conditioning_label: str


def load_ipca_bootstrap_config(path: str | Path | None = None) -> IPCABootstrapConfig:
    """Read the §5.3 conditional bootstrap constants. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "bootstrap", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "bootstrap", *keys))

    return IPCABootstrapConfig(
        n_replicates=_require_int(req("n_replicates"), dk("n_replicates"), _IPCA_SECTION),
        block_length_months=_require_int(req("block_length_months"), dk("block_length_months"), _IPCA_SECTION),
        min_effective_blocks=_require_int(req("min_effective_blocks"), dk("min_effective_blocks"), _IPCA_SECTION),
        holding_period_default=_require_int(req("holding_period_default"), dk("holding_period_default"), _IPCA_SECTION),
        alpha=_require_number(req("alpha"), dk("alpha"), _IPCA_SECTION),
        conditioning_label=_require_str(req("conditioning_label"), dk("conditioning_label"), _IPCA_SECTION),
    )


@dataclass(frozen=True)
class IPCAStabilityConfig:
    """Stability-diagnostic parameters (§5.4). R' outer refits on blocked panel resamples, each
    under the production rule §5.2; reports SIGN and ORDER-OF-MAGNITUDE survival of I. Never an
    interval; pre-registered so it cannot become a post-hoc rescue."""

    r_prime: int
    block_length_months: int
    min_effective_blocks: int
    holding_period_default: int
    order_of_magnitude_factor: float
    scope: tuple[str, ...]


def load_ipca_stability_config(path: str | Path | None = None) -> IPCAStabilityConfig:
    """Read the §5.4 stability-diagnostic constants. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "stability_diagnostic", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "stability_diagnostic", *keys))

    return IPCAStabilityConfig(
        r_prime=_require_int(req("r_prime"), dk("r_prime"), _IPCA_SECTION),
        block_length_months=_require_int(req("block_length_months"), dk("block_length_months"), _IPCA_SECTION),
        min_effective_blocks=_require_int(req("min_effective_blocks"), dk("min_effective_blocks"), _IPCA_SECTION),
        holding_period_default=_require_int(req("holding_period_default"), dk("holding_period_default"), _IPCA_SECTION),
        order_of_magnitude_factor=_require_number(
            req("order_of_magnitude_factor"), dk("order_of_magnitude_factor"), _IPCA_SECTION
        ),
        scope=_require_str_list(req("scope"), dk("scope"), _IPCA_SECTION),
    )


@dataclass(frozen=True)
class IPCATwinDGP:
    """Matched-twin DGP parameters (§6.2). Synthetic-scale: each bond has two i.i.d. idiosyncratic
    draws sharing one characteristic path and loadings — exchangeable by construction."""

    n_bonds: int
    n_months: int
    noise_sd: float


@dataclass(frozen=True)
class IPCAFprConfig:
    """Randomisation-based FPR parameters (§6.2). The mean-zero placebo is withdrawn; the instrument
    is exchangeability (label re-randomisation under the production rule §5.2)."""

    q_permutations: int              # Q
    r_datasets: int                  # R
    alpha_nominal: float             # α_nom
    acceptance_band_level: float     # binomial acceptance band coverage
    twin_dgp: IPCATwinDGP
    reduced_q: int
    reduced_r: int
    rename_fallback_label: str


def load_ipca_fpr_config(path: str | Path | None = None) -> IPCAFprConfig:
    """Read the §6.2 randomisation-FPR constants. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "randomisation_fpr", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "randomisation_fpr", *keys))

    twin = req("twin_dgp")
    if not isinstance(twin, dict):
        raise AuditorThresholdError(dk("twin_dgp") + " (not a mapping)", _IPCA_SECTION)
    twin_dgp = IPCATwinDGP(
        n_bonds=_require_int(twin.get("n_bonds"), dk("twin_dgp", "n_bonds"), _IPCA_SECTION),
        n_months=_require_int(twin.get("n_months"), dk("twin_dgp", "n_months"), _IPCA_SECTION),
        noise_sd=_require_number(twin.get("noise_sd"), dk("twin_dgp", "noise_sd"), _IPCA_SECTION),
    )
    return IPCAFprConfig(
        q_permutations=_require_int(req("q_permutations"), dk("q_permutations"), _IPCA_SECTION),
        r_datasets=_require_int(req("r_datasets"), dk("r_datasets"), _IPCA_SECTION),
        alpha_nominal=_require_number(req("alpha_nominal"), dk("alpha_nominal"), _IPCA_SECTION),
        acceptance_band_level=_require_number(
            req("acceptance_band_level"), dk("acceptance_band_level"), _IPCA_SECTION
        ),
        twin_dgp=twin_dgp,
        reduced_q=_require_int(req("reduced_q"), dk("reduced_q"), _IPCA_SECTION),
        reduced_r=_require_int(req("reduced_r"), dk("reduced_r"), _IPCA_SECTION),
        rename_fallback_label=_require_str(
            req("rename_fallback_label"), dk("rename_fallback_label"), _IPCA_SECTION
        ),
    )


@dataclass(frozen=True)
class IPCAPerturbationConfig:
    """Stochastic perturbation robustness parameters (§6.3). The mean-zero-noise run under its honest
    name — a robustness descriptive, never an FPR."""

    n_draws: int
    noise_sd: float


def load_ipca_perturbation_config(path: str | Path | None = None) -> IPCAPerturbationConfig:
    """Read the §6.3 perturbation-robustness constants. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "perturbation_robustness", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "perturbation_robustness", *keys))

    return IPCAPerturbationConfig(
        n_draws=_require_int(req("n_draws"), dk("n_draws"), _IPCA_SECTION),
        noise_sd=_require_number(req("noise_sd"), dk("noise_sd"), _IPCA_SECTION),
    )


@dataclass(frozen=True)
class IPCARefusedPairs:
    """The typed-refused (bias, anchor) pairs (§12.7, D-A50 amendment). The construction-layer
    toggles have no published IPCA implementation, so their OFF state is UNDEFINED — refused, not
    defaulted."""

    biases: tuple[str, ...]
    anchors: tuple[str, ...]
    reason: str
    note: str


@dataclass(frozen=True)
class IPCASignalPropagation:
    """The signal-propagation branch rule (decision b). ``primary`` recomputes signals per panel
    state; the ``fallback`` (frozen-at-P_N, renamed partial-channel) fires only if the spike says the
    rebuild is infeasible — decided by measurement, not preference."""

    primary: str
    spike_timebox_hours: int
    fallback: str


@dataclass(frozen=True)
class IPCAExecutionPairs:
    """The amended execution pair set (§12.7): 9 runnable + 6 typed-refused, frozen pre-result."""

    runnable_biases: tuple[str, ...]
    runnable_anchors: tuple[str, ...]
    refused: IPCARefusedPairs
    smoke_pair: tuple[str, str]
    signal_propagation: IPCASignalPropagation
    smoke_is_engineering_only: bool

    def runnable_pairs(self) -> list[tuple[str, str]]:
        return [(b, a) for b in self.runnable_biases for a in self.runnable_anchors]

    def refused_pairs(self) -> list[tuple[str, str]]:
        return [(b, a) for b in self.refused.biases for a in self.refused.anchors]


def load_ipca_execution_pairs(path: str | Path | None = None) -> IPCAExecutionPairs:
    """Read the §12.7 amended pair set + signal-propagation branch rule. Raises if unregistered."""
    block = _ipca_block(path)

    def req(*keys: str) -> object:
        return _require(block, ("ipca_differential", "execution", *keys), _IPCA_SECTION)

    def dk(*keys: str) -> str:
        return ".".join(("auditor", "ipca_differential", "execution", *keys))

    ref = req("refused")
    if not isinstance(ref, dict):
        raise AuditorThresholdError(dk("refused") + " (not a mapping)", _IPCA_SECTION)
    refused = IPCARefusedPairs(
        biases=_require_str_list(ref.get("biases"), dk("refused", "biases"), _IPCA_SECTION),
        anchors=_require_str_list(ref.get("anchors"), dk("refused", "anchors"), _IPCA_SECTION),
        reason=_require_str(ref.get("reason"), dk("refused", "reason"), _IPCA_SECTION),
        note=_require_str(ref.get("note"), dk("refused", "note"), _IPCA_SECTION),
    )
    sig = req("signal_propagation")
    if not isinstance(sig, dict):
        raise AuditorThresholdError(dk("signal_propagation") + " (not a mapping)", _IPCA_SECTION)
    signal_propagation = IPCASignalPropagation(
        primary=_require_str(sig.get("primary"), dk("signal_propagation", "primary"), _IPCA_SECTION),
        spike_timebox_hours=_require_int(
            sig.get("spike_timebox_hours"), dk("signal_propagation", "spike_timebox_hours"), _IPCA_SECTION
        ),
        fallback=_require_str(sig.get("fallback"), dk("signal_propagation", "fallback"), _IPCA_SECTION),
    )
    smoke = _require_str_list(req("smoke_pair"), dk("smoke_pair"), _IPCA_SECTION)
    if len(smoke) != 2:
        raise AuditorThresholdError(dk("smoke_pair") + " (must be a [bias, anchor] pair)", _IPCA_SECTION)
    return IPCAExecutionPairs(
        runnable_biases=_require_str_list(req("runnable_biases"), dk("runnable_biases"), _IPCA_SECTION),
        runnable_anchors=_require_str_list(req("runnable_anchors"), dk("runnable_anchors"), _IPCA_SECTION),
        refused=refused,
        smoke_pair=(smoke[0], smoke[1]),
        signal_propagation=signal_propagation,
        smoke_is_engineering_only=_require_bool(
            req("smoke_is_engineering_only"), dk("smoke_is_engineering_only"), _IPCA_SECTION
        ),
    )


@dataclass(frozen=True)
class IPCAExecutionConfig:
    """The full pre-registration bundle required before any EXTENSION RUN (§9, §12.7).
    Only obtainable once `status == 'preregistered_complete'` and every block the run consumes
    is present AND schema-valid."""

    lam: IPCALambda
    projection_gate: IPCAProjectionGate
    reporting: IPCAReporting
    bootstrap: IPCABootstrapConfig
    stability: IPCAStabilityConfig
    fpr: IPCAFprConfig
    perturbation: IPCAPerturbationConfig
    pairs: IPCAExecutionPairs


def load_ipca_execution_config(path: str | Path | None = None) -> IPCAExecutionConfig:
    """Gate for any reportable / real-data extension run (§9 requires ALL constants
    registered first; §12.7 is the run). Raises `AuditorThresholdError` unless the block's
    `status` is `preregistered_complete` AND every block the run consumes actually LOADS
    (schema-valid, not merely key-present). In the development contract this ALWAYS raises — by
    design — so a partial or malformed pre-registration can never launch a run."""
    block = _ipca_block(path)
    status = _require_str(
        _require(block, ("ipca_differential", "status"), _IPCA_SECTION),
        "auditor.ipca_differential.status", _IPCA_SECTION,
    )
    if status != "preregistered_complete":
        raise AuditorThresholdError(
            f"auditor.ipca_differential.status is {status!r}, not 'preregistered_complete' "
            "— the IPCA differential is a development contract only. Flip the status (and git-tag "
            "the pre-registration) only once every §9 block is registered; no reportable or "
            "real-data extension run may proceed until then (execution checklist §12.7)",
            _IPCA_SECTION,
        )
    # status claims completeness: every block the run consumes must actually LOAD (a bare key-
    # existence check would let a malformed FPR/stability block certify a run and fail only at
    # run time — the pre-registration gate must validate the actual constants).
    return IPCAExecutionConfig(
        lam=load_ipca_lambda(path),
        projection_gate=load_ipca_projection_gate(path),
        reporting=load_ipca_reporting(path),
        bootstrap=load_ipca_bootstrap_config(path),
        stability=load_ipca_stability_config(path),
        fpr=load_ipca_fpr_config(path),
        perturbation=load_ipca_perturbation_config(path),
        pairs=load_ipca_execution_pairs(path),
    )
