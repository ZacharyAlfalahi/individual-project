"""
inventory.py — the four-category frozen-state inventory (spec §4.1, D-A49).

Single source of truth. Every object in the IPCA fitted-state + evaluation map is
classified into exactly ONE of the four categories:

  1. Fitted state, frozen in Θ̂        — chosen by optimising the ALS fit criterion.
  2. Deterministic per-cell recompute  — closed-form point-in-time functionals of the
                                         EVALUATION panel (no fit criterion anywhere).
  3. Deliberately excluded             — not stored, not consumed.
  4. Globally registered constants λ   — fixed ex ante, identical across every panel,
                                         fit, cell, placebo and diagnostic replicate.

"The same parameter can live in different categories with different provenance
implications" (§4.1) — classification is by HOW an object was set, not by its name.
"'Deterministic' does not mean 'admissible'": category 2 admits only functionals that
respect the registered point-in-time rule (no full-sample moment may leak in).

The manifest in ``frozen_state.py`` validates its field lists against
``CAT1_FITTED_FIELDS`` and ``CAT3_EXCLUDED_FIELDS`` here, making the frozen-state
contract machine-checkable. ``docs/auditor/ipca_frozen_state_inventory.md`` is the
human-readable mirror of this module — the two must agree.

Grounded in ``agents/quant/library/ipca.py`` (the ``IPCAFit`` NamedTuple) under the
restricted α=0 specification (``fit_ipca(..., alpha=False)`` ⇒ ``gamma_alpha=None``).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Category 1 — fitted state, frozen in Θ̂ (chosen by optimising ALS).
# ---------------------------------------------------------------------------
# IPCAFit.gamma_beta (L, K): the ALS loading map. IPCAFit.factors (K, T): the fitted
# in-sample factor path (also an ALS output). Both are part of the frozen artefact and
# are hashed for identity. NOTE the consumption boundary below: `factors` is provenance
# only — it is NEVER read during cell evaluation (the cell factor path is recomputed on
# the evaluation panel), so reusing it would collapse the data channel and void the
# bracket identity. gamma_alpha would be category 1 iff α were on; under the restricted
# α=0 spec it is None and lives in category 3.
CAT1_FITTED_FIELDS: tuple[str, ...] = ("gamma_beta", "factors")

# ---------------------------------------------------------------------------
# Category 2 — deterministic per-cell recomputation from the EVALUATION panel.
# ---------------------------------------------------------------------------
# Closed-form, parameter-free, point-in-time. In ipca.py / ipca_feed.py these are:
#   rank_transform (per-month cross-sectional KPP rank map, (rank-1)/(rmax-1)-0.5),
#   the constant-column append (constant ≡ 1, last), VOL scaling R=xret/max(vol,floor),
#   universe + complete-case filtering, and the sufficient statistics W=Z'Z/N, x=Z'R/N.
# A full-sample mean or std would be closed-form AND leak the future — forbidden here.
CAT2_RECOMPUTED: tuple[str, ...] = (
    "rank_transform_per_month",
    "constant_column_append",
    "vol_scaling",
    "universe_complete_case_filter",
    "sufficient_stats_W",
    "sufficient_stats_x",
)

# ---------------------------------------------------------------------------
# Category 3 — deliberately excluded (not stored, not consumed).
# ---------------------------------------------------------------------------
# gamma_alpha: the restricted α=0 spec — the test-asset alpha is measured OUTSIDE the
# model (a regression of the fixed anchor on the recovered factors); an internal
# characteristic-aligned intercept would absorb the very mispricing whose transmission is
# the estimand. factor_risk_premium: unneeded — the estimand regresses on realised f_t.
CAT3_EXCLUDED_FIELDS: tuple[str, ...] = ("gamma_alpha", "factor_risk_premium")

# ---------------------------------------------------------------------------
# Category 4 — globally registered constants λ (thresholds.yaml, fixed ex ante).
# ---------------------------------------------------------------------------
CAT4_LAMBDA: tuple[str, ...] = (
    "factor_count",             # K
    "instrument_count",         # L (= len(characteristic_order) + 1)
    "characteristic_order",     # the L-1 instrument names, in order
    "als_tolerance",
    "als_max_iter",
    "month_weighting",          # the K22 arm
    "scaling_lane",             # VOLScaled010
    "vol_floor",
    "normalisation_rule",       # the deterministic KPP R2 identification map
    "n_initialisations",        # M (production rule §5.2)
    "initialisation_seeds",
    "tie_break",
    "pseudoinverse_permitted",  # projection gate §6.4
    "pseudoinverse_tolerance",
)

# ---------------------------------------------------------------------------
# The cell-evaluation consumption boundary (§4.2, D-A46/D-A49).
# ---------------------------------------------------------------------------
# Evaluation reads ONLY these frozen components; the factor path is recomputed from the
# evaluation panel via ipca._oos_factor_realization. `factors`/`months` are stored and
# hashed as fitted-state provenance but MUST NOT be consumed during evaluation.
PROJECTION_CONSUME: tuple[str, ...] = ("gamma_beta", "gamma_alpha")
PROVENANCE_ONLY: tuple[str, ...] = ("factors", "months")

# The deterministic identification map inside ipca._identify (R2 adjudication): sign-
# stabilised QR → order by descending factor second moment → eigenvector sign pinned by
# largest-abs-component-positive → KPP mean≥0 arbiter. A reproducibility requirement, not
# identification of the alpha estimand (which is rotation-invariant).
NORMALISATION_RULE: str = "kpp_r2_qr_descending_moment_meanpos"


@dataclass(frozen=True)
class CategoryInventory:
    """Immutable view of the four-category inventory. Consumed by the manifest validator."""

    cat1_fitted: tuple[str, ...]
    cat2_recomputed: tuple[str, ...]
    cat3_excluded: tuple[str, ...]
    cat4_lambda: tuple[str, ...]

    def category_of(self, name: str) -> int | None:
        """Return the category (1-4) a name belongs to, or None if unclassified."""
        for cat, members in (
            (1, self.cat1_fitted),
            (2, self.cat2_recomputed),
            (3, self.cat3_excluded),
            (4, self.cat4_lambda),
        ):
            if name in members:
                return cat
        return None


INVENTORY = CategoryInventory(
    cat1_fitted=CAT1_FITTED_FIELDS,
    cat2_recomputed=CAT2_RECOMPUTED,
    cat3_excluded=CAT3_EXCLUDED_FIELDS,
    cat4_lambda=CAT4_LAMBDA,
)
