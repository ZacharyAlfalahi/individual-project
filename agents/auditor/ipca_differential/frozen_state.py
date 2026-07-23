"""
frozen_state.py — the frozen-state contract (spec §4.2, D-A49).

``FrozenIPCAState`` wraps a complete IPCA fitted state (from ``ipca.fit_ipca``) plus the
registered constants λ into an immutable, content-hashed, manifest-validated artefact. Two
guarantees make "engine-cancels is restored" machine-checkable:

  * **Hash proves artefact identity** — a content hash over the fitted arrays + λ; every cell
    record carries it, so two cells can be shown to have run under the SAME frozen state.
  * **Manifest proves the correct object was frozen** — its field lists are validated against the
    four-category inventory (``inventory.py``): the category-1 fitted fields must be present, the
    category-3 excluded fields must be genuinely absent (restricted α=0 ⇒ ``gamma_alpha is None``),
    and the shapes must agree with λ.

**Consumption boundary (§4.2, the load-bearing correctness point).** ``factors`` and ``months`` are
stored and hashed as fitted-state *provenance only*. Cell evaluation must consume ONLY the frozen
projection components — ``gamma_beta`` and ``gamma_alpha`` — and recompute the factor path from the
EVALUATION panel. The narrow ``projection_params()`` accessor exposes exactly those two, so the
evaluation code cannot reach ``factors`` by construction; reusing the fitted path would collapse the
data channel and void the bracket identity.

The frozen arrays are made read-only (``writeable = False``) so the "frozen column" is frozen in
fact, not just by convention.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, NamedTuple

import numpy as np

from agents.auditor.thresholds import IPCALambda

from . import inventory as inv

SCHEMA_VERSION = "1.0"


class ProjectionParams(NamedTuple):
    """The ONLY frozen components cell evaluation may consume (§4.2). No factor path here —
    it is recomputed from the evaluation panel via ``ipca._oos_factor_realization``."""

    gamma_beta: np.ndarray            # (L, K)
    gamma_alpha: np.ndarray | None    # None under the restricted α=0 spec


class FrozenStateError(ValueError):
    """The frozen-state contract was violated (wrong object frozen, or hash mismatch)."""


@dataclass(frozen=True, eq=False)
class FrozenIPCAState:
    """A complete, immutable IPCA fitted state Θ̂ + λ metadata, content-hashed and
    manifest-validated. Construct via ``FrozenIPCAState.freeze(fit, lam)``."""

    gamma_beta: np.ndarray            # category 1 — fitted, consumed in projection
    gamma_alpha: np.ndarray | None    # category 3 — excluded under α=0 (must be None)
    factors: np.ndarray               # category 1 — fitted, PROVENANCE ONLY (never consumed)
    months: np.ndarray                # provenance
    factor_count: int                 # K (must equal λ.factor_count)
    instrument_count: int             # L
    characteristic_order: tuple[str, ...]
    normalisation_rule: str
    content_hash: str
    schema_version: str = SCHEMA_VERSION
    fit_meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Freeze the arrays in fact: an in-place mutation would silently break artefact identity.
        for name in ("gamma_beta", "factors", "months"):
            arr = getattr(self, name)
            arr.flags.writeable = False
        if self.gamma_alpha is not None:
            self.gamma_alpha.flags.writeable = False

    # ---- construction ------------------------------------------------------

    @classmethod
    def freeze(cls, fit, lam: IPCALambda) -> "FrozenIPCAState":
        """Freeze an ``ipca.IPCAFit`` under the registered constants λ. Copies the fitted arrays
        (so the source fit is untouched and the frozen copy is independent), enforces the
        restricted α=0 spec, checks K/L against λ, and computes the content hash."""
        gb = np.array(fit.gamma_beta, dtype=np.float64)          # copy (source untouched)
        ga = None if fit.gamma_alpha is None else np.array(fit.gamma_alpha, dtype=np.float64)
        f = np.array(fit.factors, dtype=np.float64)
        months = np.array(fit.months, dtype=np.int64)

        if ga is not None:
            raise FrozenStateError(
                "frozen state requires the restricted α=0 specification (gamma_alpha must be "
                "None — the test-asset alpha is measured OUTSIDE the model; see inventory category 3)"
            )
        k = int(gb.shape[1])
        n_inst = int(gb.shape[0])
        if k != lam.factor_count:
            raise FrozenStateError(
                f"fitted K={k} != λ.factor_count={lam.factor_count} (gamma_beta has {k} columns)"
            )
        if n_inst != lam.instrument_count:
            raise FrozenStateError(
                f"fitted L={n_inst} != λ.instrument_count={lam.instrument_count} "
                f"(gamma_beta has {n_inst} rows)"
            )

        content_hash = cls._compute_hash(gb, ga, f, months, lam)
        return cls(
            gamma_beta=gb,
            gamma_alpha=ga,
            factors=f,
            months=months,
            factor_count=k,
            instrument_count=n_inst,
            characteristic_order=tuple(lam.characteristic_order),
            normalisation_rule=lam.normalisation_rule,
            content_hash=content_hash,
            schema_version=SCHEMA_VERSION,
            fit_meta={
                "converged": bool(fit.converged),
                "n_iter": int(fit.n_iter),
                "tol_final": float(fit.tol_final),
                "init_kind": str(fit.init_kind),
                "weighting": str(fit.weighting),
            },
        )

    # ---- identity ----------------------------------------------------------

    @staticmethod
    def _compute_hash(
        gb: np.ndarray, ga: np.ndarray | None, f: np.ndarray, months: np.ndarray, lam: IPCALambda
    ) -> str:
        """SHA-256 over λ + the fitted arrays (canonical dtype + shape + bytes). Deterministic:
        the same fit + λ always produce the same hash."""
        h = hashlib.sha256()
        h.update(b"FrozenIPCAState/v" + SCHEMA_VERSION.encode() + b"\n")
        meta = (
            f"K={lam.factor_count};L={lam.instrument_count};"
            f"norm={lam.normalisation_rule};chars={','.join(lam.characteristic_order)}"
        )
        h.update(meta.encode("utf-8"))
        for name, arr, dt in (
            ("gamma_beta", gb, np.float64),
            ("gamma_alpha", ga, np.float64),
            ("factors", f, np.float64),
            ("months", months, np.int64),
        ):
            h.update(("\n" + name + ":").encode("utf-8"))
            if arr is None:
                h.update(b"None")
                continue
            a = np.ascontiguousarray(arr, dtype=dt)
            h.update(str(a.shape).encode("utf-8"))
            h.update(a.tobytes())
        return h.hexdigest()

    def verify_hash(self, lam: IPCALambda) -> bool:
        """Recompute the content hash from the stored arrays and λ; True iff it matches."""
        return self._compute_hash(
            self.gamma_beta, self.gamma_alpha, self.factors, self.months, lam
        ) == self.content_hash

    # ---- the consumption boundary -----------------------------------------

    def projection_params(self) -> ProjectionParams:
        """The ONLY frozen components evaluation may read (§4.2). Deliberately excludes
        ``factors`` — the cell factor path is recomputed from the evaluation panel."""
        return ProjectionParams(gamma_beta=self.gamma_beta, gamma_alpha=self.gamma_alpha)

    # ---- manifest ----------------------------------------------------------

    def manifest(self) -> dict:
        """The §4.2 semantic manifest. Field lists come from the four-category inventory, so the
        contract is machine-checkable against ``inventory.py``."""
        return {
            "content_hash": self.content_hash,
            "schema_version": self.schema_version,
            "factor_count": self.factor_count,
            "instrument_count": self.instrument_count,
            "characteristic_order": list(self.characteristic_order),
            "normalisation_rule": self.normalisation_rule,
            "fitted_fields": list(inv.CAT1_FITTED_FIELDS),
            "excluded_fields": list(inv.CAT3_EXCLUDED_FIELDS),
        }

    def validate_manifest(self, lam: IPCALambda, inventory: inv.CategoryInventory = inv.INVENTORY) -> None:
        """Assert the frozen object matches the category inventory and λ (spec §4.2). Raises
        ``FrozenStateError`` on any mismatch. Non-tautological: it checks the ACTUAL arrays, not
        just that copied field-lists agree with themselves.

        1. every category-1 fitted field is present and non-None;
        2. every category-3 excluded field is genuinely excluded (gamma_alpha is None; no
           factor_risk_premium attribute leaks in);
        3. K/L/characteristic_order/normalisation agree with λ and with gamma_beta's shape;
        4. the content hash recomputes.
        """
        # (1) category-1 fitted fields present & non-None
        for name in inventory.cat1_fitted:
            if getattr(self, name, None) is None:
                raise FrozenStateError(f"category-1 fitted field {name!r} is missing or None")

        # (2) category-3 excluded fields genuinely excluded
        if self.gamma_alpha is not None:
            raise FrozenStateError(
                "category-3 excluded field 'gamma_alpha' is present — the restricted α=0 "
                "specification is violated"
            )
        for name in inventory.cat3_excluded:
            if name == "gamma_alpha":
                continue  # checked above (None is the excluded state, not a missing attribute)
            if hasattr(self, name):
                raise FrozenStateError(f"category-3 excluded field {name!r} leaked into the state")

        # (3) shapes / λ agreement
        if self.factor_count != lam.factor_count:
            raise FrozenStateError(
                f"manifest factor_count {self.factor_count} != λ.factor_count {lam.factor_count}"
            )
        if self.gamma_beta.shape != (self.instrument_count, self.factor_count):
            raise FrozenStateError(
                f"gamma_beta shape {self.gamma_beta.shape} != (L={self.instrument_count}, "
                f"K={self.factor_count})"
            )
        if self.instrument_count != len(self.characteristic_order) + 1:
            raise FrozenStateError(
                f"instrument_count {self.instrument_count} != len(characteristic_order)+1 "
                f"({len(self.characteristic_order) + 1}; +1 for the constant column)"
            )
        if tuple(self.characteristic_order) != tuple(lam.characteristic_order):
            raise FrozenStateError("characteristic_order does not match λ")
        if self.normalisation_rule != lam.normalisation_rule:
            raise FrozenStateError(
                f"normalisation_rule {self.normalisation_rule!r} != λ {lam.normalisation_rule!r}"
            )

        # (4) hash integrity
        if not self.verify_hash(lam):
            raise FrozenStateError("content hash does not recompute — the frozen arrays were mutated")
