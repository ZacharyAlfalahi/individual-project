"""
QuantConfig -- a typed, self-validating, provenance-carrying description of ONE
characteristic-sort run, plus the factory that builds it (with graceful
refusals) and the strip step that reduces it to the engine's plain-dict
rulebook.

    StrategySpec -> [agent] -> wrapped fields -> build_quant_config
                                                     |          |
                                              ConfigRefusal   QuantConfig
                                                                 |
                                                            to_rulebook -> engine

Flat structure (all fields at one level). Field *names* are engine-native so the
strip needs no key translation; the single value translation is
``weighting "size" -> "by_size"``. ``holding_period`` is carried here (an overlap
argument, not a rulebook key); ``nw_lags`` / ``months_per_year`` are left to the
engine's own defaults.

Two failure modes, deliberately separated:
  * ConfigRefusal  -- a legitimate strategy the engine stack cannot represent
                      (recorded for RQ2). Produced by ``build_quant_config``.
  * QuantConfigError -- malformed agent output (an extraction bug). Raised by the
                      factory / ``__post_init__``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeGuard

from .provenance import Binding, Evidence, Inherited
from .refusal import ConfigRefusal, RefusalCode
from .trim_rule import TrimRule


class QuantConfigError(ValueError):
    """Malformed but structurally-supported config input -- an agent bug, not an
    unsupportable strategy (which is a ``ConfigRefusal`` instead)."""


_VALID_WEIGHTINGS: tuple[str, ...] = ("size", "equal")
_SUPPORTED_TRIM_METHODS: tuple[str, ...] = ("none", "truncate", "winsorise")


def _is_int(v: object) -> TypeGuard[int]:
    return isinstance(v, int) and not isinstance(v, bool)


def _require_int(value: object, name: str) -> int:
    """Return ``value`` as an ``int`` or raise -- also narrows the type for the
    caller (bool is rejected: it is an int subclass but never a valid count)."""
    if not _is_int(value):
        raise QuantConfigError(f"{name} must be an int; got {value!r}")
    return value


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QuantConfig:
    strategy_id: str
    score: Binding
    groups: Inherited[int]
    weighting: Inherited[str]
    signal_lag: Inherited[int]
    min_bonds: Inherited[int]
    long_group: Inherited[int]
    short_group: Inherited[int]
    control_groups: Inherited[int]
    trim_rule: Inherited[TrimRule]
    holding_period: Inherited[int]
    control: Binding | None = None

    def __post_init__(self) -> None:
        # Wrapper-type guard: a QuantConfig field must be provenance-wrapped.
        # This brands a hand-built config that passes raw values (bypassing the
        # factory) as a QuantConfigError rather than an off-taxonomy AttributeError.
        if not isinstance(self.score, Binding):
            raise QuantConfigError("score must be a Binding")
        if self.control is not None and not isinstance(self.control, Binding):
            raise QuantConfigError("control must be a Binding or None")
        for _name, _field in (
            ("groups", self.groups), ("weighting", self.weighting),
            ("signal_lag", self.signal_lag), ("min_bonds", self.min_bonds),
            ("long_group", self.long_group), ("short_group", self.short_group),
            ("control_groups", self.control_groups), ("trim_rule", self.trim_rule),
            ("holding_period", self.holding_period),
        ):
            if not isinstance(_field, Inherited):
                raise QuantConfigError(f"{_name} must be an Inherited")

        # Bindings must be resolved -- a MISSING binding never becomes a
        # QuantConfig (the factory refuses first); this guards hand-built ones.
        if not self.score.is_usable:
            raise QuantConfigError("score binding must be resolved (BOUND/AMBIGUOUS)")
        if self.control is not None and not self.control.is_usable:
            raise QuantConfigError("control binding must be resolved (BOUND/AMBIGUOUS) or None")

        g = _require_int(self.groups.value, "groups")
        if g < 2:
            raise QuantConfigError(f"groups must be an int >= 2; got {g}")

        signal_lag = _require_int(self.signal_lag.value, "signal_lag")
        if signal_lag < 0:
            raise QuantConfigError(f"signal_lag must be >= 0; got {signal_lag}")
        min_bonds = _require_int(self.min_bonds.value, "min_bonds")
        if min_bonds < 1:
            raise QuantConfigError(f"min_bonds must be >= 1; got {min_bonds}")
        holding_period = _require_int(self.holding_period.value, "holding_period")
        if holding_period < 1:
            raise QuantConfigError(f"holding_period must be >= 1; got {holding_period}")

        lg = _require_int(self.long_group.value, "long_group")
        sg = _require_int(self.short_group.value, "short_group")
        if not (0 <= lg < g):
            raise QuantConfigError(f"long_group must be in [0,{g}); got {lg}")
        if not (0 <= sg < g):
            raise QuantConfigError(f"short_group must be in [0,{g}); got {sg}")
        if lg == sg:
            raise QuantConfigError("long_group and short_group must differ")

        # Normally caught as a refusal by the factory; a safety net for
        # hand-built configs so a bad weighting can never reach the engine.
        if self.weighting.value not in _VALID_WEIGHTINGS:
            raise QuantConfigError(
                f"weighting must be one of {_VALID_WEIGHTINGS}; got {self.weighting.value!r}"
            )
        if not isinstance(self.trim_rule.value, TrimRule):
            raise QuantConfigError("trim_rule.value must be a TrimRule")

        # control_groups is validated unconditionally (default = groups >= 2),
        # so a malformed value is never silently swallowed on a single sort even
        # though to_rulebook only emits it when a control is present.
        control_groups = _require_int(self.control_groups.value, "control_groups")
        if control_groups < 2:
            raise QuantConfigError(f"control_groups must be >= 2; got {control_groups}")


# ---------------------------------------------------------------------------
# Factory: wrapped fields -> QuantConfig | ConfigRefusal
# ---------------------------------------------------------------------------

def _unknown(value: object, note: str) -> Inherited:
    return Inherited(value, "UNKNOWN", Evidence(note=note))


def _triage_trim(raw: dict) -> tuple[TrimRule | None, str | None]:
    """Map a raw engine-style trim spec to a ``TrimRule``, or to a refusal
    reason when it names an unsupported (but well-formed) variant. Malformed
    supported input (e.g. a truncate with no bounds) is left to ``TrimRule`` to
    raise on -- that is an agent bug, not a refusal."""
    method = raw.get("method", "none")
    if method not in _SUPPORTED_TRIM_METHODS:
        return None, f"trim method {method!r} not supported"
    if method == "none":
        return TrimRule(method="none"), None
    target = raw.get("target", "return")
    if target != "return":
        return None, f"trim target {target!r} not supported (only 'return')"
    bounds = raw.get("bounds", {})
    if not isinstance(bounds, dict):
        return None, "trim bounds must be a mapping"
    btype = bounds.get("type", "absolute")
    if btype != "absolute":
        return None, f"trim bounds.type {btype!r} not supported (only 'absolute')"
    sample = raw.get("sample", "full_sample")
    if sample != "full_sample":
        return None, f"trim sample {sample!r} not supported (only 'full_sample')"
    return TrimRule(method=method, lo=bounds.get("lo"), hi=bounds.get("hi")), None


def build_quant_config(
    strategy_id: str,
    score: Binding,
    *,
    control: Binding | None = None,
    groups: Inherited | None = None,
    weighting: Inherited | None = None,
    signal_lag: Inherited | None = None,
    min_bonds: Inherited | None = None,
    long_group: Inherited | None = None,
    short_group: Inherited | None = None,
    control_groups: Inherited | None = None,
    trim: Inherited | None = None,
    holding_period: Inherited | None = None,
) -> QuantConfig | ConfigRefusal:
    """
    Assemble a ``QuantConfig`` from already-provenance-wrapped fields, or return
    a ``ConfigRefusal`` when the strategy cannot be represented. The upstream
    "paper -> wrapped fields" adapter is the Librarian's work (out of scope).

    Inputs are the wrapped fields; any left ``None`` are filled with the engine
    default, stamped ``UNKNOWN`` (or ``DESIGN`` for the par-weighting default).
    Refusal conditions are checked before construction; ``strategy_id`` is read
    first so every refusal path can be stamped (including a MISSING ``score``,
    where no ``QuantConfig`` is ever built).
    """
    # --- Trust boundary: raw (non-wrapped) agent output is malformed, not a
    # refusal -> raise a branded QuantConfigError (never a bare AttributeError).
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise QuantConfigError("strategy_id must be a non-empty string")
    if not isinstance(score, Binding):
        raise QuantConfigError(f"score must be a Binding; got {type(score).__name__}")
    if control is not None and not isinstance(control, Binding):
        raise QuantConfigError(f"control must be a Binding or None; got {type(control).__name__}")
    for _name, _field in (
        ("groups", groups), ("weighting", weighting), ("signal_lag", signal_lag),
        ("min_bonds", min_bonds), ("long_group", long_group), ("short_group", short_group),
        ("control_groups", control_groups), ("trim", trim), ("holding_period", holding_period),
    ):
        if _field is not None and not isinstance(_field, Inherited):
            raise QuantConfigError(
                f"{_name} must be an Inherited or None; got {type(_field).__name__}"
            )

    # --- Bindings: MISSING -> refuse (never guess) --------------------------
    if score.tag == "MISSING":
        return ConfigRefusal(
            strategy_id, RefusalCode.MISSING_BINDING, "score",
            "the strategy signal has no corresponding column in the panel",
            score.evidence,
        )
    if control is not None and control.tag == "MISSING":
        return ConfigRefusal(
            strategy_id, RefusalCode.MISSING_BINDING, "control",
            "a double sort was described but the control column could not be bound",
            control.evidence,
        )

    # --- groups first: groups-relative defaults depend on it ----------------
    groups_w = groups if groups is not None else _unknown(5, "not stated; engine default groups=5")
    g = _require_int(groups_w.value, "groups")
    if g < 2:
        raise QuantConfigError(f"groups must be an int >= 2; got {g}")

    # --- weighting: out-of-enum -> refuse (a legit but unrepresentable scheme)
    weighting_w = (
        weighting
        if weighting is not None
        else Inherited("size", "DESIGN", Evidence(
            note="project default: par-weighting via panel size column (BBW spec §2.4/§5)"
        ))
    )
    if weighting_w.value not in _VALID_WEIGHTINGS:
        return ConfigRefusal(
            strategy_id, RefusalCode.OUT_OF_ENUM_WEIGHTING, "weighting",
            f"weighting {weighting_w.value!r} cannot be represented "
            f"(engine supports {_VALID_WEIGHTINGS})",
            weighting_w.evidence,
        )

    # --- trim: unsupported variant -> refuse; malformed-supported -> raise ---
    if trim is None:
        trim_w: Inherited = _unknown(TrimRule(method="none"), "not stated; engine default: no trim")
    else:
        raw_trim = trim.value
        if not isinstance(raw_trim, dict):
            raise QuantConfigError(f"trim spec must be a mapping (dict); got {raw_trim!r}")
        built, reason = _triage_trim(raw_trim)
        if built is None:
            return ConfigRefusal(
                strategy_id, RefusalCode.UNSUPPORTED_TRIM_VARIANT, "trim_rule",
                reason or "unsupported trim variant", trim.evidence,
            )
        trim_w = Inherited(built, trim.tag, trim.evidence)

    # --- holding_period + the control x holding>1 combination ---------------
    holding_w = holding_period if holding_period is not None else _unknown(
        1, "not stated; engine default holding_period=1"
    )
    h = holding_w.value
    control_usable = control is not None and control.is_usable
    if control_usable and _is_int(h) and h > 1:
        return ConfigRefusal(
            strategy_id, RefusalCode.UNSUPPORTED_COMBINATION, "holding_period",
            "a double sort (control) held for more than one month is not supported "
            "(overlap.run_with_holding_period cannot double-sort)",
            None,
        )

    # --- groups-relative defaults (cannot be static dataclass defaults) ------
    min_bonds_w = min_bonds if min_bonds is not None else _unknown(
        g, "not stated; engine default min_bonds=groups"
    )
    long_group_w = long_group if long_group is not None else _unknown(
        g - 1, "not stated; engine default long_group=groups-1 (top group)"
    )
    short_group_w = short_group if short_group is not None else _unknown(
        0, "not stated; engine default short_group=0 (bottom group)"
    )
    control_groups_w = control_groups if control_groups is not None else _unknown(
        g, "not stated; engine default control_groups=groups"
    )
    signal_lag_w = signal_lag if signal_lag is not None else _unknown(
        0, "not stated; engine default signal_lag=0 (contemporaneous)"
    )

    # --- construct (may raise QuantConfigError on malformed-supported input) -
    return QuantConfig(
        strategy_id=strategy_id,
        score=score,
        groups=groups_w,
        weighting=weighting_w,
        signal_lag=signal_lag_w,
        min_bonds=min_bonds_w,
        long_group=long_group_w,
        short_group=short_group_w,
        control_groups=control_groups_w,
        trim_rule=trim_w,
        holding_period=holding_w,
        control=control,
    )


# ---------------------------------------------------------------------------
# Strip: QuantConfig -> engine rulebook
# ---------------------------------------------------------------------------

def to_rulebook(config: QuantConfig) -> dict:
    """Strip provenance to the bare engine rulebook.

    * ``weighting "size" -> "by_size"`` (the one value translation).
    * ``control`` / ``control_groups`` are emitted only when the control binding
      resolved (BOUND or AMBIGUOUS) -- an AMBIGUOUS-but-resolved binding is
      usable and must not be dropped.
    * ``holding_period`` is omitted (an overlap argument, not a rulebook key);
      ``nw_lags`` / ``months_per_year`` are omitted (engine fills its defaults).
    """
    weighting = "by_size" if config.weighting.value == "size" else config.weighting.value
    trim = config.trim_rule.value
    assert isinstance(trim, TrimRule)  # invariant: QuantConfig.__post_init__ guarantees it
    rulebook: dict = {
        "score": config.score.value,
        "groups": config.groups.value,
        "weighting": weighting,
        "signal_lag": config.signal_lag.value,
        "min_bonds": config.min_bonds.value,
        "long_group": config.long_group.value,
        "short_group": config.short_group.value,
        "trim_rule": trim.to_engine_dict(),
    }
    if config.control is not None and config.control.is_usable:
        rulebook["control"] = config.control.value
        rulebook["control_groups"] = config.control_groups.value
    return rulebook
