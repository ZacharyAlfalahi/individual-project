"""Correction-translation task (spec §8.3) — a DIAGNOSTIC side-branch. Its output is NEVER
executed; it measures TRANSLATION accuracy (can the model emit the exact typed toggle correction
for a diagnosis?), an RQ2-adjacent capability, scored against the toggle-registry gold.

The check IDs and toggle names are the same strings, so a bare "identification" would be string
matching — the DUMMY / NA items ("change nothing", where a paper applied no bias) are what string
matching cannot fake, and they are the whole point. Those NA gold answers are DERIVED from the
Auditor's bit-exact invariance results (a machine-verified known-null), which do NOT exist until
the Auditor real-data runs land. So the HEADLINE accuracy is UNREPORTABLE until then — marked
explicitly, never silently reported as zero (`mom6 x lib_gap` is a near-dummy, NOT a valid
known-null, and may never be a gold NA item).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToggleCorrection:
    """A model's (or the gold) translation of a diagnosis into a typed toggle correction. A
    'change nothing' answer is `refused=True` with no toggle/edits."""
    toggle: str | None = None
    polarity: str | None = None
    field: str | None = None
    target: object | None = None
    collateral_edits: tuple = ()
    refused: bool = False


@dataclass(frozen=True)
class TranslationScore:
    item_id: str
    is_na: bool
    correct_toggle: bool
    correct_polarity: bool
    correct_field: bool
    correct_target: bool
    zero_collateral: bool
    correct_refusal: bool

    @property
    def exact_translation(self) -> bool:
        return all((self.correct_toggle, self.correct_polarity, self.correct_field,
                    self.correct_target, self.zero_collateral, self.correct_refusal))


def score_translation(item_id: str, model: ToggleCorrection, gold: ToggleCorrection,
                      *, is_na: bool) -> TranslationScore:
    """Field-level edit precision of a single translation against its gold."""
    return TranslationScore(
        item_id=item_id, is_na=is_na,
        correct_toggle=model.toggle == gold.toggle,
        correct_polarity=model.polarity == gold.polarity,
        correct_field=model.field == gold.field,
        correct_target=model.target == gold.target,
        zero_collateral=len(model.collateral_edits) == 0,
        correct_refusal=model.refused == gold.refused,
    )


@dataclass(frozen=True)
class TranslationReport:
    n: int
    additional_change_error_rate: float          # any collateral edit is an error (always scorable)
    exact_translation_rate: float | None         # None until reportable (NEVER silently 0)
    na_item_accuracy: float | None               # None until reportable
    reportable: bool
    note: str


def summarize_translations(scores, *, na_gold_available: bool) -> TranslationReport:
    """The headline exact-translation rate and NA accuracy are UNREPORTABLE until the NA gold (from
    the Auditor's bit-exact invariance results) lands — reported as None, never 0, so an incomplete
    scorer can never appear in a funnel as if it were complete. The collateral-edit error rate is
    scorable now (it does not need the NA gold)."""
    n = len(scores)
    collateral_err = (sum(1 for s in scores if not s.zero_collateral) / n) if n else 0.0
    if not na_gold_available:
        return TranslationReport(
            n=n, additional_change_error_rate=collateral_err, exact_translation_rate=None,
            na_item_accuracy=None, reportable=False,
            note="headline UNREPORTABLE: NA gold (Auditor invariance results) not yet available")
    exact = (sum(1 for s in scores if s.exact_translation) / n) if n else 0.0
    na = [s for s in scores if s.is_na]
    na_acc = (sum(1 for s in na if s.exact_translation) / len(na)) if na else None
    return TranslationReport(n=n, additional_change_error_rate=collateral_err,
                             exact_translation_rate=exact, na_item_accuracy=na_acc,
                             reportable=True, note="reportable")
