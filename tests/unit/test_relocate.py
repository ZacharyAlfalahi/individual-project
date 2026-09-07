"""Relocator unit tests (evaluation/harness/relocate.py) — synthetic
in-memory canonical texts only; no archives, no gold, no LLM."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from agents.librarian.config.canonical_text import CanonicalText            # noqa: E402
from agents.librarian.config.normalise import normalise                     # noqa: E402
from evaluation.harness import relocate as relocate_mod                     # noqa: E402
from evaluation.harness.perturb import PerturbError                         # noqa: E402
from evaluation.harness.relocate import (                                   # noqa: E402
    RelocateError,
    relocate,
    score_quote,
)

BAR = 0.8
MIN_CHARS = 30

_SENTENCE = ("We sort bonds into quintiles based on their downside risk and form "
             "value-weighted portfolios each month.")
_SWAPPED = ("We sort bonds into deciles based on their downside risk and form "
            "value-weighted portfolios each month.")

_PAGE0 = ("Introduction. We study corporate credit markets in a large panel. "
          + _SENTENCE + " Additional context follows with several unrelated remarks.")
_PAGE1 = ("Robustness. The findings hold across subsamples and alternative "
          "specifications throughout the sample period considered here.")
_PAGE2 = ("Conclusion. Transaction expenses are computed from quoted spreads and "
          "reported separately for investment grade and high yield segments.")


def _ct(pages: tuple[str, ...]) -> CanonicalText:
    return CanonicalText(
        source_pdf="stub.pdf", source_sha256="deadbeef",
        parser={"name": "stub", "version": "0"},
        normalisation={"ladder_level": "L1", "rules": []},
        pages=pages, status="stub",
    )


def _relocate(ct, quote, *, bar=BAR, min_quote_chars=MIN_CHARS):
    return relocate(ct, quote, bar=bar, min_quote_chars=min_quote_chars)


def test_exact_match_short_circuits_with_score_one():
    ct = _ct((_PAGE0, _PAGE1, _PAGE2))
    res = _relocate(ct, _SENTENCE)
    assert res.accepted and res.method == "exact" and res.reason == "exact_match"
    assert res.score == 1.0
    assert res.locator is not None and ct.slice_text(res.locator) == normalise(_SENTENCE, "L1")


def test_relocates_word_substituted_quote():
    ct = _ct((_PAGE0, _PAGE1, _PAGE2))
    res = _relocate(ct, _SWAPPED)
    assert res.accepted and res.method == "relocated" and res.reason == "accepted_at_bar"
    assert BAR <= res.score < 1.0
    # The certified span is the paper's own sentence, not the model's quote.
    assert res.l1_text == _SENTENCE
    assert res.locator.page == 0 and not res.cross_page


def test_rejects_cross_paper_quote():
    ct = _ct((_PAGE0, _PAGE1, _PAGE2))
    quote = ("Deep learning architectures forecast equity volatility using intraday "
             "order flow imbalance signals across exchanges.")
    res = _relocate(ct, quote)
    assert not res.accepted and res.method == "rejected"
    assert res.reason in ("no_anchor", "below_bar")
    if res.reason == "below_bar":
        assert res.score < BAR


def test_min_length_guard_rejects_short_quotes():
    ct = _ct((_PAGE0,))
    res = _relocate(ct, "not present here", bar=0.0)
    assert not res.accepted and res.reason == "too_short" and res.score == 0.0


def test_empty_and_whitespace_quote_rejected():
    ct = _ct((_PAGE0,))
    for q in ("", "  \n\t "):
        res = _relocate(ct, q)
        assert not res.accepted and res.reason == "empty_quote"


def test_determinism_and_idempotence():
    decoy = ("The liquidity premium is estimated from repeated transactions within "
             "the same issuer and month across the panel.")
    swapped = decoy.replace("estimated", "computed")
    ct = _ct((decoy + " Filler text one.", _PAGE1, decoy + " Filler text two."))
    first = _relocate(ct, swapped)
    # Interleave a DIFFERENT canonical text: the search-text cache must not
    # cross-contaminate between texts (keys are page digests, not identities).
    other = _ct((_PAGE2, _PAGE1))
    assert not _relocate(other, swapped).accepted
    second = _relocate(ct, swapped)
    assert first == second
    assert first.accepted


def test_l0_certification_round_trip():
    raw_page = ("The eﬃcient market hypothesis\nfails for corporate bonds in our "
                "sample period overall, according to the evidence assembled here.")
    ct = _ct((raw_page,))
    quote = ("The efficient market hypothesis holds for corporate bonds in our "
             "sample period overall, according to the evidence")
    res = _relocate(ct, quote)
    assert res.accepted and res.method == "relocated"
    assert res.l0_span is not None
    l0s, l0e = res.l0_span
    assert ct.pages[0][l0s:l0e] == res.l0_text
    assert normalise(res.l0_text, "L1") == res.l1_text == ct.slice_text(res.locator)


def test_certification_failure_raises_relocate_error(monkeypatch):
    ct = _ct((_PAGE0,))

    def boom(*args, **kwargs):
        raise PerturbError("synthetic boundary failure")

    monkeypatch.setattr(relocate_mod, "map_l1_span_to_l0", boom)
    with pytest.raises(RelocateError, match="could not be L0-certified"):
        _relocate(ct, _SWAPPED)


def test_cross_page_span_relocates_with_end_page():
    page_a = ("Section four discusses measurement. We compute the liquidity measure "
              "as the average of")
    page_b = ("daily bid-ask spreads over the previous month for each bond in the "
              "sample, then aggregate to portfolios.")
    ct = _ct((page_a, page_b, _PAGE2))
    quote = ("We compute the liquidity measure as the mean of daily bid-ask spreads "
             "over the previous month for each bond in the sample")
    res = _relocate(ct, quote, bar=0.75)
    assert res.accepted and res.method == "relocated"
    assert res.cross_page and res.locator.end_page == res.locator.page + 1
    assert res.locator.page == 0
    assert ct.slice_text(res.locator) == res.l1_text
    assert normalise(res.l0_text, "L1") == res.l1_text


def test_tie_prefers_lowest_page_then_offset():
    decoy = ("The credit spread widens sharply around downgrade events in every "
             "rating category we examine in the data.")
    swapped = decoy.replace("widens", "moves")
    ct = _ct((decoy + " Padding sentence one here.", _PAGE1,
              decoy + " Padding sentence two here."))
    cand = score_quote(ct, swapped)
    assert cand is not None
    assert cand.page == 0
    assert cand.runner_up == pytest.approx(cand.score)


def test_runner_up_margin_recorded():
    original = ("Default probabilities rise monotonically across the five portfolios "
                "sorted on the distress measure in our sample.")
    corrupted = ("Default probabilities rise gradually across the ten portfolios "
                 "ranked on the alternative distress proxy in that sample.")
    swapped = original.replace("monotonically", "steadily")
    ct = _ct((original + " Tail padding here.", corrupted + " Other padding here."))
    cand = score_quote(ct, swapped)
    assert cand is not None and cand.page == 0
    assert 0.0 < cand.runner_up < cand.score


def test_page_indexing_is_zero_based_runtime_convention():
    ct = _ct((_PAGE1, _PAGE2, _PAGE0))
    res = _relocate(ct, _SWAPPED)
    assert res.accepted and res.locator.page == 2
    l0s, l0e = res.l0_span
    assert ct.pages[2][l0s:l0e] == res.l0_text
