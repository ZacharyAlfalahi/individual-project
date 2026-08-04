"""
KPP gold loader (schema v1.2): parses the fitted-factor-model gold
(``gold_kpp_ipca.md``) into a full, validating ``StrategySpec`` -- 11 estimation
fields + a 29-instrument set + paper_facts + an in-code stub Part2. Every STATED
quote is bound to a REAL L1 locator in the FROZEN canonical text.

The accuracy gate is ``test_every_stated_quote_locates`` (re-locates every STATED
quote against the frozen text and asserts the offsets match the loader's locator).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (evaluation/*)

import pytest  # noqa: E402
import yaml  # noqa: E402

from agents.librarian.config.locate import locate_quote  # noqa: E402
from agents.librarian.schema.estimation_fields import ESTIMATION_FIELDS  # noqa: E402
from agents.librarian.validators import (  # noqa: E402
    validate_estimation_block,
    validate_librarian_spec,
)
from agents.librarian.registries import load_instrument_concept_registry  # noqa: E402
from evaluation.gold_specs.gold_loader import GoldParseError, is_binding, load_gold_spec  # noqa: E402
from evaluation.gold_specs.kpp_gold_loader import load_kpp_gold_spec  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def spec():
    return load_kpp_gold_spec("kpp")


@pytest.fixture(scope="module")
def frozen_pages():
    frozen = yaml.safe_load(
        (_REPO / "evaluation" / "canonical_texts" / "kpp_2023.frozen.yaml").read_text("utf-8")
    )
    return tuple(frozen["pages"])


# --- loads + validates ------------------------------------------------------

def test_kpp_loads_via_public_entry():
    # load_gold_spec("kpp") routes to the parallel estimation loader.
    s = load_gold_spec("kpp")
    assert s.estimation is not None and s.instruments is not None


def test_is_binding_true():
    assert is_binding("kpp") is True


def test_kpp_validates_fail_closed(spec):
    assert validate_librarian_spec(spec) == []
    assert validate_estimation_block(spec, load_instrument_concept_registry()) == []


# --- full coverage (a missing field is a hard error) ------------------------

def test_estimation_full_coverage(spec):
    for name in ESTIMATION_FIELDS:
        inh = getattr(spec.estimation, name)
        assert inh is not None, name
        assert inh.tag in ("STATED", "UNKNOWN")


def test_29_instruments_present_and_unique(spec):
    ids = [i.concept_id.value for i in spec.instruments.instruments]
    assert len(ids) == 29
    assert len(set(ids)) == 29


def test_reused_and_new_ids(spec):
    ids = {i.concept_id.value for i in spec.instruments.instruments}
    assert "past_6m_cumulative_return" in ids  # KPP "Mom. 6m" reuses the sort id
    assert "credit_rating" in ids              # KPP "Rating" reuses the sort id
    assert "bond_var_36m" in ids               # KPP VaR is a distinct new id
    assert "var_5pct" not in ids               # the generic sort VaR is NOT reused


# --- spot values ------------------------------------------------------------

def test_spot_values(spec):
    e = spec.estimation
    assert e.n_factors_preferred.value == 5
    assert e.n_factors_preferred.tag == "STATED"
    assert sorted(e.n_factors_tested.value) == [1, 2, 3, 4, 5]
    assert e.intercept_spec.value == "restricted"
    assert e.model_family.value == "instrumented_pca"
    assert e.oos_split.value == 36
    assert spec.part1.formation_structure.value == "estimated_factor_model"
    assert spec.paper_facts.sample_start.value == "1999-01"
    assert spec.paper_facts.sample_end.value == "2020-12"


def test_instrument_source_class_distribution(spec):
    from collections import Counter

    dist = Counter(i.source_class.value for i in spec.instruments.instruments)
    # 13 bond, 5 equity, 10 accounting, 1 macro (see papers/processed/kelly_2023_ipca.md)
    assert dist == Counter({"bond": 13, "accounting": 10, "equity": 5, "macro": 1})


def test_per_instrument_transform_lag_unknown(spec):
    # transform/lag are uniformly UNKNOWN (loader-injected); concept_id + source_class STATED.
    for i in spec.instruments.instruments:
        assert i.concept_id.tag == "STATED"
        assert i.source_class.tag == "STATED"
        assert i.transform.tag == "UNKNOWN"
        assert i.lag.tag == "UNKNOWN"


# --- THE accuracy gate: every STATED quote binds to the frozen text ---------

def _stated_quotes(spec):
    yield spec.part1.formation_structure
    yield spec.part1.asset_class
    yield spec.header.strategy_label
    for name in ESTIMATION_FIELDS:
        yield getattr(spec.estimation, name)
    for i in spec.instruments.instruments:
        yield i.concept_id
        yield i.source_class
    for name in ("sample_start", "sample_end", "universe_filter", "claimed_headline_metric"):
        yield getattr(spec.paper_facts, name)


def test_every_stated_quote_locates(spec, frozen_pages):
    checked = 0
    for inh in _stated_quotes(spec):
        if inh.tag != "STATED":
            continue
        loc = inh.evidence.locator
        assert loc is not None, inh
        m = locate_quote(frozen_pages, inh.evidence.quote, "L1")
        assert m.matched, f"NOT FOUND: {inh.evidence.quote!r}"
        assert not m.used_cross_page, f"cross-page: {inh.evidence.quote!r}"
        assert (m.page + 1, m.char_start, m.char_end) == (loc.page, loc.char_start, loc.char_end), (
            f"offset drift for {inh.evidence.quote!r}"
        )
        checked += 1
    # sanity: we actually exercised a large batch (2 part1 + label + ~11 estimation + ~58 instrument
    # cells + 4 paper_facts), not a trivially-empty loop.
    assert checked >= 40


# --- unknown anchor still raises -------------------------------------------

def test_unknown_anchor_raises():
    with pytest.raises(GoldParseError):
        load_kpp_gold_spec("not_kpp")
