"""The §12 funnels (denominators visible, monotone) and the §8.3 correction-translation scorer
(headline UNREPORTABLE until the Auditor-invariance NA gold lands — never silently zero)."""

import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.scientist.reporting.funnels import agent_quality_funnel, economic_funnel  # noqa: E402
from agents.scientist.researcher.translation import (  # noqa: E402
    ToggleCorrection,
    score_translation,
    summarize_translations,
)
from agents.scientist.schemas.evaluation import BOOLEAN_FIELDS, Booleans, EvaluationRecord  # noqa: E402


def _record(**over):
    d = {f: False for f in BOOLEAN_FIELDS}
    d.update(over)
    return EvaluationRecord(proposal_id="p", booleans=Booleans(**d))


def _g0_true(**over):
    base = dict(schema_valid=True, mechanism_authorised=True, template_supported=True,
                toggles_preserved=True, inputs_available=True, not_duplicate=True)
    base.update(over)
    return _record(**base)


# ---- funnels ------------------------------------------------------------------------------

def test_economic_funnel_is_monotone_non_increasing():
    records = [
        _record(schema_valid=False),                                   # invalid
        _g0_true(compiled=True, execution_verified=True, audit_clean=False),  # audit failure
        _g0_true(compiled=True, execution_verified=True, audit_clean=True, bh_survived=True,
                 cpcv_qualified=True),                                  # a survivor
    ]
    f = economic_funnel(records)
    assert f == {"generated": 3, "valid": 2, "compiled": 2, "executed": 2, "audit_clean": 1,
                 "bh_survivor": 1, "cpcv_qualified": 1, "holdout_evaluated": 0}
    stages = [f[k] for k in ("generated", "valid", "compiled", "executed", "audit_clean",
                             "bh_survivor", "cpcv_qualified", "holdout_evaluated")]
    assert all(stages[i] >= stages[i + 1] for i in range(len(stages) - 1))   # monotone


@dataclass
class _Gen:
    n_requested: int
    n_valid_unique: int
    n_invalid: int
    n_duplicate: int


def test_agent_quality_funnel_rates():
    f = agent_quality_funnel([_Gen(6, 4, 1, 1), _Gen(6, 6, 0, 0)])
    assert f["requested"] == 12 and f["valid_unique"] == 10
    assert f["invalid_rate"] == 1 / 12 and f["duplicate_rate"] == 1 / 12


# ---- correction-translation scorer --------------------------------------------------------

def test_score_translation_exact_and_wrong():
    gold = ToggleCorrection(toggle="lib_gap", polarity="ON", field="signal_lag", target=1)
    exact = score_translation("i1", gold, gold, is_na=False)
    assert exact.exact_translation
    wrong = score_translation("i2", ToggleCorrection(toggle="lab_trim", polarity="ON",
                              field="signal_lag", target=1), gold, is_na=False)
    assert not wrong.exact_translation and not wrong.correct_toggle


def test_na_item_change_nothing():
    na_gold = ToggleCorrection(refused=True)                           # 'change nothing'
    right = score_translation("na1", ToggleCorrection(refused=True), na_gold, is_na=True)
    assert right.exact_translation and right.correct_refusal
    wrong = score_translation("na2", ToggleCorrection(toggle="lib_gap", polarity="ON"),
                              na_gold, is_na=True)
    assert not wrong.exact_translation                                 # invented a change


def test_headline_unreportable_until_na_gold_lands():
    scores = [score_translation("i1", ToggleCorrection(toggle="lib_gap"),
                                ToggleCorrection(toggle="lib_gap"), is_na=False)]
    rpt = summarize_translations(scores, na_gold_available=False)
    assert rpt.reportable is False
    assert rpt.exact_translation_rate is None                          # None, NEVER silently 0
    assert rpt.na_item_accuracy is None
    assert rpt.additional_change_error_rate == 0.0                     # collateral rate IS scorable
    ok = summarize_translations(scores, na_gold_available=True)
    assert ok.reportable and ok.exact_translation_rate == 1.0
