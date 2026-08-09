"""one-shot holdout seed-start derivation (spec §2) — driven by module config keys, fail-loud on dev floor."""

from __future__ import annotations

import pytest

from agents.scientist.experimentalist.oneshot_holdout.seed_start import (
    SeedStartError,
    collect_module_burn_ins,
    derive_seed_start,
)


def test_collect_burn_ins_reads_module_configs():
    sources = {s.label: s.months for s in collect_module_burn_ins()}
    # From docs/thresholds.yaml: var_5pct.min_obs=24, mom6.min_obs=6, bond_vol.min_obs=18;
    # IPCA recursive-OOS burn-in=36 (module default). Max = 36.
    assert sources["var_5pct"] == 24
    assert sources["mom6"] == 6
    assert sources["bond_vol"] == 18
    assert sources["ipca_oos"] == 36


def test_seed_start_uses_max_burn_in_plus_margin():
    # 2022-01 − (36 + 3) months = 2018-10 (the IPCA OOS burn-in dominates).
    seed_start, sources = derive_seed_start("2022-01")
    assert seed_start == "2018-10"
    assert max(s.months for s in sources) == 36


def test_seed_start_margin_is_applied():
    assert derive_seed_start("2022-01", margin_months=0)[0] == "2019-01"   # 2022-01 − 36


def test_seed_start_refuses_to_precede_development_start():
    # A window start close to the development floor would push the seed before it — fail loud.
    with pytest.raises(SeedStartError):
        derive_seed_start("2002-08")           # 2002-08 − 39 = 1999-05 < 2002-07 dev start
