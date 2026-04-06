"""Tests for PACT eligibility filters."""

from __future__ import annotations

import pytest

from indices.sp_pact.eligibility import (
    apply_exclusions,
    exclude_alcohol,
    exclude_controversial_weapons,
    exclude_fossil_fuel_revenue,
    exclude_gambling,
    exclude_military_contracting,
    exclude_msa_flagged,
    exclude_no_carbon_coverage,
    exclude_oil_sands,
    exclude_shale_oil_gas,
    exclude_small_arms,
    exclude_thermal_coal,
    exclude_tobacco,
    exclude_ungc_non_compliant,
)
from indices.sp_pact.models import UNGCStatus, Variant
from .conftest import make_stock, make_universe


class TestExcludeNoCarbonCoverage:
    def test_no_coverage_excluded(self):
        s = make_stock(ticker="X", has_carbon_coverage=False)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_no_carbon_coverage(df).iloc[0]

    def test_with_coverage_not_excluded(self):
        s = make_stock(ticker="X", has_carbon_coverage=True)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_no_carbon_coverage(df).iloc[0]


class TestExcludeControversialWeapons:
    def test_revenue_above_zero_excluded(self):
        s = make_stock(ticker="X", controversial_weapons_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_at_25_excluded(self):
        s = make_stock(ticker="X", controversial_weapons_ownership_pct=25.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_below_25_not_excluded(self):
        s = make_stock(ticker="X", controversial_weapons_ownership_pct=24.9)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_controversial_weapons(df).iloc[0]


class TestExcludeTobacco:
    def test_production_excluded_both_variants(self):
        s = make_stock(ticker="X", tobacco_production_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_tobacco(df, Variant.CTB).iloc[0]
        assert exclude_tobacco(df, Variant.PAB).iloc[0]

    def test_retail_5pct_excluded_pab_only(self):
        s = make_stock(ticker="X", tobacco_retail_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_tobacco(df, Variant.PAB).iloc[0]
        assert not exclude_tobacco(df, Variant.CTB).iloc[0]

    def test_retail_10pct_excluded_both(self):
        s = make_stock(ticker="X", tobacco_retail_revenue_pct=10.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_tobacco(df, Variant.CTB).iloc[0]
        assert exclude_tobacco(df, Variant.PAB).iloc[0]


class TestPABOnlyExclusions:
    def test_small_arms_excluded_pab(self):
        s = make_stock(ticker="X", small_arms_civilian_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_small_arms(df).iloc[0]

    def test_military_contracting_excluded_pab(self):
        s = make_stock(ticker="X", military_integral_weapons_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_military_contracting(df).iloc[0]

    def test_thermal_coal_at_threshold_excluded(self):
        s = make_stock(ticker="X", thermal_coal_generation_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_thermal_coal(df).iloc[0]

    def test_oil_sands_at_threshold_excluded(self):
        s = make_stock(ticker="X", oil_sands_extraction_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_oil_sands(df).iloc[0]

    def test_shale_at_threshold_excluded(self):
        s = make_stock(ticker="X", shale_oil_gas_extraction_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_shale_oil_gas(df).iloc[0]

    def test_gambling_at_threshold_excluded(self):
        s = make_stock(ticker="X", gambling_operations_revenue_pct=10.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_gambling(df).iloc[0]

    def test_alcohol_production_excluded(self):
        s = make_stock(ticker="X", alcohol_production_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_alcohol(df).iloc[0]

    def test_fossil_fuel_coal_excluded(self):
        s = make_stock(ticker="X", coal_revenue_pct=1.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_fossil_fuel_revenue(df).iloc[0]

    def test_fossil_fuel_oil_excluded(self):
        s = make_stock(ticker="X", oil_revenue_pct=10.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_fossil_fuel_revenue(df).iloc[0]


class TestApplyExclusions:
    def test_ctb_does_not_exclude_pab_only_activities(self):
        # Coal revenue < 1% should not be excluded in CTB
        s = make_stock(ticker="COAL", coal_revenue_pct=0.5)
        u = make_universe(s)
        df = u.to_dataframe()
        remaining, excluded = apply_exclusions(df, Variant.CTB)
        assert "COAL" not in excluded

    def test_pab_excludes_fossil_fuel(self):
        s = make_stock(ticker="COAL", coal_revenue_pct=2.0)
        u = make_universe(s)
        df = u.to_dataframe()
        remaining, excluded = apply_exclusions(df, Variant.PAB)
        assert "COAL" in excluded

    def test_shared_exclusions_apply_to_both(self):
        s = make_stock(ticker="WEAPONS", controversial_weapons_revenue_pct=1.0)
        u = make_universe(s)
        df = u.to_dataframe()
        _, ctb_excl = apply_exclusions(df, Variant.CTB)
        _, pab_excl = apply_exclusions(df, Variant.PAB)
        assert "WEAPONS" in ctb_excl
        assert "WEAPONS" in pab_excl

    def test_clean_stocks_pass(self):
        stocks = [make_stock(ticker=f"T{i}") for i in range(5)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        remaining, excluded = apply_exclusions(df, Variant.CTB)
        assert len(remaining) == 5
        assert len(excluded) == 0
