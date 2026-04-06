"""Tests for DJSI Diversified eligibility filters."""

from __future__ import annotations

import pytest

from indices.djsi_diversified.eligibility import (
    apply_hard_exclusions,
    exclude_adult_entertainment,
    exclude_alcohol,
    exclude_controversial_weapons,
    exclude_gambling,
    exclude_military_contracting,
    exclude_msa_flagged,
    exclude_no_esg_coverage,
    exclude_small_arms,
    exclude_tobacco,
    exclude_ungc_non_compliant,
    select_best_in_class,
)
from indices.djsi_diversified.models import UNGCStatus
from .conftest import make_stock, make_universe


class TestExcludeNoESGCoverage:
    def test_no_coverage_excluded(self):
        s = make_stock(ticker="X", has_esg_coverage=False, esg_score=None)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_no_esg_coverage(df).iloc[0]

    def test_null_score_excluded(self):
        s = make_stock(ticker="X", esg_score=None)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_no_esg_coverage(df).iloc[0]

    def test_valid_score_not_excluded(self):
        s = make_stock(ticker="X", esg_score=80.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_no_esg_coverage(df).iloc[0]


class TestExcludeControversialWeapons:
    def test_revenue_above_zero_excluded(self):
        s = make_stock(ticker="X", controversial_weapons_revenue_pct=0.5)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_at_threshold_excluded(self):
        s = make_stock(ticker="X", controversial_weapons_ownership_pct=10.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_controversial_weapons(df).iloc[0]

    def test_zero_revenue_zero_ownership_not_excluded(self):
        s = make_stock(ticker="X")
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_controversial_weapons(df).iloc[0]


class TestExcludeTobacco:
    def test_any_production_excluded(self):
        s = make_stock(ticker="X", tobacco_production_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_tobacco(df).iloc[0]

    def test_zero_production_not_excluded(self):
        s = make_stock(ticker="X", tobacco_production_revenue_pct=0.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_tobacco(df).iloc[0]


class TestExcludeAdultEntertainment:
    def test_production_at_threshold_excluded(self):
        s = make_stock(ticker="X", adult_entertainment_production_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_adult_entertainment(df).iloc[0]

    def test_retail_at_threshold_excluded(self):
        s = make_stock(ticker="X", adult_entertainment_retail_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_adult_entertainment(df).iloc[0]

    def test_below_threshold_not_excluded(self):
        s = make_stock(ticker="X", adult_entertainment_production_revenue_pct=4.9)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_adult_entertainment(df).iloc[0]


class TestExcludeAlcohol:
    def test_any_production_excluded(self):
        s = make_stock(ticker="X", alcohol_production_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_alcohol(df).iloc[0]

    def test_zero_production_not_excluded(self):
        s = make_stock(ticker="X", alcohol_production_revenue_pct=0.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_alcohol(df).iloc[0]


class TestExcludeGambling:
    def test_any_operations_excluded(self):
        s = make_stock(ticker="X", gambling_operations_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_gambling(df).iloc[0]

    def test_any_equipment_excluded(self):
        s = make_stock(ticker="X", gambling_equipment_revenue_pct=0.1)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_gambling(df).iloc[0]

    def test_zero_not_excluded(self):
        s = make_stock(ticker="X")
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_gambling(df).iloc[0]


class TestExcludeMilitaryContracting:
    def test_integral_weapons_at_threshold_excluded(self):
        s = make_stock(ticker="X", military_integral_weapons_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_military_contracting(df).iloc[0]

    def test_weapon_related_at_threshold_excluded(self):
        s = make_stock(ticker="X", military_weapon_related_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_military_contracting(df).iloc[0]

    def test_below_threshold_not_excluded(self):
        s = make_stock(ticker="X", military_integral_weapons_revenue_pct=4.9)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_military_contracting(df).iloc[0]


class TestExcludeSmallArms:
    def test_civilian_production_at_threshold_excluded(self):
        s = make_stock(ticker="X", small_arms_civilian_production_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_small_arms(df).iloc[0]

    def test_key_components_at_threshold_excluded(self):
        s = make_stock(ticker="X", small_arms_key_components_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_small_arms(df).iloc[0]

    def test_noncivilian_at_threshold_excluded(self):
        s = make_stock(ticker="X", small_arms_noncivilian_production_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_small_arms(df).iloc[0]

    def test_retail_at_threshold_excluded(self):
        s = make_stock(ticker="X", small_arms_retail_revenue_pct=5.0)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_small_arms(df).iloc[0]

    def test_below_threshold_not_excluded(self):
        s = make_stock(ticker="X", small_arms_civilian_production_revenue_pct=4.9)
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_small_arms(df).iloc[0]


class TestExcludeUNGC:
    def test_non_compliant_excluded(self):
        s = make_stock(ticker="X", ungc_status=UNGCStatus.NON_COMPLIANT)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_no_coverage_excluded(self):
        s = make_stock(ticker="X", ungc_status=UNGCStatus.NO_COVERAGE)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_compliant_not_excluded(self):
        s = make_stock(ticker="X")
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_ungc_non_compliant(df).iloc[0]


class TestExcludeMSA:
    def test_flagged_excluded(self):
        s = make_stock(ticker="X", msa_flagged=True)
        u = make_universe(s)
        df = u.to_dataframe()
        assert exclude_msa_flagged(df).iloc[0]

    def test_not_flagged_not_excluded(self):
        s = make_stock(ticker="X")
        u = make_universe(s)
        df = u.to_dataframe()
        assert not exclude_msa_flagged(df).iloc[0]


class TestApplyHardExclusions:
    def test_clean_stocks_pass_through(self):
        stocks = [make_stock(ticker=f"T{i}") for i in range(5)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        remaining, excluded = apply_hard_exclusions(df)
        assert len(remaining) == 5
        assert len(excluded) == 0

    def test_multiple_exclusion_reasons(self):
        clean = make_stock(ticker="CLEAN")
        tobacco = make_stock(ticker="TOBACCO", tobacco_production_revenue_pct=5.0)
        weapons = make_stock(ticker="WEAPONS", controversial_weapons_revenue_pct=1.0)
        u = make_universe(clean, tobacco, weapons)
        df = u.to_dataframe()
        remaining, excluded = apply_hard_exclusions(df)
        assert len(remaining) == 1
        assert "TOBACCO" in excluded
        assert "WEAPONS" in excluded

    def test_first_matching_reason_recorded(self):
        s = make_stock(
            ticker="MULTI",
            tobacco_production_revenue_pct=5.0,
            gambling_operations_revenue_pct=2.0,
        )
        u = make_universe(s)
        df = u.to_dataframe()
        _, excluded = apply_hard_exclusions(df)
        assert excluded["MULTI"] == "Tobacco"


class TestSelectBestInClass:
    def test_selects_top_esg_stocks(self):
        # Same sector+region, different ESG scores. Should select top scorers
        # that cover ~50% of FMC.
        stocks = [
            make_stock(ticker="TOP1", esg_score=95.0, market_cap_usd=1e9),
            make_stock(ticker="TOP2", esg_score=90.0, market_cap_usd=1e9),
            make_stock(ticker="MID", esg_score=70.0, market_cap_usd=1e9),
            make_stock(ticker="LOW", esg_score=50.0, market_cap_usd=1e9),
        ]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        selected = select_best_in_class(df)
        assert "TOP1" in selected
        assert "TOP2" in selected

    def test_respects_sector_region_grouping(self):
        # Two different sector-region groups
        s1 = make_stock(ticker="NA_IT", esg_score=90.0, gics_sector="IT", region="North America")
        s2 = make_stock(ticker="EU_FIN", esg_score=85.0, gics_sector="Financials", region="EMEA")
        s3 = make_stock(ticker="NA_IT2", esg_score=60.0, gics_sector="IT", region="North America")
        s4 = make_stock(ticker="EU_FIN2", esg_score=55.0, gics_sector="Financials", region="EMEA")
        u = make_universe(s1, s2, s3, s4)
        df = u.to_dataframe()
        selected = select_best_in_class(df)
        # Each group should have at least one selection
        assert "NA_IT" in selected
        assert "EU_FIN" in selected
