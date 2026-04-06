"""
Tests for all eligibility filters in the S&P Carbon Aware Index Series.

Each exclusion rule is tested in isolation (unit) and the combined pipeline
is tested with the sample universe (integration).
"""

from __future__ import annotations

import pandas as pd
import pytest

from indices.sp_carbon_aware.eligibility import (
    apply_eligibility_filters,
    exclude_adult_entertainment,
    exclude_alcoholic_beverages,
    exclude_arctic_oil_gas,
    exclude_bottom_quartile_esg,
    exclude_controversial_weapons,
    exclude_gambling,
    exclude_missing_esg,
    exclude_msa_flagged,
    exclude_no_sustainalytics_coverage,
    exclude_oil_gas,
    exclude_oil_sands,
    exclude_shale_energy,
    exclude_thermal_coal,
    exclude_tobacco,
    exclude_ungc_non_compliant,
)
from indices.sp_carbon_aware.models import BusinessActivityExposures, UNGCStatus
from tests.conftest import make_stock, make_universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _df_from_stocks(stocks):
    universe = make_universe(stocks)
    return universe.to_dataframe()


def _single_df(**kwargs):
    """Return a single-row DataFrame for one stock with given overrides."""
    stock = make_stock(**kwargs)
    return _df_from_stocks([stock])


# ---------------------------------------------------------------------------
# ESG filters
# ---------------------------------------------------------------------------


class TestExcludeMissingESG:
    def test_excludes_no_coverage(self):
        df = _single_df(has_esg_coverage=False, esg_score=None)
        assert exclude_missing_esg(df).iloc[0]

    def test_excludes_null_score_with_coverage_flag(self):
        df = _single_df(has_esg_coverage=True, esg_score=None)
        assert exclude_missing_esg(df).iloc[0]

    def test_includes_valid_score(self):
        df = _single_df(has_esg_coverage=True, esg_score=60.0)
        assert not exclude_missing_esg(df).iloc[0]


class TestExcludeBottomQuartileESG:
    def _make_group(self, scores: list[float | None], ig: str = "Software & Services"):
        """Build a multi-stock DataFrame with specified ESG scores in one industry group."""
        stocks = [
            make_stock(
                ticker=f"T{i}",
                gics_industry_group=ig,
                esg_score=s,
                has_esg_coverage=s is not None,
            )
            for i, s in enumerate(scores)
        ]
        return _df_from_stocks(stocks)

    def test_bottom_25_excluded_developed(self):
        # 4 stocks with scores [10, 50, 60, 70].
        # pandas quantile(0.25) on [10,50,60,70] = 10 + 0.75*(50-10) = 40.
        # We use strict '<', so score 10 (<40) is excluded; score 50 is not.
        df = self._make_group([10.0, 50.0, 60.0, 70.0])
        excluded = exclude_bottom_quartile_esg(df, universe_type="developed")
        assert excluded.iloc[0]   # score 10 excluded (10 < 40)
        assert not excluded.iloc[1]  # score 50 not excluded (50 >= 40)
        assert not excluded.iloc[2]
        assert not excluded.iloc[3]

    def test_emerging_treats_no_score_as_worst(self):
        # None is treated as -1 → bottom quartile in emerging
        df = self._make_group([None, 50.0, 60.0, 70.0])
        excluded = exclude_bottom_quartile_esg(df, universe_type="emerging")
        assert excluded.iloc[0]  # None → bottom

    def test_separate_industry_groups(self):
        # Stocks in different groups are screened independently
        stocks = [
            make_stock("A", gics_industry_group="Energy", esg_score=10.0),
            make_stock("B", gics_industry_group="Energy", esg_score=90.0),
            make_stock("C", gics_industry_group="Banks", esg_score=15.0),
            make_stock("D", gics_industry_group="Banks", esg_score=85.0),
        ]
        df = _df_from_stocks(stocks)
        excluded = exclude_bottom_quartile_esg(df, universe_type="developed")
        # A (10) is bottom quartile in Energy; C (15) is bottom in Banks
        assert excluded[df["ticker"] == "A"].iloc[0]
        assert excluded[df["ticker"] == "C"].iloc[0]
        assert not excluded[df["ticker"] == "B"].iloc[0]
        assert not excluded[df["ticker"] == "D"].iloc[0]


# ---------------------------------------------------------------------------
# Business activity filters
# ---------------------------------------------------------------------------


class TestExcludeControversialWeapons:
    def test_tailor_made_direct_involvement(self):
        ba = BusinessActivityExposures(controversial_weapons_tailor_made_essential_pct=0.1)
        df = _single_df(business_activities=ba)
        assert exclude_controversial_weapons(df).iloc[0]

    def test_tailor_made_ownership_threshold(self):
        # Ownership exactly 10% → excluded (>=10%)
        ba = BusinessActivityExposures(controversial_weapons_tailor_made_essential_ownership_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_below_threshold_not_excluded(self):
        ba = BusinessActivityExposures(controversial_weapons_tailor_made_essential_ownership_pct=9.9)
        df = _single_df(business_activities=ba)
        assert not exclude_controversial_weapons(df).iloc[0]

    def test_non_tailor_made_direct(self):
        ba = BusinessActivityExposures(controversial_weapons_non_tailor_made_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_controversial_weapons(df).iloc[0]

    def test_zero_involvement_not_excluded(self):
        df = _single_df(business_activities=BusinessActivityExposures())
        assert not exclude_controversial_weapons(df).iloc[0]


class TestExcludeTobacco:
    def test_production_any_amount(self):
        ba = BusinessActivityExposures(tobacco_production_revenue_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_tobacco(df).iloc[0]

    def test_related_at_threshold(self):
        # >=5% → excluded
        ba = BusinessActivityExposures(tobacco_related_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_tobacco(df).iloc[0]

    def test_related_below_threshold(self):
        ba = BusinessActivityExposures(tobacco_related_revenue_pct=4.9)
        df = _single_df(business_activities=ba)
        assert not exclude_tobacco(df).iloc[0]

    def test_retail_at_threshold(self):
        ba = BusinessActivityExposures(tobacco_retail_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_tobacco(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_tobacco(df).iloc[0]


class TestExcludeThermalCoal:
    def test_extraction_any_amount(self):
        ba = BusinessActivityExposures(thermal_coal_extraction_revenue_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_thermal_coal(df).iloc[0]

    def test_power_generation_any_amount(self):
        ba = BusinessActivityExposures(thermal_coal_power_generation_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_thermal_coal(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_thermal_coal(df).iloc[0]


class TestExcludeOilSands:
    def test_any_extraction(self):
        ba = BusinessActivityExposures(oil_sands_extraction_revenue_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_oil_sands(df).iloc[0]

    def test_zero(self):
        df = _single_df()
        assert not exclude_oil_sands(df).iloc[0]


class TestExcludeShaleEnergy:
    def test_any_extraction(self):
        ba = BusinessActivityExposures(shale_energy_extraction_revenue_pct=1.0)
        df = _single_df(business_activities=ba)
        assert exclude_shale_energy(df).iloc[0]

    def test_zero(self):
        df = _single_df()
        assert not exclude_shale_energy(df).iloc[0]


class TestExcludeArcticOilGas:
    def test_any_extraction(self):
        ba = BusinessActivityExposures(arctic_oil_gas_extraction_revenue_pct=0.5)
        df = _single_df(business_activities=ba)
        assert exclude_arctic_oil_gas(df).iloc[0]

    def test_zero(self):
        df = _single_df()
        assert not exclude_arctic_oil_gas(df).iloc[0]


class TestExcludeOilGas:
    def test_production_any_amount(self):
        ba = BusinessActivityExposures(oil_gas_production_revenue_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_oil_gas(df).iloc[0]

    def test_generation_any_amount(self):
        ba = BusinessActivityExposures(oil_gas_generation_revenue_pct=0.1)
        df = _single_df(business_activities=ba)
        assert exclude_oil_gas(df).iloc[0]

    def test_supporting_at_threshold(self):
        ba = BusinessActivityExposures(oil_gas_supporting_revenue_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_oil_gas(df).iloc[0]

    def test_supporting_below_threshold(self):
        ba = BusinessActivityExposures(oil_gas_supporting_revenue_pct=9.9)
        df = _single_df(business_activities=ba)
        assert not exclude_oil_gas(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_oil_gas(df).iloc[0]


class TestExcludeGambling:
    def test_operations_at_threshold(self):
        ba = BusinessActivityExposures(gambling_operations_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_gambling(df).iloc[0]

    def test_operations_below_threshold(self):
        ba = BusinessActivityExposures(gambling_operations_revenue_pct=4.9)
        df = _single_df(business_activities=ba)
        assert not exclude_gambling(df).iloc[0]

    def test_equipment_at_threshold(self):
        ba = BusinessActivityExposures(gambling_equipment_revenue_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_gambling(df).iloc[0]

    def test_supporting_at_threshold(self):
        ba = BusinessActivityExposures(gambling_supporting_revenue_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_gambling(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_gambling(df).iloc[0]


class TestExcludeAdultEntertainment:
    def test_production_any_amount(self):
        ba = BusinessActivityExposures(adult_entertainment_production_revenue_pct=0.01)
        df = _single_df(business_activities=ba)
        assert exclude_adult_entertainment(df).iloc[0]

    def test_distribution_at_threshold(self):
        ba = BusinessActivityExposures(adult_entertainment_distribution_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_adult_entertainment(df).iloc[0]

    def test_distribution_below_threshold(self):
        ba = BusinessActivityExposures(adult_entertainment_distribution_revenue_pct=4.9)
        df = _single_df(business_activities=ba)
        assert not exclude_adult_entertainment(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_adult_entertainment(df).iloc[0]


class TestExcludeAlcoholicBeverages:
    def test_production_at_threshold(self):
        ba = BusinessActivityExposures(alcoholic_beverages_production_revenue_pct=5.0)
        df = _single_df(business_activities=ba)
        assert exclude_alcoholic_beverages(df).iloc[0]

    def test_production_below_threshold(self):
        ba = BusinessActivityExposures(alcoholic_beverages_production_revenue_pct=4.9)
        df = _single_df(business_activities=ba)
        assert not exclude_alcoholic_beverages(df).iloc[0]

    def test_retail_at_threshold(self):
        ba = BusinessActivityExposures(alcoholic_beverages_retail_revenue_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_alcoholic_beverages(df).iloc[0]

    def test_related_at_threshold(self):
        ba = BusinessActivityExposures(alcoholic_beverages_related_revenue_pct=10.0)
        df = _single_df(business_activities=ba)
        assert exclude_alcoholic_beverages(df).iloc[0]

    def test_zero_involvement(self):
        df = _single_df()
        assert not exclude_alcoholic_beverages(df).iloc[0]


# ---------------------------------------------------------------------------
# UNGC / GSS filters
# ---------------------------------------------------------------------------


class TestExcludeUNGC:
    def test_non_compliant_excluded(self):
        df = _single_df(ungc_status=UNGCStatus.NON_COMPLIANT)
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_no_coverage_excluded(self):
        df = _single_df(ungc_status=UNGCStatus.NO_COVERAGE)
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_compliant_included(self):
        df = _single_df(ungc_status=UNGCStatus.COMPLIANT)
        assert not exclude_ungc_non_compliant(df).iloc[0]

    def test_watchlist_included(self):
        # Watchlist is NOT excluded at rebalancing (only Non-Compliant is)
        df = _single_df(ungc_status=UNGCStatus.WATCHLIST)
        assert not exclude_ungc_non_compliant(df).iloc[0]


class TestExcludeNoSustainalyticsCoverage:
    def test_no_coverage_excluded(self):
        ba = BusinessActivityExposures(has_sustainalytics_coverage=False)
        df = _single_df(business_activities=ba)
        assert exclude_no_sustainalytics_coverage(df).iloc[0]

    def test_with_coverage_included(self):
        df = _single_df()
        assert not exclude_no_sustainalytics_coverage(df).iloc[0]


# ---------------------------------------------------------------------------
# MSA filter
# ---------------------------------------------------------------------------


class TestExcludeMSAFlagged:
    def test_flagged_excluded(self):
        df = _single_df(msa_flagged=True)
        assert exclude_msa_flagged(df).iloc[0]

    def test_not_flagged_included(self):
        df = _single_df(msa_flagged=False)
        assert not exclude_msa_flagged(df).iloc[0]


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


class TestApplyEligibilityFilters:
    def _universe_with_peers(self, focal_stock: "Stock") -> "pd.DataFrame":
        """
        Return a DataFrame with the focal stock plus two peers in the same
        industry group: one low-ESG peer (will be the bottom quartile) and
        one high-ESG peer. This ensures the focal stock is NOT excluded by
        the ESG quartile screen, so we can isolate the filter under test.

        With scores [low=30, focal=70, high=95]:
          quantile(0.25) = 30 + 0.25*(70-30) = 40 → focal(70) ≥ 40 → passes
        """
        peers = [
            make_stock(
                "PEER_LOW",
                gics_industry_group=focal_stock.gics_industry_group,
                esg_score=30.0,
                market_cap_usd=100_000_000,
            ),
            make_stock(
                "PEER_HIGH",
                gics_industry_group=focal_stock.gics_industry_group,
                esg_score=95.0,
                market_cap_usd=100_000_000,
            ),
        ]
        return make_universe([focal_stock] + peers).to_dataframe()

    def test_clean_stock_passes_all_filters(self):
        stock = make_stock(esg_score=70.0)
        df = self._universe_with_peers(stock)
        eligible, excluded = apply_eligibility_filters(df)
        assert "TEST" in eligible["ticker"].values
        assert "TEST" not in excluded

    def test_tobacco_producer_excluded(self):
        ba = BusinessActivityExposures(tobacco_production_revenue_pct=20.0)
        stock = make_stock(ticker="SMOK", esg_score=70.0, business_activities=ba)
        df = self._universe_with_peers(stock)
        eligible, excluded = apply_eligibility_filters(df)
        assert "SMOK" not in eligible["ticker"].values
        assert excluded["SMOK"] == "Tobacco"

    def test_ungc_non_compliant_excluded(self):
        stock = make_stock(ticker="BAD", esg_score=70.0, ungc_status=UNGCStatus.NON_COMPLIANT)
        df = self._universe_with_peers(stock)
        eligible, excluded = apply_eligibility_filters(df)
        assert "BAD" not in eligible["ticker"].values
        assert "UNGC" in excluded["BAD"]

    def test_msa_flagged_excluded(self):
        stock = make_stock(ticker="MSA", esg_score=70.0, msa_flagged=True)
        df = self._universe_with_peers(stock)
        eligible, excluded = apply_eligibility_filters(df)
        assert "MSA" not in eligible["ticker"].values

    def test_first_exclusion_reason_recorded(self):
        """A stock failing multiple criteria records only the first reason hit."""
        ba = BusinessActivityExposures(tobacco_production_revenue_pct=20.0)
        stock = make_stock(
            ticker="MULTI",
            esg_score=70.0,
            ungc_status=UNGCStatus.NON_COMPLIANT,
            business_activities=ba,
        )
        df = self._universe_with_peers(stock)
        _, excluded = apply_eligibility_filters(df)
        # "Tobacco" comes before "UNGC" in the check order
        assert excluded["MULTI"] == "Tobacco"

    def test_sample_universe_excludes_known_bad_stocks(self, sample_universe_csv):
        """Integration test: known bad companies in the sample CSV are excluded."""
        from indices.sp_carbon_aware.rebalancer import load_universe_from_csv

        universe = load_universe_from_csv(sample_universe_csv)
        df = universe.to_dataframe()
        _, excluded = apply_eligibility_filters(df, universe_type="developed")

        # These tickers are deliberately set up to fail specific screens
        assert "SMOK1" in excluded    # tobacco
        assert "COAL1" in excluded    # thermal coal
        assert "OIL1" in excluded     # oil & gas
        assert "WEAP1" in excluded    # controversial weapons
        assert "GAMB1" in excluded    # gambling
        assert "BEER1" in excluded    # alcoholic beverages
        assert "UNGC1" in excluded    # UNGC non-compliant
        assert "NOSC1" in excluded    # no Sustainalytics coverage
        assert "MSA1" in excluded     # MSA flagged
        assert "NOESC" in excluded    # missing ESG coverage
