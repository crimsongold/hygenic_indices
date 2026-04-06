"""Tests for sp_esg eligibility filters."""

from __future__ import annotations

import pandas as pd
import pytest

from indices.sp_esg.eligibility import (
    apply_eligibility_filters,
    exclude_bottom_quartile_esg,
    exclude_controversial_weapons,
    exclude_missing_esg,
    exclude_msa_flagged,
    exclude_small_arms,
    exclude_thermal_coal,
    exclude_tobacco,
    exclude_ungc_non_compliant,
)
from indices.sp_esg.models import BusinessActivityExposures, UNGCStatus
from .conftest import make_stock, make_universe


def _df(*stocks):
    return make_universe(*stocks).to_dataframe()


class TestExcludeMissingESG:
    def test_no_coverage_flag_excluded(self):
        s = make_stock(has_esg_coverage=False, esg_score=None)
        df = _df(s)
        assert exclude_missing_esg(df).iloc[0]

    def test_null_score_with_coverage_excluded(self):
        s = make_stock(has_esg_coverage=True, esg_score=None)
        df = _df(s)
        assert exclude_missing_esg(df).iloc[0]

    def test_valid_score_retained(self):
        s = make_stock(esg_score=60.0)
        df = _df(s)
        assert not exclude_missing_esg(df).iloc[0]


class TestExcludeBottomQuartileESG:
    def test_bottom_25pct_excluded(self):
        # 4 stocks: scores 10, 40, 70, 90; bottom quartile (<40) → score 10 excluded
        stocks = [
            make_stock(ticker="A", esg_score=10.0),
            make_stock(ticker="B", esg_score=40.0),
            make_stock(ticker="C", esg_score=70.0),
            make_stock(ticker="D", esg_score=90.0),
        ]
        df = _df(*stocks)
        mask = exclude_bottom_quartile_esg(df)
        excluded = df.loc[mask, "ticker"].tolist()
        assert "A" in excluded
        assert "B" not in excluded

    def test_stock_on_threshold_retained(self):
        # 4 equal-score stocks: all at threshold — none strictly below → none excluded
        stocks = [make_stock(ticker=f"T{i}", esg_score=50.0) for i in range(4)]
        df = _df(*stocks)
        assert not exclude_bottom_quartile_esg(df).any()

    def test_industry_group_separation(self):
        # Group A: scores 5, 50, 80, 90 — bottom quartile threshold ≈ 28.75 → AL excluded
        # Group B: scores 75, 80, 82, 85, 90 — threshold ≈ 77.5 → B0 (75) excluded, B1 (80) retained
        # This verifies that each group's exclusion is based on its OWN distribution.
        a_low = make_stock(ticker="AL", esg_score=5.0, gics_industry_group="Group A")
        a_mid = make_stock(ticker="AM", esg_score=50.0, gics_industry_group="Group A")
        a_hi = make_stock(ticker="AH", esg_score=80.0, gics_industry_group="Group A")
        a_vh = make_stock(ticker="AV", esg_score=90.0, gics_industry_group="Group A")
        b0 = make_stock(ticker="B0", esg_score=75.0, gics_industry_group="Group B")
        b1 = make_stock(ticker="B1", esg_score=80.0, gics_industry_group="Group B")
        b2 = make_stock(ticker="B2", esg_score=82.0, gics_industry_group="Group B")
        b3 = make_stock(ticker="B3", esg_score=85.0, gics_industry_group="Group B")
        b4 = make_stock(ticker="B4", esg_score=90.0, gics_industry_group="Group B")
        df = _df(a_low, a_mid, a_hi, a_vh, b0, b1, b2, b3, b4)
        mask = exclude_bottom_quartile_esg(df)
        excluded = df.loc[mask, "ticker"].tolist()
        assert "AL" in excluded   # worst performer in Group A
        assert "B1" not in excluded  # 80 is well above Group B's threshold (~77.5)


class TestExcludeControversialWeapons:
    def test_revenue_triggers_exclusion(self):
        ba = BusinessActivityExposures(controversial_weapons_revenue_pct=1.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_10pct_triggers(self):
        ba = BusinessActivityExposures(controversial_weapons_ownership_pct=10.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_controversial_weapons(df).iloc[0]

    def test_ownership_9pct_retained(self):
        ba = BusinessActivityExposures(controversial_weapons_ownership_pct=9.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert not exclude_controversial_weapons(df).iloc[0]

    def test_zero_involvement_retained(self):
        df = _df(make_stock())
        assert not exclude_controversial_weapons(df).iloc[0]


class TestExcludeTobacco:
    def test_any_production_excluded(self):
        ba = BusinessActivityExposures(tobacco_production_revenue_pct=0.1)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_tobacco(df).iloc[0]

    def test_retail_below_10pct_retained(self):
        ba = BusinessActivityExposures(tobacco_retail_revenue_pct=9.9)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert not exclude_tobacco(df).iloc[0]

    def test_retail_at_10pct_excluded(self):
        ba = BusinessActivityExposures(tobacco_retail_revenue_pct=10.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_tobacco(df).iloc[0]


class TestExcludeThermalCoal:
    def test_extraction_above_5pct_excluded(self):
        ba = BusinessActivityExposures(thermal_coal_extraction_revenue_pct=6.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_thermal_coal(df).iloc[0]

    def test_extraction_at_5pct_retained(self):
        ba = BusinessActivityExposures(thermal_coal_extraction_revenue_pct=5.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert not exclude_thermal_coal(df).iloc[0]

    def test_power_above_25pct_excluded(self):
        ba = BusinessActivityExposures(thermal_coal_power_revenue_pct=26.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_thermal_coal(df).iloc[0]

    def test_power_at_25pct_retained(self):
        ba = BusinessActivityExposures(thermal_coal_power_revenue_pct=25.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert not exclude_thermal_coal(df).iloc[0]


class TestExcludeSmallArms:
    def test_manufacture_at_5pct_excluded(self):
        ba = BusinessActivityExposures(small_arms_manufacture_revenue_pct=5.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_small_arms(df).iloc[0]

    def test_manufacture_below_5pct_retained(self):
        ba = BusinessActivityExposures(small_arms_manufacture_revenue_pct=4.9)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert not exclude_small_arms(df).iloc[0]

    def test_retail_at_10pct_excluded(self):
        ba = BusinessActivityExposures(small_arms_retail_revenue_pct=10.0)
        s = make_stock(business_activities=ba)
        df = _df(s)
        assert exclude_small_arms(df).iloc[0]


class TestExcludeUNGC:
    def test_non_compliant_excluded(self):
        s = make_stock(ungc_status=UNGCStatus.NON_COMPLIANT)
        df = _df(s)
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_no_coverage_excluded(self):
        s = make_stock(ungc_status=UNGCStatus.NO_COVERAGE)
        df = _df(s)
        assert exclude_ungc_non_compliant(df).iloc[0]

    def test_watchlist_retained(self):
        s = make_stock(ungc_status=UNGCStatus.WATCHLIST)
        df = _df(s)
        assert not exclude_ungc_non_compliant(df).iloc[0]

    def test_compliant_retained(self):
        s = make_stock(ungc_status=UNGCStatus.COMPLIANT)
        df = _df(s)
        assert not exclude_ungc_non_compliant(df).iloc[0]


class TestExcludeMSA:
    def test_flagged_excluded(self):
        s = make_stock(msa_flagged=True)
        df = _df(s)
        assert exclude_msa_flagged(df).iloc[0]

    def test_not_flagged_retained(self):
        df = _df(make_stock())
        assert not exclude_msa_flagged(df).iloc[0]


class TestApplyEligibilityFilters:
    def test_clean_stock_passes(self):
        s = make_stock()
        df = _df(s)
        eligible, excluded = apply_eligibility_filters(df)
        assert "TEST" in eligible["ticker"].values
        assert "TEST" not in excluded

    def test_first_failure_logged(self):
        # Stock fails both ESG (missing) and UNGC — should be logged as ESG missing
        s = make_stock(has_esg_coverage=False, esg_score=None, ungc_status=UNGCStatus.NON_COMPLIANT)
        df = _df(s)
        _, excluded = apply_eligibility_filters(df)
        assert excluded.get("TEST") == "Missing ESG score"

    def test_weights_sum_to_one_after_eligibility(self):
        from indices.sp_esg.rebalancer import rebalance
        stocks = [make_stock(ticker=f"T{i}", esg_score=60.0 + i * 5) for i in range(10)]
        u = make_universe(*stocks)
        result = rebalance(u)
        assert sum(result.rebalanced_weights.values()) == pytest.approx(1.0, abs=1e-9)
