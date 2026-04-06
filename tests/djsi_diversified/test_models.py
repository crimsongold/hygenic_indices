"""Tests for DJSI Diversified data models."""

from __future__ import annotations

import pytest

from indices.djsi_diversified.models import UNGCStatus
from .conftest import make_stock, make_universe


class TestStock:
    def test_float_adjusted_market_cap(self):
        s = make_stock(market_cap_usd=1e9, float_ratio=0.75)
        assert s.float_adjusted_market_cap == pytest.approx(7.5e8)

    def test_default_business_activities_are_zero(self):
        s = make_stock()
        ba = s.business_activities
        assert ba.controversial_weapons_revenue_pct == 0.0
        assert ba.tobacco_production_revenue_pct == 0.0
        assert ba.alcohol_production_revenue_pct == 0.0

    def test_ungc_status_enum(self):
        assert UNGCStatus.NON_COMPLIANT.value == "Non-Compliant"
        assert UNGCStatus.COMPLIANT.value == "Compliant"


class TestIndexUniverse:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        assert sum(u.underlying_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_to_dataframe_has_all_columns(self):
        s = make_stock()
        u = make_universe(s)
        df = u.to_dataframe()
        expected_cols = {
            "ticker", "company_name", "country", "region", "gics_sector",
            "gics_industry_group", "market_cap_usd", "float_ratio",
            "underlying_weight", "float_adjusted_market_cap",
            "esg_score", "has_esg_coverage", "ungc_status", "msa_flagged",
            "controversial_weapons_revenue_pct", "controversial_weapons_ownership_pct",
            "tobacco_production_revenue_pct",
            "adult_entertainment_production_revenue_pct", "adult_entertainment_retail_revenue_pct",
            "alcohol_production_revenue_pct",
            "gambling_operations_revenue_pct", "gambling_equipment_revenue_pct",
            "military_integral_weapons_revenue_pct", "military_weapon_related_revenue_pct",
            "small_arms_civilian_production_revenue_pct", "small_arms_key_components_revenue_pct",
            "small_arms_noncivilian_production_revenue_pct", "small_arms_retail_revenue_pct",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_to_dataframe_row_count(self):
        stocks = [make_stock(ticker=f"T{i}") for i in range(3)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        assert len(df) == 3
