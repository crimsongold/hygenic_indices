"""Tests for sp_esg data models."""

from __future__ import annotations

import pytest

from indices.sp_esg.models import BusinessActivityExposures, IndexUniverse, Stock, UNGCStatus
from .conftest import make_stock, make_universe


class TestStock:
    def test_float_adjusted_market_cap(self):
        s = make_stock(market_cap_usd=1_000_000, float_ratio=0.8)
        assert s.float_adjusted_market_cap == pytest.approx(800_000)

    def test_defaults_are_clean(self):
        s = make_stock()
        assert s.ungc_status == UNGCStatus.COMPLIANT
        assert not s.msa_flagged
        assert s.business_activities.controversial_weapons_revenue_pct == 0.0

    def test_optional_esg_score_none(self):
        s = make_stock(esg_score=None, has_esg_coverage=False)
        assert s.esg_score is None


class TestIndexUniverse:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        assert sum(u.underlying_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_weights_proportional_to_float_cap(self):
        a = make_stock(ticker="A", market_cap_usd=3e9, float_ratio=1.0)
        b = make_stock(ticker="B", market_cap_usd=1e9, float_ratio=1.0)
        u = make_universe(a, b)
        assert u.underlying_weights["A"] == pytest.approx(0.75)
        assert u.underlying_weights["B"] == pytest.approx(0.25)

    def test_float_ratio_applied(self):
        a = make_stock(ticker="A", market_cap_usd=2e9, float_ratio=0.5)
        b = make_stock(ticker="B", market_cap_usd=2e9, float_ratio=1.0)
        u = make_universe(a, b)
        # A float-cap = 1e9, B float-cap = 2e9 → weights 1/3 and 2/3
        assert u.underlying_weights["A"] == pytest.approx(1 / 3)
        assert u.underlying_weights["B"] == pytest.approx(2 / 3)

    def test_to_dataframe_has_all_columns(self):
        u = make_universe(make_stock())
        df = u.to_dataframe()
        for col in ["ticker", "esg_score", "ungc_status", "msa_flagged",
                    "tobacco_production_revenue_pct", "thermal_coal_extraction_revenue_pct"]:
            assert col in df.columns
