"""Tests for sp_carbon_efficient data models."""

from __future__ import annotations

import pytest

from indices.sp_carbon_efficient.models import IndexUniverse, Stock
from .conftest import make_stock, make_universe


class TestStock:
    def test_float_adjusted_market_cap(self):
        s = make_stock(market_cap_usd=2_000_000, float_ratio=0.5)
        assert s.float_adjusted_market_cap == pytest.approx(1_000_000)

    def test_none_carbon_intensity_allowed(self):
        s = make_stock(carbon_intensity=None)
        assert s.carbon_intensity is None


class TestIndexUniverse:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        assert sum(u.underlying_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_to_dataframe_columns(self):
        u = make_universe(make_stock())
        df = u.to_dataframe()
        for col in ["ticker", "carbon_intensity", "underlying_weight", "gics_industry_group"]:
            assert col in df.columns
