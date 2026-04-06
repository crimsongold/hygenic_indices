"""Tests for PACT data models."""

from __future__ import annotations

import pytest

from indices.sp_pact.models import Variant, UNGCStatus
from .conftest import make_stock, make_universe


class TestStock:
    def test_float_adjusted_market_cap(self):
        s = make_stock(market_cap_usd=1e9, float_ratio=0.6)
        assert s.float_adjusted_market_cap == pytest.approx(6e8)

    def test_total_carbon_intensity(self):
        s = make_stock(scope_1_2_carbon_intensity=100.0, scope_3_carbon_intensity=50.0)
        assert s.total_carbon_intensity == pytest.approx(150.0)

    def test_total_carbon_intensity_no_scope3(self):
        s = make_stock(scope_1_2_carbon_intensity=100.0, scope_3_carbon_intensity=None)
        assert s.total_carbon_intensity == pytest.approx(100.0)

    def test_total_carbon_intensity_no_coverage(self):
        s = make_stock(scope_1_2_carbon_intensity=None)
        assert s.total_carbon_intensity is None

    def test_variant_enum(self):
        assert Variant.CTB.value == "ctb"
        assert Variant.PAB.value == "pab"


class TestIndexUniverse:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        assert sum(u.underlying_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_to_dataframe_has_carbon_columns(self):
        s = make_stock()
        u = make_universe(s)
        df = u.to_dataframe()
        assert "scope_1_2_carbon_intensity" in df.columns
        assert "total_carbon_intensity" in df.columns
        assert "has_sbti_target" in df.columns

    def test_to_dataframe_row_count(self):
        stocks = [make_stock(ticker=f"T{i}") for i in range(4)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        assert len(df) == 4
