"""Tests for the carbon efficiency tilt weighting logic."""

from __future__ import annotations

import math

import pytest

from indices.sp_carbon_efficient.weighting import (
    apply_tilt,
    compute_carbon_efficiency_factors,
    weighted_avg_carbon_intensity,
)
from .conftest import make_stock, make_universe


class TestComputeCEF:
    def test_low_emitter_gets_cef_above_one(self):
        # Low CI relative to group → positive z-score flipped → CEF > 1
        low = make_stock(ticker="L", carbon_intensity=10.0)
        high = make_stock(ticker="H", carbon_intensity=90.0)
        u = make_universe(low, high)
        df = u.to_dataframe()
        cef = compute_carbon_efficiency_factors(df, tilt_lambda=0.5)
        assert cef.iloc[0] > 1.0  # low emitter
        assert cef.iloc[1] < 1.0  # high emitter

    def test_no_carbon_data_gets_neutral_cef(self):
        s_no_data = make_stock(ticker="X", carbon_intensity=None)
        s_data = make_stock(ticker="Y", carbon_intensity=50.0)
        u = make_universe(s_no_data, s_data)
        df = u.to_dataframe()
        cef = compute_carbon_efficiency_factors(df)
        # X has no data → neutral
        assert cef.iloc[0] == pytest.approx(1.0)

    def test_single_stock_group_gets_neutral_cef(self):
        s = make_stock(ticker="SOLO", carbon_intensity=100.0, gics_industry_group="Unique Sector")
        u = make_universe(s)
        df = u.to_dataframe()
        cef = compute_carbon_efficiency_factors(df)
        assert cef.iloc[0] == pytest.approx(1.0)

    def test_equal_ci_in_group_gets_neutral_cef(self):
        # All stocks with same CI → std = 0 → all CEF = 1
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=50.0) for i in range(4)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        cef = compute_carbon_efficiency_factors(df)
        for v in cef:
            assert v == pytest.approx(1.0)

    def test_lambda_zero_gives_uniform_cef(self):
        low = make_stock(ticker="L", carbon_intensity=10.0)
        high = make_stock(ticker="H", carbon_intensity=100.0)
        u = make_universe(low, high)
        df = u.to_dataframe()
        cef = compute_carbon_efficiency_factors(df, tilt_lambda=0.0)
        # exp(0) = 1 for all stocks
        for v in cef:
            assert v == pytest.approx(1.0)


class TestApplyTilt:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=float(i + 1) * 20) for i in range(5)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        tilted = apply_tilt(df, u.underlying_weights)
        assert sum(tilted.values()) == pytest.approx(1.0, abs=1e-9)

    def test_low_emitter_gets_higher_weight_than_benchmark(self):
        # Two equal market-cap stocks; low CI should get weight > 0.5
        low = make_stock(ticker="L", carbon_intensity=10.0, market_cap_usd=1e9)
        high = make_stock(ticker="H", carbon_intensity=90.0, market_cap_usd=1e9)
        u = make_universe(low, high)
        df = u.to_dataframe()
        tilted = apply_tilt(df, u.underlying_weights)
        assert tilted["L"] > 0.5
        assert tilted["H"] < 0.5

    def test_lambda_zero_preserves_underlying_weights(self):
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=float(i + 1) * 20) for i in range(4)]
        u = make_universe(*stocks)
        df = u.to_dataframe()
        tilted = apply_tilt(df, u.underlying_weights, tilt_lambda=0.0)
        for ticker, w in u.underlying_weights.items():
            assert tilted[ticker] == pytest.approx(w, abs=1e-9)


class TestWACI:
    def test_waci_calculated_correctly(self):
        a = make_stock(ticker="A", carbon_intensity=100.0, market_cap_usd=1e9)
        b = make_stock(ticker="B", carbon_intensity=200.0, market_cap_usd=1e9)
        u = make_universe(a, b)
        df = u.to_dataframe()
        # Equal weights (0.5 each) → WACI = 0.5*100 + 0.5*200 = 150
        assert weighted_avg_carbon_intensity(df, {"A": 0.5, "B": 0.5}) == pytest.approx(150.0)

    def test_waci_skips_missing_ci(self):
        a = make_stock(ticker="A", carbon_intensity=None)
        b = make_stock(ticker="B", carbon_intensity=100.0)
        u = make_universe(a, b)
        df = u.to_dataframe()
        # Only B contributes → WACI = B's CI (weight-normalised over B only)
        result = weighted_avg_carbon_intensity(df, {"A": 0.5, "B": 0.5})
        # B has CI=100 and weight=0.5; total_w for CI stocks = 0.5; waci = 0.5*100/0.5
        assert result == pytest.approx(100.0)
