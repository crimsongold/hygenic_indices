"""Integration tests for the sp_carbon_efficient rebalancer."""

from __future__ import annotations

import pytest

from indices.sp_carbon_efficient.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe
from .conftest import make_stock, make_universe


class TestRebalance:
    def test_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=float(i + 1) * 15) for i in range(6)]
        u = make_universe(*stocks)
        result = rebalance(u)
        assert sum(result.tilted_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_tilt_reduces_waci(self):
        # Mix of low and high emitters; tilt should reduce WACI vs. benchmark
        low = make_stock(ticker="L1", carbon_intensity=10.0, market_cap_usd=1e9)
        low2 = make_stock(ticker="L2", carbon_intensity=15.0, market_cap_usd=1e9)
        high = make_stock(ticker="H1", carbon_intensity=300.0, market_cap_usd=1e9)
        high2 = make_stock(ticker="H2", carbon_intensity=400.0, market_cap_usd=1e9)
        u = make_universe(low, low2, high, high2)
        result = rebalance(u)
        assert result.weighted_avg_carbon_intensity < result.underlying_waci
        assert result.waci_reduction_pct > 0

    def test_lambda_zero_unchanged_waci(self):
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=float(i + 1) * 30) for i in range(4)]
        u = make_universe(*stocks)
        result = rebalance(u, tilt_lambda=0.0)
        # Lambda=0 → all CEF=1 → weights unchanged → WACI = underlying WACI
        assert result.waci_reduction_pct == pytest.approx(0.0, abs=1e-6)

    def test_result_dataframe_sorted_by_weight(self):
        stocks = [make_stock(ticker=f"T{i}", carbon_intensity=float(i + 1) * 20, market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        result = rebalance(u)
        df = result_to_dataframe(result, u)
        weights = df["tilted_weight"].tolist()
        assert weights == sorted(weights, reverse=True)


class TestLoadUniverseFromCSV:
    def test_loads_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        assert len(u.stocks) > 0

    def test_rebalance_from_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        result = rebalance(u)
        assert sum(result.tilted_weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert result.waci_reduction_pct > 0
