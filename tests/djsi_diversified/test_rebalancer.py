"""Integration tests for the DJSI Diversified rebalancer."""

from __future__ import annotations

import pytest

from indices.djsi_diversified.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe
from .conftest import make_stock, make_universe


class TestRebalance:
    def test_weights_sum_to_one(self):
        stocks = [
            make_stock(ticker=f"T{i}", esg_score=float(95 - i), market_cap_usd=1e9)
            for i in range(20)
        ]
        u = make_universe(*stocks)
        result = rebalance(u)
        if result.rebalanced_weights:
            assert sum(result.rebalanced_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_excluded_stocks_not_in_weights(self):
        clean = make_stock(ticker="CLEAN", esg_score=90.0)
        tobacco = make_stock(ticker="TOBACCO", esg_score=85.0, tobacco_production_revenue_pct=5.0)
        u = make_universe(clean, tobacco)
        result = rebalance(u)
        assert "TOBACCO" not in result.rebalanced_weights
        assert "TOBACCO" in result.excluded_tickers

    def test_weight_cap_applied(self):
        # One large stock + many smaller ones; large stock should be capped
        stocks = [
            make_stock(ticker="BIG", esg_score=99.0, market_cap_usd=5e10),
        ] + [
            make_stock(ticker=f"S{i}", esg_score=float(95 - i), market_cap_usd=1e9)
            for i in range(30)
        ]
        u = make_universe(*stocks)
        result = rebalance(u, max_stock_weight=0.10)
        if "BIG" in result.rebalanced_weights:
            assert result.rebalanced_weights["BIG"] <= 0.10 + 1e-9

    def test_result_dataframe_sorted_by_weight(self):
        stocks = [
            make_stock(ticker=f"T{i}", esg_score=float(90 - i * 5), market_cap_usd=float(i + 1) * 1e9)
            for i in range(6)
        ]
        u = make_universe(*stocks)
        result = rebalance(u)
        df = result_to_dataframe(result, u)
        weights = df["rebalanced_weight"].tolist()
        assert weights == sorted(weights, reverse=True)


class TestLoadUniverseFromCSV:
    def test_loads_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        assert len(u.stocks) > 0

    def test_rebalance_from_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        result = rebalance(u)
        if result.rebalanced_weights:
            assert sum(result.rebalanced_weights.values()) == pytest.approx(1.0, abs=1e-9)
        assert len(result.excluded_tickers) > 0
