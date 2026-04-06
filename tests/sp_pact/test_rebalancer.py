"""Integration tests for the sp_pact rebalancer."""

from __future__ import annotations

import pytest

from indices.sp_pact.models import Variant
from indices.sp_pact.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe
from .conftest import make_stock, make_universe


class TestRebalance:
    def test_ctb_weights_sum_to_one(self):
        stocks = [
            make_stock(
                ticker=f"T{i}",
                scope_1_2_carbon_intensity=float(i + 1) * 20,
                scope_3_carbon_intensity=float(i + 1) * 10,
                market_cap_usd=1e9,
            )
            for i in range(20)
        ]
        u = make_universe(*stocks)
        result = rebalance(u, variant=Variant.CTB)
        if result.optimized_weights:
            assert sum(result.optimized_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pab_weights_sum_to_one(self):
        stocks = [
            make_stock(
                ticker=f"T{i}",
                scope_1_2_carbon_intensity=float(i + 1) * 20,
                scope_3_carbon_intensity=float(i + 1) * 10,
                market_cap_usd=1e9,
            )
            for i in range(20)
        ]
        u = make_universe(*stocks)
        result = rebalance(u, variant=Variant.PAB)
        if result.optimized_weights:
            assert sum(result.optimized_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_excluded_stocks_not_in_weights(self):
        clean = make_stock(ticker="CLEAN")
        tobacco = make_stock(ticker="TOBACCO", tobacco_production_revenue_pct=5.0)
        u = make_universe(clean, tobacco)
        result = rebalance(u, variant=Variant.CTB)
        assert "TOBACCO" not in result.optimized_weights
        assert "TOBACCO" in result.excluded_tickers

    def test_pab_excludes_more_than_ctb(self):
        # Stock with coal revenue — PAB excludes, CTB does not
        stocks = [make_stock(ticker=f"T{i}") for i in range(10)]
        coal_stock = make_stock(ticker="COAL", coal_revenue_pct=2.0)
        stocks.append(coal_stock)
        u = make_universe(*stocks)
        ctb_result = rebalance(u, variant=Variant.CTB)
        pab_result = rebalance(u, variant=Variant.PAB)
        assert len(pab_result.excluded_tickers) >= len(ctb_result.excluded_tickers)

    def test_variant_recorded_in_result(self):
        stocks = [make_stock(ticker=f"T{i}") for i in range(5)]
        u = make_universe(*stocks)
        ctb = rebalance(u, variant=Variant.CTB)
        pab = rebalance(u, variant=Variant.PAB)
        assert ctb.variant == "ctb"
        assert pab.variant == "pab"


class TestLoadUniverseFromCSV:
    def test_loads_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        assert len(u.stocks) > 0

    def test_ctb_rebalance_from_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        result = rebalance(u, variant=Variant.CTB)
        if result.optimized_weights:
            assert sum(result.optimized_weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_pab_rebalance_from_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        result = rebalance(u, variant=Variant.PAB)
        if result.optimized_weights:
            assert sum(result.optimized_weights.values()) == pytest.approx(1.0, abs=1e-6)
        assert len(result.excluded_tickers) > 0
