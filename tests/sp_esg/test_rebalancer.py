"""Integration tests for the sp_esg rebalancer pipeline."""

from __future__ import annotations

import pytest

from indices.sp_esg.models import BusinessActivityExposures, UNGCStatus
from indices.sp_esg.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe
from .conftest import make_stock, make_universe


class TestRebalance:
    def test_full_pipeline_weights_sum_to_one(self):
        stocks = [make_stock(ticker=f"T{i}", esg_score=55.0 + i * 4) for i in range(8)]
        u = make_universe(*stocks)
        result = rebalance(u)
        assert sum(result.rebalanced_weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_excluded_not_in_weights(self):
        good = make_stock(ticker="GOOD", esg_score=80.0)
        bad = make_stock(ticker="BAD", ungc_status=UNGCStatus.NON_COMPLIANT, esg_score=80.0)
        u = make_universe(good, bad)
        result = rebalance(u)
        assert "BAD" not in result.rebalanced_weights
        assert "GOOD" in result.rebalanced_weights

    def test_tobacco_producer_excluded(self):
        clean = make_stock(ticker="CLEAN", esg_score=70.0)
        dirty = make_stock(
            ticker="DIRTY",
            esg_score=70.0,
            business_activities=BusinessActivityExposures(tobacco_production_revenue_pct=50.0),
        )
        u = make_universe(clean, dirty)
        result = rebalance(u)
        assert "DIRTY" in result.excluded_tickers
        assert result.excluded_tickers["DIRTY"] == "Tobacco"

    def test_weights_proportional_to_float_cap(self):
        # Two equal-ESG stocks, A has 3× market cap → should get ~3× weight
        a = make_stock(ticker="A", esg_score=70.0, market_cap_usd=3e9)
        b = make_stock(ticker="B", esg_score=70.0, market_cap_usd=1e9)
        u = make_universe(a, b)
        result = rebalance(u)
        assert result.rebalanced_weights["A"] == pytest.approx(0.75, abs=1e-6)
        assert result.rebalanced_weights["B"] == pytest.approx(0.25, abs=1e-6)

    def test_empty_universe_returns_empty(self):
        # Single stock fails every filter
        s = make_stock(has_esg_coverage=False, esg_score=None)
        u = make_universe(s)
        result = rebalance(u)
        assert result.rebalanced_weights == {}

    def test_result_to_dataframe_sorted_by_weight(self):
        stocks = [make_stock(ticker=f"T{i}", esg_score=60.0 + i * 3, market_cap_usd=float(i + 1) * 1e9) for i in range(5)]
        u = make_universe(*stocks)
        result = rebalance(u)
        df = result_to_dataframe(result, u)
        weights = df["rebalanced_weight"].tolist()
        assert weights == sorted(weights, reverse=True)


class TestLoadUniverseFromCSV:
    def test_loads_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        assert len(u.stocks) > 0
        assert abs(sum(u.underlying_weights.values()) - 1.0) < 1e-9

    def test_rebalance_sample_csv(self, sample_universe_csv):
        u = load_universe_from_csv(sample_universe_csv)
        result = rebalance(u)
        assert len(result.eligible_tickers) > 0
        assert sum(result.rebalanced_weights.values()) == pytest.approx(1.0, abs=1e-9)
