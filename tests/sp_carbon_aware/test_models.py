"""
Tests for data models: Stock, IndexUniverse, BusinessActivityExposures.
"""

from __future__ import annotations

import pytest

from indices.sp_carbon_aware.models import BusinessActivityExposures, IndexUniverse, Stock, UNGCStatus
from tests.conftest import make_stock, make_universe


class TestStock:
    def test_float_adjusted_market_cap(self):
        stock = make_stock(market_cap_usd=1_000_000, float_ratio=0.75)
        assert stock.float_adjusted_market_cap == pytest.approx(750_000)

    def test_defaults(self):
        stock = make_stock()
        assert stock.ungc_status == UNGCStatus.COMPLIANT
        assert not stock.msa_flagged
        assert isinstance(stock.business_activities, BusinessActivityExposures)

    def test_esg_score_can_be_none(self):
        stock = make_stock(esg_score=None, has_esg_coverage=False)
        assert stock.esg_score is None
        assert not stock.has_esg_coverage

    def test_carbon_intensity_can_be_none(self):
        stock = make_stock(carbon_intensity=None)
        assert stock.carbon_intensity is None


class TestIndexUniverse:
    def test_underlying_weights_sum_to_one(self, two_stock_universe):
        total = sum(two_stock_universe.underlying_weights.values())
        assert total == pytest.approx(1.0)

    def test_underlying_weights_proportional_to_float_cap(self):
        s1 = make_stock("A", market_cap_usd=300_000_000, float_ratio=1.0)
        s2 = make_stock("B", market_cap_usd=700_000_000, float_ratio=1.0)
        universe = make_universe([s1, s2])
        assert universe.underlying_weights["A"] == pytest.approx(0.30)
        assert universe.underlying_weights["B"] == pytest.approx(0.70)

    def test_float_ratio_applied_in_weights(self):
        # A has twice the market cap but half the float → same float-cap
        s1 = make_stock("A", market_cap_usd=200_000_000, float_ratio=0.5)
        s2 = make_stock("B", market_cap_usd=100_000_000, float_ratio=1.0)
        universe = make_universe([s1, s2])
        assert universe.underlying_weights["A"] == pytest.approx(0.5)
        assert universe.underlying_weights["B"] == pytest.approx(0.5)

    def test_zero_float_cap_raises(self):
        s1 = make_stock("A", market_cap_usd=0, float_ratio=0.0)
        with pytest.raises(ValueError, match="zero"):
            IndexUniverse(stocks=[s1])

    def test_to_dataframe_contains_all_tickers(self, two_stock_universe):
        df = two_stock_universe.to_dataframe()
        assert set(df["ticker"]) == {"A", "B"}

    def test_to_dataframe_weights_match(self, two_stock_universe):
        df = two_stock_universe.to_dataframe().set_index("ticker")
        assert df.loc["A", "underlying_weight"] == pytest.approx(0.6)
        assert df.loc["B", "underlying_weight"] == pytest.approx(0.4)

    def test_explicit_weights_override_calculation(self):
        s1 = make_stock("A")
        s2 = make_stock("B")
        explicit = {"A": 0.3, "B": 0.7}
        universe = IndexUniverse(stocks=[s1, s2], underlying_weights=explicit)
        assert universe.underlying_weights == explicit
