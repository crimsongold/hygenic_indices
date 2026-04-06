"""
Integration tests for the full rebalancing pipeline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.sp_carbon_aware.models import UNGCStatus
from indices.sp_carbon_aware.rebalancer import (
    load_universe_from_csv,
    rebalance,
    result_to_dataframe,
)
from tests.conftest import make_stock, make_universe


# ---------------------------------------------------------------------------
# load_universe_from_csv
# ---------------------------------------------------------------------------


class TestLoadUniverse:
    def test_loads_sample_csv(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        assert len(universe.stocks) > 0

    def test_underlying_weights_sum_to_one(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        total = sum(universe.underlying_weights.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_missing_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("ticker,company_name\nA,Alpha\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_universe_from_csv(csv)

    def test_carbon_intensity_none_for_missing(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        missing_ci = [s for s in universe.stocks if s.carbon_intensity is None]
        # At least some stocks have missing CI in the sample
        assert len(missing_ci) > 0

    def test_ungc_status_parsed(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        statuses = {s.ungc_status for s in universe.stocks}
        # Sample data includes Non-Compliant, Compliant
        assert UNGCStatus.COMPLIANT in statuses


# ---------------------------------------------------------------------------
# rebalance — full pipeline
# ---------------------------------------------------------------------------


class TestRebalance:
    def test_weights_sum_to_one(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        total = sum(result.optimized_weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_excluded_tickers_not_in_result(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        for ticker in result.excluded_tickers:
            assert ticker not in result.optimized_weights

    def test_waci_lower_than_underlying(self, sample_universe_csv):
        """
        The primary objective is to reduce weighted average carbon intensity.
        The optimized WACI should be <= the underlying WACI.
        """
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        assert result.weighted_avg_carbon_intensity <= (
            result.underlying_weighted_avg_carbon_intensity + 1e-4
        )

    def test_no_weight_below_min_threshold(self, sample_universe_csv):
        """All optimized weights must be >= 1bps (or zero for excluded stocks)."""
        from indices.sp_carbon_aware.optimization import MIN_STOCK_WEIGHT

        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        for ticker, w in result.optimized_weights.items():
            assert w >= MIN_STOCK_WEIGHT - 1e-9, f"{ticker} weight {w} below threshold"

    def test_solver_status_reported(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        assert result.solver_status  # non-empty string

    def test_emerging_variant(self, sample_universe_csv):
        """Emerging universe type runs without error and returns weights."""
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe, universe_type="emerging")
        assert sum(result.optimized_weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_all_excluded_returns_empty_weights(self):
        """If every stock fails eligibility the result has no weights."""
        from indices.sp_carbon_aware.models import BusinessActivityExposures

        # Create stocks that all fail tobacco screen
        ba = BusinessActivityExposures(tobacco_production_revenue_pct=50.0)
        stocks = [
            make_stock(f"T{i}", business_activities=ba, market_cap_usd=(i + 1) * 1e8)
            for i in range(3)
        ]
        universe = make_universe(stocks)
        result = rebalance(universe)
        assert result.optimized_weights == {} or sum(result.optimized_weights.values()) < 1e-9

    def test_single_eligible_stock_gets_full_weight(self):
        """With only one eligible stock it must receive 100% weight.

        We use three stocks in the same industry group so the ESG quartile filter
        has a proper distribution to work with. DIRTY is excluded by the tobacco
        screen; LOW_ESG is excluded by the ESG quartile screen; CLEAN passes both.
        """
        from indices.sp_carbon_aware.models import BusinessActivityExposures

        ba_bad = BusinessActivityExposures(tobacco_production_revenue_pct=50.0)
        clean = make_stock("CLEAN", esg_score=80.0, market_cap_usd=500_000_000, carbon_intensity=30.0)
        dirty = make_stock(
            "DIRTY", esg_score=70.0, market_cap_usd=300_000_000, business_activities=ba_bad
        )
        # Third peer with a much lower ESG score so CLEAN is clearly NOT in the bottom quartile
        low_esg = make_stock("LOWQ", esg_score=5.0, market_cap_usd=200_000_000, carbon_intensity=200.0)
        universe = make_universe([clean, dirty, low_esg])
        result = rebalance(universe)
        assert "CLEAN" in result.optimized_weights
        assert result.optimized_weights.get("CLEAN", 0) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# result_to_dataframe
# ---------------------------------------------------------------------------


class TestResultToDataframe:
    def test_returns_dataframe_with_expected_columns(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        df = result_to_dataframe(result, universe)
        for col in ["ticker", "optimized_weight", "underlying_weight", "active_weight"]:
            assert col in df.columns

    def test_active_weight_is_diff(self, sample_universe_csv):
        universe = load_universe_from_csv(sample_universe_csv)
        result = rebalance(universe)
        df = result_to_dataframe(result, universe).set_index("ticker")
        for ticker, row in df.iterrows():
            expected = row["optimized_weight"] - row["underlying_weight"]
            # result_to_dataframe rounds components independently, so rounding error
            # between active_weight and (opt - underlying) can be up to 2e-6
            assert row["active_weight"] == pytest.approx(expected, abs=2e-6)
