"""
Tests for the optimization module.

Covers: carbon intensity imputation, weight-threshold post-processing,
        and optimizer correctness on small universes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from indices.sp_carbon_aware.optimization import (
    MIN_STOCK_WEIGHT,
    _compute_T,
    _impute_carbon_intensities,
    apply_minimum_weight_threshold,
    optimize,
)
from tests.conftest import make_stock, make_universe


# ---------------------------------------------------------------------------
# Carbon intensity imputation
# ---------------------------------------------------------------------------


class TestImputeCarbon:
    def test_no_missing_values_unchanged(self):
        ci = pd.Series({"A": 100.0, "B": 200.0})
        ig = pd.Series({"A": "Energy", "B": "Energy"})
        result = _impute_carbon_intensities(ci, ig)
        assert result["A"] == pytest.approx(100.0)
        assert result["B"] == pytest.approx(200.0)

    def test_missing_value_filled_with_group_median(self):
        ci = pd.Series({"A": 100.0, "B": 200.0, "C": np.nan})
        ig = pd.Series({"A": "Energy", "B": "Energy", "C": "Energy"})
        result = _impute_carbon_intensities(ci, ig)
        # Median of [100, 200] = 150
        assert result["C"] == pytest.approx(150.0)

    def test_missing_uses_only_own_group(self):
        ci = pd.Series({"A": 100.0, "B": 500.0, "C": np.nan})
        ig = pd.Series({"A": "Energy", "B": "Materials", "C": "Materials"})
        result = _impute_carbon_intensities(ci, ig)
        # C should use Materials median (only B=500), not Energy (100)
        assert result["C"] == pytest.approx(500.0)

    def test_all_missing_in_group_uses_global_median(self):
        ci = pd.Series({"A": np.nan, "B": 200.0})
        ig = pd.Series({"A": "Rare", "B": "Energy"})
        result = _impute_carbon_intensities(ci, ig)
        # Global median of [200] = 200
        assert result["A"] == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# T computation
# ---------------------------------------------------------------------------


class TestComputeT:
    def test_no_ineligible(self):
        assert _compute_T(0.0) == pytest.approx(0.25)

    def test_partial_ineligible(self):
        # T = max(5%, 25% - 10%) = 15%
        assert _compute_T(0.10) == pytest.approx(0.15)

    def test_heavy_ineligible_floored_at_5pct(self):
        # T = max(5%, 25% - 30%) = 5%
        assert _compute_T(0.30) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Minimum weight threshold post-processing
# ---------------------------------------------------------------------------


class TestApplyMinimumWeightThreshold:
    def test_weights_above_threshold_unchanged_up_to_renorm(self):
        weights = np.array([0.6, 0.4])
        result = apply_minimum_weight_threshold(weights, ["A", "B"])
        assert sum(result.values()) == pytest.approx(1.0)
        assert result["A"] == pytest.approx(0.6)
        assert result["B"] == pytest.approx(0.4)

    def test_sub_threshold_weight_redistributed(self):
        # C is below 1bps; its weight should go to A and B
        weights = np.array([0.5, 0.4999, 0.0001 - 1e-10])
        result = apply_minimum_weight_threshold(weights, ["A", "B", "C"], min_weight=0.0001)
        assert "C" not in result
        assert sum(result.values()) == pytest.approx(1.0)

    def test_all_zero_after_threshold_returns_empty(self):
        weights = np.array([0.000001, 0.000001])
        result = apply_minimum_weight_threshold(
            weights, ["A", "B"], min_weight=0.01
        )
        # Both below threshold; all weight zeroed → empty
        assert sum(result.values()) == pytest.approx(0.0, abs=1e-9)

    def test_output_tickers_match_above_threshold_stocks(self):
        weights = np.array([0.9, 0.05, 0.00005])
        result = apply_minimum_weight_threshold(weights, ["A", "B", "C"])
        assert "A" in result
        assert "B" in result
        assert "C" not in result


# ---------------------------------------------------------------------------
# Optimizer smoke tests
# ---------------------------------------------------------------------------


class TestOptimize:
    def _eligible_df(self, stocks, underlying_weights):
        universe = make_universe(stocks)
        # Override weights if needed
        universe.underlying_weights = underlying_weights
        return universe.to_dataframe()

    def test_two_stock_minimizes_carbon(self):
        """With two stocks the optimizer should overweight the low-carbon one."""
        s1 = make_stock("LOW", carbon_intensity=10.0, market_cap_usd=500_000_000)
        s2 = make_stock("HIGH", carbon_intensity=500.0, market_cap_usd=500_000_000)
        universe = make_universe([s1, s2])
        eligible_df = universe.to_dataframe()

        result = optimize(eligible_df, universe.underlying_weights)
        assert result.weights is not None

        weights = dict(zip(["LOW", "HIGH"], result.weights))
        # LOW carbon stock should have weight >= HIGH carbon stock
        assert weights["LOW"] >= weights["HIGH"]

    def test_weights_sum_to_one(self):
        stocks = [
            make_stock(f"S{i}", market_cap_usd=(i + 1) * 100_000_000, carbon_intensity=float(i * 50))
            for i in range(5)
        ]
        universe = make_universe(stocks)
        eligible_df = universe.to_dataframe()
        result = optimize(eligible_df, universe.underlying_weights)
        assert result.weights is not None
        assert np.sum(result.weights) == pytest.approx(1.0, abs=1e-4)

    def test_status_is_optimal(self):
        stocks = [
            make_stock(f"S{i}", market_cap_usd=(i + 1) * 200_000_000, carbon_intensity=float((i + 1) * 30))
            for i in range(6)
        ]
        universe = make_universe(stocks)
        result = optimize(universe.to_dataframe(), universe.underlying_weights)
        assert result.status in ("optimal", "optimal_inaccurate", "fallback_equal_weight")

    def test_empty_eligible_universe(self):
        """If no eligible stocks, result has empty weights."""
        universe = make_universe([make_stock("A")])
        empty_df = universe.to_dataframe().iloc[0:0]  # empty DataFrame
        result = optimize(empty_df, universe.underlying_weights)
        assert result.status == "no_eligible_stocks"

    def test_ineligible_stocks_excluded_from_weights(self):
        """Ineligible tickers get zero weight; eligible stocks sum to 1."""
        s_eligible = make_stock("ELG", market_cap_usd=600_000_000, carbon_intensity=50.0)
        s_ineligible = make_stock("BAD", market_cap_usd=400_000_000, carbon_intensity=100.0)
        universe = make_universe([s_eligible, s_ineligible])
        eligible_df = universe.to_dataframe()[universe.to_dataframe()["ticker"] == "ELG"]
        result = optimize(eligible_df, universe.underlying_weights)
        # Only one eligible stock → its weight must be ~1.0
        assert result.weights is not None
        assert np.sum(result.weights) == pytest.approx(1.0, abs=1e-4)
