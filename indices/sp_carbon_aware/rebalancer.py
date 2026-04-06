"""
Rebalancer for the S&P Carbon Aware Index Series.

Orchestrates the full rebalancing pipeline:
  1. Load the universe from a CSV file into typed Pydantic models.
  2. Apply eligibility filters to identify which stocks may enter the index.
  3. Run the carbon-intensity minimization optimizer on the eligible universe.
  4. Post-process to enforce the minimum stock weight threshold (1 bps).
  5. Compute diagnostics (WACI before and after optimization).
  6. Return a RebalanceResult with final weights and a full exclusion log.

Ref: "S&P Carbon Aware Index Series Methodology" (March 2026)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from indices.sp_carbon_aware.eligibility import apply_eligibility_filters
from indices.sp_carbon_aware.models import (
    BusinessActivityExposures,
    IndexUniverse,
    RebalanceResult,
    Stock,
    UNGCStatus,
)
from indices.sp_carbon_aware.optimization import (
    apply_minimum_weight_threshold,
    optimize,
)


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

# All business-activity percentage columns that map directly onto fields of
# BusinessActivityExposures. The list drives both the CSV-to-model mapping
# in load_universe_from_csv and column validation in _validate_columns.
_BUSINESS_ACTIVITY_COLUMNS = [
    "controversial_weapons_tailor_made_essential_pct",
    "controversial_weapons_tailor_made_essential_ownership_pct",
    "controversial_weapons_non_tailor_made_pct",
    "controversial_weapons_non_tailor_made_ownership_pct",
    "tobacco_production_revenue_pct",
    "tobacco_related_revenue_pct",
    "tobacco_retail_revenue_pct",
    "thermal_coal_extraction_revenue_pct",
    "thermal_coal_power_generation_revenue_pct",
    "oil_sands_extraction_revenue_pct",
    "shale_energy_extraction_revenue_pct",
    "arctic_oil_gas_extraction_revenue_pct",
    "oil_gas_production_revenue_pct",
    "oil_gas_generation_revenue_pct",
    "oil_gas_supporting_revenue_pct",
    "gambling_operations_revenue_pct",
    "gambling_equipment_revenue_pct",
    "gambling_supporting_revenue_pct",
    "adult_entertainment_production_revenue_pct",
    "adult_entertainment_distribution_revenue_pct",
    "alcoholic_beverages_production_revenue_pct",
    "alcoholic_beverages_retail_revenue_pct",
    "alcoholic_beverages_related_revenue_pct",
    "has_sustainalytics_coverage",  # boolean flag, handled separately below
]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_universe_from_csv(csv_path: str | Path) -> IndexUniverse:
    """
    Parse a universe CSV into an IndexUniverse of typed Stock models.

    The CSV must contain one row per security. The underlying index weights are
    derived from the market_cap_usd and float_ratio columns — you do not need
    to supply pre-computed weights.

    Type coercion:
        - esg_score and carbon_intensity may be blank (NaN in pandas) to
          represent missing data; they become None in the Stock model.
        - ungc_status is parsed as a UNGCStatus enum value; unrecognised strings
          default to UNGCStatus.NO_COVERAGE (treat as uncovered).
        - has_sustainalytics_coverage and msa_flagged are stored as 0/1 integers
          in the CSV and cast to bool here.
        - All business activity percentage columns are cast to float.

    Ref: §Eligibility Criteria, §Index Construction
    """
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    stocks: list[Stock] = []
    for _, row in df.iterrows():
        # Build the BusinessActivityExposures model.
        # All percentage columns are float; has_sustainalytics_coverage is bool.
        ba_kwargs = {
            col: float(row[col]) if col != "has_sustainalytics_coverage"
            else bool(row[col])
            for col in _BUSINESS_ACTIVITY_COLUMNS
            if col in row.index
        }
        ba = BusinessActivityExposures(**ba_kwargs)

        # Optional fields: blank CSV cells become None (not 0.0 or NaN),
        # so that downstream code can distinguish "no data" from "zero".
        esg_score = float(row["esg_score"]) if pd.notna(row["esg_score"]) else None
        carbon_intensity = (
            float(row["carbon_intensity"]) if pd.notna(row["carbon_intensity"]) else None
        )

        # Parse UNGC status; fall back to NO_COVERAGE for any unrecognised value
        ungc_status_str = (
            str(row["ungc_status"]) if pd.notna(row["ungc_status"]) else "No Coverage"
        )
        try:
            ungc_status = UNGCStatus(ungc_status_str)
        except ValueError:
            # Unrecognised string → treat as if the company has no GSS coverage
            ungc_status = UNGCStatus.NO_COVERAGE

        stock = Stock(
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            country=str(row["country"]),
            gics_sector=str(row["gics_sector"]),
            gics_industry_group=str(row["gics_industry_group"]),
            market_cap_usd=float(row["market_cap_usd"]),
            float_ratio=float(row["float_ratio"]),
            esg_score=esg_score,
            has_esg_coverage=bool(row.get("has_esg_coverage", True)),
            carbon_intensity=carbon_intensity,
            ungc_status=ungc_status,
            business_activities=ba,
            msa_flagged=bool(row.get("msa_flagged", False)),
        )
        stocks.append(stock)

    # IndexUniverse automatically computes float-adjusted market-cap weights
    # for each stock if no explicit underlying_weights dict is provided.
    return IndexUniverse(stocks=stocks)


def _validate_columns(df: pd.DataFrame) -> None:
    """
    Raise a ValueError listing any required columns missing from the CSV.

    Called before any row iteration so that errors surface early with a
    clear message rather than as a KeyError deep in the parsing loop.
    """
    required = {
        "ticker", "company_name", "country", "gics_sector",
        "gics_industry_group", "market_cap_usd", "float_ratio",
        "esg_score", "has_esg_coverage", "carbon_intensity",
        "ungc_status", "msa_flagged",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Rebalancing pipeline
# ---------------------------------------------------------------------------


def rebalance(
    universe: IndexUniverse,
    universe_type: str = "developed",
) -> RebalanceResult:
    """
    Execute the full S&P Carbon Aware rebalancing pipeline and return results.

    Pipeline steps:

      1. Flatten the universe to a DataFrame for vectorised eligibility checks.
      2. Apply all eligibility filters (ESG, business activities, UNGC, MSA).
         Stocks that fail any filter are removed; the first failure reason is
         recorded in the exclusion log.
      3. Run the optimizer on the eligible subset. The optimizer receives both
         the eligible DataFrame and the FULL set of underlying weights so it
         can compute the ineligible weight total and set constraint parameters
         correctly.
      4. Apply the minimum weight threshold: remove any stock whose optimized
         weight is below 1 bps (0.01%) and redistribute that weight to the rest.
      5. Compute the Weighted Average Carbon Intensity (WACI) of the optimized
         portfolio and of the original underlying index, for reporting.

    Parameters
    ----------
    universe:
        Full candidate universe including stocks that may be ineligible.
    universe_type:
        'developed' — S&P Developed ex-Australia LargeMidCap Carbon Aware Index.
                      ESG quartile screen uses global industry-group thresholds.
        'emerging'  — S&P Emerging LargeMidCap Carbon Aware Index.
                      ESG quartile screen uses the underlying index's own groups;
                      unscored companies are always treated as bottom quartile.

    Returns
    -------
    RebalanceResult with:
      - optimized_weights: final ticker → weight mapping (sums to ~1.0)
      - eligible_tickers: tickers included in the optimized portfolio
      - excluded_tickers: ticker → first exclusion reason for every removed stock
      - weighted_avg_carbon_intensity: WACI of the optimized index
      - underlying_weighted_avg_carbon_intensity: WACI of the underlying index
      - solver_status: cvxpy status string
      - relaxation_level: how many levels of constraint relaxation were needed

    Ref: §Index Construction (p. 8), §Eligibility Criteria (pp. 5-7)
    """
    # Flatten the Pydantic universe into a DataFrame that the eligibility and
    # optimization modules can work with using standard pandas operations.
    df = universe.to_dataframe()

    # --- Step 1: Eligibility filtering ---
    # eligible_df contains only stocks that passed every filter.
    # excluded maps every removed ticker to the reason it was first rejected.
    eligible_df, excluded = apply_eligibility_filters(df, universe_type=universe_type)

    # --- Step 2: Optimization ---
    # Pass the full underlying_weights dict (not just eligible stocks) so the
    # optimizer can compute ineligible_weight and calibrate the active-share
    # and diversification constraints correctly.
    opt_result = optimize(
        eligible_df=eligible_df,
        all_underlying_weights=universe.underlying_weights,
    )

    eligible_tickers = eligible_df["ticker"].tolist()

    # --- Step 3: Minimum weight threshold post-processing ---
    # The optimizer may assign very small weights (< 1 bps) to some stocks.
    # Per the methodology, these are zeroed and their weight is redistributed
    # proportionally to the stocks that remain above the threshold.
    # Ref: §Minimum Stock Weight Lower Threshold footnote 3 (p. 9)
    if opt_result.weights is not None and len(opt_result.weights) > 0:
        final_weights = apply_minimum_weight_threshold(
            opt_result.weights, eligible_tickers
        )
    else:
        # Optimizer returned nothing (e.g. no eligible stocks) — empty portfolio
        final_weights = {}

    # --- Step 4: Diagnostics ---
    # WACI of the optimized index (over eligible stocks only, at final weights)
    waci = _weighted_avg_carbon_intensity(eligible_df, final_weights)
    # WACI of the full underlying index (benchmark for comparison)
    underlying_waci = _underlying_waci(df, universe.underlying_weights)

    return RebalanceResult(
        optimized_weights=final_weights,
        # eligible_tickers reflects the post-threshold set (stocks zeroed by
        # the threshold step are not in final_weights and hence not listed here)
        eligible_tickers=list(final_weights.keys()),
        excluded_tickers=excluded,
        weighted_avg_carbon_intensity=waci,
        underlying_weighted_avg_carbon_intensity=underlying_waci,
        solver_status=opt_result.status,
        relaxation_level=opt_result.relaxation_level,
    )


# ---------------------------------------------------------------------------
# WACI helper functions
# ---------------------------------------------------------------------------


def _weighted_avg_carbon_intensity(
    eligible_df: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """
    Compute the Weighted Average Carbon Intensity of the optimized portfolio.

    WACI = Σᵢ (w[i] × CI[i])  summed over stocks with known carbon intensity.

    Stocks with missing carbon intensity (None) are excluded from both the
    numerator and the denominator. The denominator is the sum of weights for
    stocks that DO have CI data, so the result is not biased by missing coverage.

    Returns NaN if no weights are provided or if no stock has CI data.
    """
    if not weights:
        return float("nan")

    # Build a ticker → CI lookup from the eligible DataFrame
    ci_map = eligible_df.set_index("ticker")["carbon_intensity"].to_dict()

    total_waci = 0.0
    total_weight_with_ci = 0.0

    for ticker, w in weights.items():
        ci = ci_map.get(ticker)
        # Skip stocks with no carbon intensity data
        if ci is not None and not np.isnan(ci):
            total_waci += w * ci
            total_weight_with_ci += w

    return total_waci / total_weight_with_ci if total_weight_with_ci > 0 else float("nan")


def _underlying_waci(
    df: pd.DataFrame,
    underlying_weights: dict[str, float],
) -> float:
    """
    Compute the Weighted Average Carbon Intensity of the full underlying index.

    This is the pre-optimization benchmark WACI, computed over ALL underlying
    stocks (including those that will be excluded by eligibility filters) using
    their original benchmark weights. It provides the baseline for measuring
    how much the optimization reduced carbon intensity.

    Stocks without carbon intensity data are excluded from the calculation
    (same treatment as in _weighted_avg_carbon_intensity).
    """
    # Build a ticker → CI lookup from the full universe DataFrame
    ci_map = df.set_index("ticker")["carbon_intensity"].to_dict()

    total_waci = 0.0
    total_weight_with_ci = 0.0

    for ticker, w in underlying_weights.items():
        ci = ci_map.get(ticker)
        if ci is not None and not np.isnan(float(ci)):
            total_waci += w * float(ci)
            total_weight_with_ci += w

    return total_waci / total_weight_with_ci if total_weight_with_ci > 0 else float("nan")


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def result_to_dataframe(
    result: RebalanceResult,
    universe: IndexUniverse,
) -> pd.DataFrame:
    """
    Convert a RebalanceResult into a tidy DataFrame for display or CSV export.

    Each row represents one constituent of the optimized index and includes:
      - ticker / company_name / country / gics_industry_group — identifiers
      - underlying_weight — the stock's weight in the benchmark index
      - optimized_weight  — the stock's weight in the carbon-aware index
      - active_weight     — the difference (positive = overweight vs benchmark)
      - carbon_intensity  — tCO2e per million USD revenue (None if not available)

    Sorted by optimized_weight descending (largest position first).
    Weights are rounded to 6 decimal places to avoid floating-point noise.
    """
    rows = []
    # Build a lookup so we can pull stock metadata by ticker efficiently
    stock_by_ticker = {s.ticker: s for s in universe.stocks}

    for ticker, opt_weight in result.optimized_weights.items():
        stock = stock_by_ticker.get(ticker)
        underlying_w = universe.underlying_weights.get(ticker, 0.0)

        # active_weight > 0 means the optimizer overweighted this stock
        # active_weight < 0 means the optimizer underweighted it
        rows.append(
            {
                "ticker": ticker,
                "company_name": stock.company_name if stock else "",
                "country": stock.country if stock else "",
                "gics_industry_group": stock.gics_industry_group if stock else "",
                "underlying_weight": round(underlying_w, 6),
                "optimized_weight": round(opt_weight, 6),
                "active_weight": round(opt_weight - underlying_w, 6),
                "carbon_intensity": stock.carbon_intensity if stock else None,
            }
        )

    return pd.DataFrame(rows).sort_values("optimized_weight", ascending=False)
