"""
Rebalancer for the S&P ESG (Scored & Screened) Index Series.

Orchestrates the full rebalancing pipeline:
  1. Load the universe from a CSV file into typed Pydantic models.
  2. Apply eligibility filters to identify which stocks may enter the index.
  3. Rescale surviving stocks' float-adjusted market-cap weights to sum to 1.0.
  4. Return a RebalanceResult with final weights and a full exclusion log.

Unlike the S&P Carbon Aware index which uses a cvxpy optimizer to minimise
carbon intensity, the ESG index uses simple float-adjusted market-cap
rescaling. After excluded stocks are removed, the remaining stocks' underlying
weights are proportionally rescaled so they sum to 1.0. This preserves the
relative capitalisation ranking of the parent index while targeting ~75% of
each GICS Industry Group's float-adjusted market cap.

Ref: "S&P ESG Index Series Methodology" (S&P Dow Jones Indices)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .eligibility import apply_eligibility_filters
from .models import (
    BusinessActivityExposures,
    IndexUniverse,
    RebalanceResult,
    Stock,
    UNGCStatus,
)


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

# All columns that must be present in the input CSV. Missing any of these
# causes an early ValueError rather than a cryptic KeyError during parsing.
_REQUIRED_COLUMNS = {
    "ticker", "company_name", "country", "gics_sector", "gics_industry_group",
    "market_cap_usd", "float_ratio",
    "esg_score", "has_esg_coverage",
    "ungc_status", "msa_flagged",
    # Business activity columns -- one per screened activity sub-category
    "controversial_weapons_revenue_pct", "controversial_weapons_ownership_pct",
    "tobacco_production_revenue_pct", "tobacco_retail_revenue_pct",
    "thermal_coal_extraction_revenue_pct", "thermal_coal_power_revenue_pct",
    "small_arms_manufacture_revenue_pct", "small_arms_retail_revenue_pct",
}

# Business-activity percentage columns that map directly onto fields of
# BusinessActivityExposures. The list drives the CSV-to-model mapping in
# load_universe_from_csv. All are floats representing percentages (0-100).
_BUSINESS_ACTIVITY_COLS = [
    "controversial_weapons_revenue_pct",
    "controversial_weapons_ownership_pct",
    "tobacco_production_revenue_pct",
    "tobacco_retail_revenue_pct",
    "thermal_coal_extraction_revenue_pct",
    "thermal_coal_power_revenue_pct",
    "small_arms_manufacture_revenue_pct",
    "small_arms_retail_revenue_pct",
]


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_universe_from_csv(csv_path: Path | str) -> IndexUniverse:
    """
    Parse a universe CSV into an IndexUniverse of typed Stock models.

    The CSV must contain one row per security with all columns listed in
    _REQUIRED_COLUMNS. The underlying index weights are derived from the
    market_cap_usd and float_ratio columns -- you do not need to supply
    pre-computed weights.

    Type coercion:
        - esg_score may be blank (NaN in pandas) to represent missing data;
          it becomes None in the Stock model so downstream code can
          distinguish "no data" from "zero".
        - ungc_status is parsed as a UNGCStatus enum value; unrecognised
          strings default to UNGCStatus.NO_COVERAGE (treat as uncovered).
        - has_esg_coverage and msa_flagged are stored as 0/1 integers in
          the CSV and cast to bool here.
        - All business activity percentage columns are cast to float,
          defaulting to 0.0 for NaN values (missing = no involvement).

    Ref: §Eligibility Criteria, §Index Construction
    """
    df = pd.read_csv(csv_path)

    # Validate up front so errors surface early with a clear message
    # rather than as a KeyError deep in the parsing loop.
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    stocks = []
    for _, row in df.iterrows():
        # Optional field: blank CSV cells become None (not 0.0 or NaN),
        # so that downstream code can distinguish "no data" from "zero".
        esg = row["esg_score"]
        esg_val = None if (isinstance(esg, float) and np.isnan(esg)) else float(esg)

        # Build the BusinessActivityExposures model.
        # NaN values default to 0.0 -- missing data in revenue columns
        # means no known involvement, which is the safe default.
        ba_kwargs = {
            col: float(row[col]) if not (isinstance(row[col], float) and np.isnan(row[col])) else 0.0
            for col in _BUSINESS_ACTIVITY_COLS
        }

        # Parse UNGC status; fall back to NO_COVERAGE for any unrecognised value.
        # This treats unknown classifications conservatively -- the stock will be
        # excluded by the UNGC filter rather than incorrectly treated as compliant.
        ungc_raw = str(row["ungc_status"]).strip()
        try:
            ungc = UNGCStatus(ungc_raw)
        except ValueError:
            # Unrecognised string -> treat as if the company has no GSS coverage
            ungc = UNGCStatus.NO_COVERAGE

        stocks.append(Stock(
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            country=str(row["country"]),
            gics_sector=str(row["gics_sector"]),
            gics_industry_group=str(row["gics_industry_group"]),
            market_cap_usd=float(row["market_cap_usd"]),
            float_ratio=float(row["float_ratio"]),
            esg_score=esg_val,
            has_esg_coverage=bool(int(row["has_esg_coverage"])),
            ungc_status=ungc,
            msa_flagged=bool(int(row["msa_flagged"])),
            business_activities=BusinessActivityExposures(**ba_kwargs),
        ))

    # IndexUniverse automatically computes float-adjusted market-cap weights
    # for each stock if no explicit underlying_weights dict is provided.
    return IndexUniverse(stocks=stocks)


# ---------------------------------------------------------------------------
# Rebalancing pipeline
# ---------------------------------------------------------------------------


def rebalance(
    universe: IndexUniverse,
    universe_type: str = "standard",
) -> RebalanceResult:
    """
    Execute the full S&P ESG Index rebalancing pipeline and return results.

    Pipeline steps:

      1. Flatten the universe to a DataFrame for vectorised eligibility checks.
      2. Apply all eligibility filters (ESG score, business activities, UNGC,
         MSA). Stocks that fail any filter are removed; the first failure
         reason is recorded in the exclusion log.
      3. Rescale surviving stocks' underlying weights to sum to 1.0.

    The weighting step is deliberately simple: the S&P ESG index does NOT use
    an optimizer. It takes the float-adjusted market-cap weights from the
    parent index and rescales them proportionally after removing excluded
    stocks. This means the relative weight ordering from the parent index is
    preserved -- the largest company in the parent remains the largest in the
    ESG index (assuming it passes all screens).

    Parameters
    ----------
    universe:
        Full parent-index universe including stocks that may be ineligible.
    universe_type:
        'standard' -- default ESG index behaviour. Passed through to the
        ESG quartile filter for potential variant-specific logic.

    Returns
    -------
    RebalanceResult with:
      - rebalanced_weights: final ticker -> weight mapping (sums to ~1.0)
      - eligible_tickers: tickers included in the rebalanced portfolio
      - excluded_tickers: ticker -> first exclusion reason for every removed stock

    Ref: §Index Construction, §Eligibility Criteria
    """
    # Flatten the Pydantic universe into a DataFrame that the eligibility
    # module can work with using standard pandas operations.
    df = universe.to_dataframe()

    # --- Step 1: Eligibility filtering ---
    # eligible_df contains only stocks that passed every filter.
    # excluded maps every removed ticker to the reason it was first rejected.
    eligible_df, excluded = apply_eligibility_filters(df, universe_type=universe_type)

    if eligible_df.empty:
        # Edge case: every stock was excluded. Return an empty portfolio
        # rather than raising an error -- the caller can inspect excluded_tickers
        # to understand why.
        return RebalanceResult(
            rebalanced_weights={},
            eligible_tickers=[],
            excluded_tickers=excluded,
        )

    # --- Step 2: Float-cap rescaling ---
    # Take each eligible stock's underlying weight (which is already
    # proportional to float-adjusted market cap, computed in IndexUniverse)
    # and normalise so the eligible subset sums to 1.0. This is the ESG
    # index's weighting scheme -- no optimizer, just proportional rescaling.
    eligible_tickers = eligible_df["ticker"].tolist()
    raw_weights = {t: universe.underlying_weights[t] for t in eligible_tickers}
    total = sum(raw_weights.values())
    # Divide each weight by the total to normalise; this redistributes the
    # weight of excluded stocks proportionally across the survivors.
    rebalanced = {t: w / total for t, w in raw_weights.items()}

    return RebalanceResult(
        rebalanced_weights=rebalanced,
        eligible_tickers=eligible_tickers,
        excluded_tickers=excluded,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def result_to_dataframe(result: RebalanceResult, universe: IndexUniverse) -> pd.DataFrame:
    """
    Convert a RebalanceResult into a tidy DataFrame for display or CSV export.

    Each row represents one constituent of the rebalanced ESG index and includes:
      - ticker / company_name / country / gics_industry_group -- identifiers
      - underlying_weight -- the stock's weight in the parent benchmark index
      - rebalanced_weight -- the stock's weight in the ESG index
      - active_weight     -- the difference (positive = overweight vs benchmark)
      - esg_score         -- S&P Global ESG score (None if not available)

    Sorted by rebalanced_weight descending (largest position first).
    Weights are rounded to 6 decimal places to avoid floating-point noise.
    """
    # Build a lookup so we can pull stock metadata by ticker efficiently
    stock_map = {s.ticker: s for s in universe.stocks}
    rows = []
    for ticker, weight in sorted(
        result.rebalanced_weights.items(), key=lambda x: -x[1]
    ):
        s = stock_map[ticker]
        underlying_w = universe.underlying_weights[ticker]
        # active_weight > 0 means the stock is overweight vs the parent index
        # active_weight < 0 means the stock is underweight
        rows.append({
            "ticker": ticker,
            "company_name": s.company_name,
            "country": s.country,
            "gics_industry_group": s.gics_industry_group,
            "underlying_weight": round(underlying_w, 6),
            "rebalanced_weight": round(weight, 6),
            "active_weight": round(weight - underlying_w, 6),
            "esg_score": s.esg_score,
        })
    return pd.DataFrame(rows)
