"""
Rebalancer pipeline for the S&P Global Carbon Efficient Index Series.

Orchestrates the full rebalancing process for the Carbon Efficient index:

  1. Load the universe from a CSV file into typed Pydantic models (Stock, IndexUniverse).
  2. Flatten the universe to a DataFrame for vectorised operations.
  3. Compute Carbon Efficiency Factors (CEFs) using the exponential tilt formula.
  4. Multiply underlying weights by CEFs and normalise within GICS Industry Groups.
  5. Compute WACI diagnostics: pre-tilt (benchmark) and post-tilt (tilted index),
     plus the percentage reduction achieved by the tilt.
  6. Return a RebalanceResult with final weights and all diagnostics.

Key difference from S&P Carbon Aware:
    The Carbon Aware index uses hard exclusions (ESG, business activities, UNGC)
    followed by convex optimization with constraint relaxation. The Carbon Efficient
    index has NO exclusions and NO optimizer — it applies a deterministic
    multiplicative tilt to the benchmark weights. This makes the pipeline much
    simpler: load → tilt → report.

Ref: "S&P Global Carbon Efficient Index Series Methodology" (S&P Dow Jones Indices),
     §Index Construction, §Rebalancing Schedule
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .models import IndexUniverse, RebalanceResult, Stock
from .weighting import DEFAULT_LAMBDA, apply_tilt, weighted_avg_carbon_intensity


# ---------------------------------------------------------------------------
# CSV column definitions
# ---------------------------------------------------------------------------

# Minimum set of columns required in the universe CSV. Unlike the Carbon Aware
# index, we do not need ESG scores, UNGC status, business activity percentages,
# or MSA flags — the Carbon Efficient methodology uses only carbon intensity
# data from Trucost.
_REQUIRED_COLUMNS = {
    "ticker", "company_name", "country", "gics_sector", "gics_industry_group",
    "market_cap_usd", "float_ratio", "carbon_intensity",
}


# ---------------------------------------------------------------------------
# CSV loading
# Ref: §Constituent Data Requirements
# ---------------------------------------------------------------------------


def load_universe_from_csv(csv_path: Path | str) -> IndexUniverse:
    """
    Parse a universe CSV into an IndexUniverse of typed Stock models.

    The CSV must contain one row per security with the columns listed in
    _REQUIRED_COLUMNS. The underlying index weights are automatically derived
    from market_cap_usd and float_ratio — no pre-computed weights are needed.

    Type coercion:
        - carbon_intensity may be blank or NaN in the CSV, indicating the
          company has no Trucost coverage. These are converted to None in the
          Stock model so that downstream code can distinguish "no data" from
          "zero emissions". Companies without CI data receive a neutral CEF
          of 1.0 (their weight is unchanged by the tilt).

    Ref: §Constituent Data Requirements, §Carbon-to-Revenue Footprint
    """
    df = pd.read_csv(csv_path)

    # Validate columns before iterating rows — fail fast with a clear message
    # rather than a cryptic KeyError deep in the loop
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    stocks = []
    for _, row in df.iterrows():
        # Handle missing carbon intensity: NaN in pandas → None in the model.
        # This preserves the distinction between "zero CI" and "no data".
        ci = row["carbon_intensity"]
        ci_val = None if (isinstance(ci, float) and np.isnan(ci)) else float(ci)
        stocks.append(Stock(
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            country=str(row["country"]),
            gics_sector=str(row["gics_sector"]),
            gics_industry_group=str(row["gics_industry_group"]),
            market_cap_usd=float(row["market_cap_usd"]),
            float_ratio=float(row["float_ratio"]),
            carbon_intensity=ci_val,
        ))

    # IndexUniverse automatically computes float-adjusted market-cap weights
    # via its model_validator if no explicit weights are provided.
    return IndexUniverse(stocks=stocks)


# ---------------------------------------------------------------------------
# Rebalancing pipeline
# Ref: §Index Construction, §Rebalancing Schedule
# ---------------------------------------------------------------------------


def rebalance(
    universe: IndexUniverse,
    tilt_lambda: float = DEFAULT_LAMBDA,
) -> RebalanceResult:
    """
    Execute the full S&P Carbon Efficient rebalancing pipeline and return results.

    Pipeline steps:

      1. Flatten the Pydantic universe into a DataFrame for vectorised operations.
      2. Apply the carbon efficiency tilt: compute CEFs, multiply by underlying
         weights, and normalise. This is the core index construction step —
         unlike Carbon Aware, there is no exclusion phase or optimizer.
      3. Compute pre-tilt WACI (benchmark baseline) and post-tilt WACI (tilted
         index) for diagnostic reporting.
      4. Calculate the WACI reduction percentage: (1 - post/pre) * 100.
         A positive value means the tilt successfully reduced carbon intensity.

    Parameters
    ----------
    universe:
        Full benchmark universe (all constituents, no exclusions).
    tilt_lambda:
        Exponential tilt strength (default 0.5). Controls the trade-off between
        carbon intensity reduction and tracking error vs. the benchmark:
          - Higher lambda → stronger tilt → lower WACI but higher tracking error
          - Lower lambda → weaker tilt → WACI closer to benchmark
          - lambda = 0 → no tilt at all (tilted weights = underlying weights)

    Returns
    -------
    RebalanceResult with:
      - tilted_weights: final ticker -> weight mapping (sums to 1.0)
      - weighted_avg_carbon_intensity: post-tilt WACI
      - underlying_waci: pre-tilt benchmark WACI
      - waci_reduction_pct: percentage reduction achieved by the tilt

    Ref: §Index Construction, §Rebalancing Schedule
    """
    # Flatten the Pydantic universe into a DataFrame that the weighting module
    # can work with using standard pandas operations.
    df = universe.to_dataframe()

    # --- Step 1: Apply carbon efficiency tilt ---
    # This computes CEFs for every stock, multiplies by underlying weights,
    # and normalises. No stocks are excluded — all receive a tilted weight.
    tilted = apply_tilt(df, universe.underlying_weights, tilt_lambda=tilt_lambda)

    # --- Step 2: WACI diagnostics ---
    # Pre-tilt WACI: the benchmark's carbon intensity (what we are trying to reduce)
    pre_waci = weighted_avg_carbon_intensity(df, universe.underlying_weights)
    # Post-tilt WACI: the tilted index's carbon intensity (our result)
    post_waci = weighted_avg_carbon_intensity(df, tilted)

    # --- Step 3: Compute reduction percentage ---
    # reduction_pct = (1 - post/pre) * 100
    # Positive values indicate the tilt reduced WACI (the desired outcome).
    # NaN if the pre-tilt WACI is zero or NaN (cannot compute a meaningful ratio).
    if not np.isnan(pre_waci) and pre_waci > 0:
        reduction = (1.0 - post_waci / pre_waci) * 100.0
    else:
        reduction = float("nan")

    return RebalanceResult(
        tilted_weights=tilted,
        weighted_avg_carbon_intensity=post_waci,
        underlying_waci=pre_waci,
        waci_reduction_pct=reduction,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def result_to_dataframe(result: RebalanceResult, universe: IndexUniverse) -> pd.DataFrame:
    """
    Convert a RebalanceResult into a tidy DataFrame for display or CSV export.

    Each row represents one constituent of the tilted index and includes:
      - ticker / company_name / country / gics_industry_group — identifiers
      - underlying_weight — the stock's weight in the benchmark index
      - tilted_weight     — the stock's weight after the carbon efficiency tilt
      - active_weight     — the difference (positive = overweight vs benchmark,
                            meaning the stock has lower-than-average CI in its group)
      - carbon_intensity  — tCO2e per $M revenue from Trucost (None if not covered)

    Sorted by tilted_weight descending (largest position first).
    Weights are rounded to 6 decimal places to avoid floating-point noise in
    display output.
    """
    # Build a ticker -> Stock lookup for efficient metadata access
    stock_map = {s.ticker: s for s in universe.stocks}
    rows = []
    for ticker, weight in sorted(result.tilted_weights.items(), key=lambda x: -x[1]):
        s = stock_map[ticker]
        # active_weight > 0 → stock was overweighted by the tilt (low CI relative to group)
        # active_weight < 0 → stock was underweighted by the tilt (high CI relative to group)
        rows.append({
            "ticker": ticker,
            "company_name": s.company_name,
            "country": s.country,
            "gics_industry_group": s.gics_industry_group,
            "underlying_weight": round(universe.underlying_weights[ticker], 6),
            "tilted_weight": round(weight, 6),
            "active_weight": round(weight - universe.underlying_weights[ticker], 6),
            "carbon_intensity": s.carbon_intensity,
        })
    return pd.DataFrame(rows)
