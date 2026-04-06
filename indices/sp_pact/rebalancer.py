"""
Rebalancer for the S&P PAB ESG & S&P CTB Indices.

Orchestrates the full rebalancing pipeline:
  1. Load the universe from a CSV file into typed Pydantic models.
  2. Convert the universe to a DataFrame for vectorised operations.
  3. Apply eligibility exclusions (variant-specific: CTB core screens only,
     PAB adds fossil fuel, weapons, and sin-stock screens).
  4. Run the constrained optimization (minimize tracking error subject to
     WACI reduction, decarbonization trajectory, SBTI, ESG, sector exposure,
     and non-disclosing company constraints).
  5. Post-process to enforce the minimum stock weight threshold (1 bps).
  6. Compute WACI diagnostics (optimized vs underlying, reduction %).
  7. Return a RebalanceResult with final weights, exclusion log, and diagnostics.

Design note:
    The rebalancer is variant-agnostic -- the Variant enum (CTB or PAB) is
    threaded through to both the eligibility module (which decides which
    exclusion screens to apply) and the optimization module (which uses
    different WACI reduction targets). This means the same rebalance()
    function handles both index variants without code duplication.

Ref: "S&P PAB ESG and S&P CTB Indices Methodology" (S&P Dow Jones Indices)
     §Index Construction (pp. 13-16)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .eligibility import apply_exclusions
from .models import (
    BusinessActivityExposures,
    IndexUniverse,
    RebalanceResult,
    Stock,
    UNGCStatus,
    Variant,
)
from .optimization import apply_minimum_weight_threshold, optimize


# ---------------------------------------------------------------------------
# Rebalancing pipeline
# ---------------------------------------------------------------------------


def rebalance(
    universe: IndexUniverse,
    variant: Variant = Variant.CTB,
    rebalance_quarter: int = 0,
    anchor_waci: Optional[float] = None,
) -> RebalanceResult:
    """
    Execute the full PACT rebalancing pipeline and return results.

    Pipeline steps:

      1. Flatten the universe to a DataFrame for vectorised eligibility checks.
      2. Apply all exclusion filters (carbon coverage, controversial weapons,
         tobacco, UNGC, MSA — plus PAB-only screens if variant is PAB).
         Stocks that fail any filter are removed; the first failure reason
         is recorded in the exclusion log.
      3. Run the optimizer on the eligible subset. The optimizer minimizes
         tracking error from the parent index while meeting climate
         constraints (WACI, SBTI, ESG, sector exposure, non-disclosing cap).
      4. Apply the minimum weight threshold: remove any stock whose optimized
         weight is below 1 bps (0.01%) and redistribute that weight.
      5. Compute WACI diagnostics for reporting.

    Parameters
    ----------
    universe:
        Full candidate universe including stocks that may be ineligible.
    variant:
        CTB (30% WACI reduction, core screens only) or
        PAB (50% WACI reduction, extended fossil fuel / sin-stock screens).
    rebalance_quarter:
        Number of quarters since the anchor date. Used to compute the 7%
        annual decarbonization trajectory target.
    anchor_waci:
        WACI at the anchor date. If None, the current underlying WACI is
        used as anchor (appropriate for the first rebalancing).

    Returns
    -------
    RebalanceResult with:
      - optimized_weights: final ticker -> weight mapping (sums to ~1.0)
      - eligible_tickers: tickers that passed all exclusion screens
      - excluded_tickers: ticker -> first exclusion reason for rejected stocks
      - weighted_avg_carbon_intensity: WACI of the optimized index
      - underlying_weighted_avg_carbon_intensity: WACI of the parent index
      - waci_reduction_pct: percentage WACI reduction achieved
      - solver_status: cvxpy status string
      - relaxation_level: how many constraint relaxation levels were needed
      - variant: 'ctb' or 'pab'

    Ref: §Index Construction (pp. 13-16)
    """
    # Flatten the Pydantic universe into a DataFrame that the eligibility
    # and optimization modules can work with using standard pandas operations.
    df = universe.to_dataframe()

    # --- Step 1: Eligibility filtering ---
    # eligible_df contains only stocks that passed every filter.
    # excluded maps every removed ticker to the reason it was first rejected.
    eligible_df, excluded = apply_exclusions(df, variant)
    eligible_tickers = eligible_df["ticker"].tolist()

    # Edge case: if no stocks survive eligibility, return an empty result
    # immediately rather than attempting optimization with zero variables.
    if not eligible_tickers:
        return RebalanceResult(
            optimized_weights={},
            eligible_tickers=[],
            excluded_tickers=excluded,
            weighted_avg_carbon_intensity=0.0,
            underlying_weighted_avg_carbon_intensity=0.0,
            waci_reduction_pct=0.0,
            solver_status="no_eligible_stocks",
            relaxation_level=0,
            variant=variant.value,
        )

    # --- Step 2: Optimization ---
    # Pass the full underlying_weights dict (not just eligible stocks) so
    # the optimizer can compute the rescaled eligible weights correctly.
    opt_result = optimize(
        eligible_df,
        universe.underlying_weights,
        variant=variant,
        rebalance_quarter=rebalance_quarter,
        anchor_waci=anchor_waci,
    )

    # --- Step 3: Minimum weight threshold post-processing ---
    # The optimizer may assign very small weights (< 1 bps) to some stocks.
    # Per the methodology, these are zeroed and their weight is redistributed
    # proportionally to the stocks that remain above the threshold.
    # Ref: §Minimum Stock Weight Lower Threshold (p. 16)
    if opt_result.weights is not None and len(opt_result.weights) > 0:
        final_weights = apply_minimum_weight_threshold(opt_result.weights, eligible_tickers)
    else:
        # Optimizer returned nothing (e.g. solver error) — empty portfolio
        final_weights = {}

    # --- Step 4: WACI diagnostics ---
    # Compute WACI for both the optimized and underlying portfolios.
    # total_carbon_intensity is the Scope 1+2+3 figure.
    ci_col = "total_carbon_intensity"
    opt_waci = _compute_waci_from_weights(eligible_df, final_weights, ci_col)
    underlying_waci = _compute_waci_from_weights(df, universe.underlying_weights, ci_col)

    # Calculate the percentage reduction achieved.
    # A positive value means the optimized WACI is lower (better).
    waci_reduction = 0.0
    if underlying_waci > 0:
        waci_reduction = (1 - opt_waci / underlying_waci) * 100

    return RebalanceResult(
        optimized_weights=final_weights,
        eligible_tickers=eligible_tickers,
        excluded_tickers=excluded,
        weighted_avg_carbon_intensity=opt_waci,
        underlying_weighted_avg_carbon_intensity=underlying_waci,
        waci_reduction_pct=waci_reduction,
        solver_status=opt_result.status,
        relaxation_level=opt_result.relaxation_level,
        variant=variant.value,
    )


# ---------------------------------------------------------------------------
# WACI helper function
# ---------------------------------------------------------------------------


def _compute_waci_from_weights(
    df: pd.DataFrame, weights: dict[str, float], ci_col: str
) -> float:
    """
    Compute the Weighted Average Carbon Intensity from a DataFrame and weight dict.

    WACI = sum(w[i] * CI[i]) / sum(w[i])  for all i where CI[i] is valid.

    Stocks with missing or NaN carbon intensity are excluded from both the
    numerator and the denominator so the result is not biased by missing
    coverage.

    Returns 0.0 if no valid data exists (defensive fallback rather than NaN,
    since downstream code uses WACI in division for reduction %).
    """
    total_w = 0.0
    waci = 0.0
    for _, row in df.iterrows():
        ci = row.get(ci_col)
        w = weights.get(row["ticker"], 0.0)
        # Skip stocks with no carbon intensity data
        if ci is None or (isinstance(ci, float) and np.isnan(ci)):
            continue
        waci += w * ci
        total_w += w
    if total_w == 0:
        return 0.0
    return waci / total_w


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def result_to_dataframe(result: RebalanceResult, universe: IndexUniverse) -> pd.DataFrame:
    """
    Convert a RebalanceResult into a summary DataFrame for display or CSV export.

    Each row represents one constituent of the optimized index and includes
    all fields from the universe DataFrame plus the optimized weight. Rows
    are sorted by optimized weight descending (largest position first) for
    easy inspection of the portfolio's largest holdings.
    """
    df = universe.to_dataframe()
    # Filter to only stocks that received non-zero optimized weight
    selected = df[df["ticker"].isin(result.optimized_weights.keys())].copy()
    # Map the optimized weight onto each row
    selected["optimized_weight"] = selected["ticker"].map(result.optimized_weights)
    return selected.sort_values("optimized_weight", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_universe_from_csv(path: Path) -> IndexUniverse:
    """
    Parse a PACT universe CSV into an IndexUniverse of typed Stock models.

    The CSV must contain one row per security. Underlying index weights are
    derived from the market_cap_usd and float_ratio columns -- you do not
    need to supply pre-computed weights.

    Type coercion:
        - esg_score, scope_1_2_carbon_intensity, and scope_3_carbon_intensity
          may be blank (NaN in pandas) to represent missing data; they become
          None in the Stock model.
        - ungc_status is parsed as a UNGCStatus enum value; unrecognised
          strings default to "Compliant".
        - Boolean flags (has_esg_coverage, has_carbon_coverage, has_sbti_target,
          msa_flagged) are stored as 0/1 in the CSV and cast to bool here.
        - All business activity percentage columns are cast to float with a
          default of 0.0 for missing values.

    Design note:
        row.get() with defaults is used for all business activity columns so
        that CSVs with only a subset of activity columns (e.g. CTB-only data
        without small arms or military fields) can still be loaded without
        error. Missing columns default to 0.0 (no exposure).

    Ref: §Eligibility Criteria (pp. 8-11), §Index Construction (pp. 13-16)
    """
    df = pd.read_csv(path)
    stocks = []
    for _, row in df.iterrows():
        # Build the BusinessActivityExposures model.
        # All percentage columns default to 0.0 if missing from the CSV,
        # so the model can handle CSVs with varying column sets.
        ba = BusinessActivityExposures(
            controversial_weapons_revenue_pct=float(row.get("controversial_weapons_revenue_pct", 0)),
            controversial_weapons_ownership_pct=float(row.get("controversial_weapons_ownership_pct", 0)),
            tobacco_production_revenue_pct=float(row.get("tobacco_production_revenue_pct", 0)),
            tobacco_related_revenue_pct=float(row.get("tobacco_related_revenue_pct", 0)),
            tobacco_retail_revenue_pct=float(row.get("tobacco_retail_revenue_pct", 0)),
            small_arms_civilian_revenue_pct=float(row.get("small_arms_civilian_revenue_pct", 0)),
            small_arms_noncivilian_revenue_pct=float(row.get("small_arms_noncivilian_revenue_pct", 0)),
            small_arms_key_components_revenue_pct=float(row.get("small_arms_key_components_revenue_pct", 0)),
            small_arms_retail_revenue_pct=float(row.get("small_arms_retail_revenue_pct", 0)),
            military_integral_weapons_revenue_pct=float(row.get("military_integral_weapons_revenue_pct", 0)),
            military_weapon_related_revenue_pct=float(row.get("military_weapon_related_revenue_pct", 0)),
            thermal_coal_generation_revenue_pct=float(row.get("thermal_coal_generation_revenue_pct", 0)),
            oil_sands_extraction_revenue_pct=float(row.get("oil_sands_extraction_revenue_pct", 0)),
            shale_oil_gas_extraction_revenue_pct=float(row.get("shale_oil_gas_extraction_revenue_pct", 0)),
            gambling_operations_revenue_pct=float(row.get("gambling_operations_revenue_pct", 0)),
            alcohol_production_revenue_pct=float(row.get("alcohol_production_revenue_pct", 0)),
            alcohol_related_revenue_pct=float(row.get("alcohol_related_revenue_pct", 0)),
            alcohol_retail_revenue_pct=float(row.get("alcohol_retail_revenue_pct", 0)),
            coal_revenue_pct=float(row.get("coal_revenue_pct", 0)),
            oil_revenue_pct=float(row.get("oil_revenue_pct", 0)),
            natural_gas_revenue_pct=float(row.get("natural_gas_revenue_pct", 0)),
            power_generation_revenue_pct=float(row.get("power_generation_revenue_pct", 0)),
        )

        # Optional fields: blank CSV cells become None (not 0.0 or NaN),
        # so that downstream code can distinguish "no data" from "zero".
        esg_score = float(row["esg_score"]) if pd.notna(row.get("esg_score")) else None
        scope_1_2 = float(row["scope_1_2_carbon_intensity"]) if pd.notna(row.get("scope_1_2_carbon_intensity")) else None
        scope_3 = float(row["scope_3_carbon_intensity"]) if pd.notna(row.get("scope_3_carbon_intensity")) else None

        # Parse UNGC status; default to Compliant for any unrecognised value
        ungc_status = UNGCStatus(row.get("ungc_status", "Compliant"))

        stocks.append(Stock(
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            country=str(row["country"]),
            gics_sector=str(row["gics_sector"]),
            gics_industry_group=str(row["gics_industry_group"]),
            market_cap_usd=float(row["market_cap_usd"]),
            float_ratio=float(row["float_ratio"]),
            esg_score=esg_score,
            has_esg_coverage=bool(row.get("has_esg_coverage", True)),
            scope_1_2_carbon_intensity=scope_1_2,
            scope_3_carbon_intensity=scope_3,
            has_carbon_coverage=bool(row.get("has_carbon_coverage", True)),
            has_sbti_target=bool(row.get("has_sbti_target", False)),
            ungc_status=ungc_status,
            msa_flagged=bool(row.get("msa_flagged", False)),
            business_activities=ba,
        ))

    # IndexUniverse automatically computes float-adjusted market-cap weights
    # for each stock if no explicit underlying_weights dict is provided.
    return IndexUniverse(stocks=stocks)
