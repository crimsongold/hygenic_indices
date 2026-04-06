"""
Rebalancer for the S&P Carbon Aware Index Series.

Orchestrates the full rebalancing pipeline:
  1. Build universe from CSV input
  2. Apply eligibility filters
  3. Run optimization
  4. Apply minimum-weight post-processing
  5. Return a RebalanceResult with diagnostics

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
# CSV loading
# ---------------------------------------------------------------------------

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
    "has_sustainalytics_coverage",
]


def load_universe_from_csv(csv_path: str | Path) -> IndexUniverse:
    """
    Load an index universe from a CSV file.

    Required columns: ticker, company_name, country, gics_sector,
    gics_industry_group, market_cap_usd, float_ratio, esg_score,
    has_esg_coverage, carbon_intensity, ungc_status, msa_flagged,
    plus all business activity columns.

    Ref: §Eligibility Criteria, §Index Construction
    """
    df = pd.read_csv(csv_path)
    _validate_columns(df)

    stocks: list[Stock] = []
    for _, row in df.iterrows():
        ba_kwargs = {
            col: float(row[col]) if col != "has_sustainalytics_coverage"
            else bool(row[col])
            for col in _BUSINESS_ACTIVITY_COLUMNS
            if col in row.index
        }
        ba = BusinessActivityExposures(**ba_kwargs)

        esg_score = float(row["esg_score"]) if pd.notna(row["esg_score"]) else None
        carbon_intensity = (
            float(row["carbon_intensity"]) if pd.notna(row["carbon_intensity"]) else None
        )
        ungc_status_str = str(row["ungc_status"]) if pd.notna(row["ungc_status"]) else "No Coverage"
        try:
            ungc_status = UNGCStatus(ungc_status_str)
        except ValueError:
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

    return IndexUniverse(stocks=stocks)


def _validate_columns(df: pd.DataFrame) -> None:
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
    Run the full rebalancing pipeline for the S&P Carbon Aware Index.

    Parameters
    ----------
    universe:
        The full index universe including all candidate stocks.
    universe_type:
        'developed' for S&P Developed ex-Australia LargeMidCap Carbon Aware Index
        'emerging'  for S&P Emerging LargeMidCap Carbon Aware Index

    Returns
    -------
    RebalanceResult with optimized weights and diagnostics.

    Ref: §Index Construction (p. 8), §Eligibility Criteria (pp. 5-7)
    """
    df = universe.to_dataframe()

    # Step 1: Apply eligibility filters
    eligible_df, excluded = apply_eligibility_filters(df, universe_type=universe_type)

    # Step 2: Run the optimizer
    opt_result = optimize(
        eligible_df=eligible_df,
        all_underlying_weights=universe.underlying_weights,
    )

    eligible_tickers = eligible_df["ticker"].tolist()

    # Step 3: Apply minimum stock weight threshold (1 bps post-processing)
    # Ref: §Minimum Stock Weight Lower Threshold footnote 3 (p. 9)
    if opt_result.weights is not None and len(opt_result.weights) > 0:
        final_weights = apply_minimum_weight_threshold(
            opt_result.weights, eligible_tickers
        )
    else:
        final_weights = {}

    # Step 4: Compute diagnostics
    waci = _weighted_avg_carbon_intensity(eligible_df, final_weights)
    underlying_waci = _underlying_waci(df, universe.underlying_weights)

    return RebalanceResult(
        optimized_weights=final_weights,
        eligible_tickers=list(final_weights.keys()),
        excluded_tickers=excluded,
        weighted_avg_carbon_intensity=waci,
        underlying_weighted_avg_carbon_intensity=underlying_waci,
        solver_status=opt_result.status,
        relaxation_level=opt_result.relaxation_level,
    )


def _weighted_avg_carbon_intensity(
    eligible_df: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """Compute the weighted-average carbon intensity of the optimized index."""
    if not weights:
        return float("nan")
    ci_map = eligible_df.set_index("ticker")["carbon_intensity"].to_dict()
    total = 0.0
    total_weight = 0.0
    for ticker, w in weights.items():
        ci = ci_map.get(ticker)
        if ci is not None and not np.isnan(ci):
            total += w * ci
            total_weight += w
    return total / total_weight if total_weight > 0 else float("nan")


def _underlying_waci(
    df: pd.DataFrame,
    underlying_weights: dict[str, float],
) -> float:
    """Compute the weighted-average carbon intensity of the underlying index."""
    ci_map = df.set_index("ticker")["carbon_intensity"].to_dict()
    total = 0.0
    total_weight = 0.0
    for ticker, w in underlying_weights.items():
        ci = ci_map.get(ticker)
        if ci is not None and not np.isnan(float(ci) if ci is not None else float("nan")):
            total += w * float(ci)
            total_weight += w
    return total / total_weight if total_weight > 0 else float("nan")


# ---------------------------------------------------------------------------
# Result formatting helpers
# ---------------------------------------------------------------------------


def result_to_dataframe(
    result: RebalanceResult,
    universe: IndexUniverse,
) -> pd.DataFrame:
    """
    Return the rebalance result as a tidy DataFrame for display or export.
    Includes optimized weight, underlying weight, and active weight.
    """
    rows = []
    all_tickers = {s.ticker: s for s in universe.stocks}

    for ticker, opt_weight in result.optimized_weights.items():
        stock = all_tickers.get(ticker)
        underlying_w = universe.underlying_weights.get(ticker, 0.0)
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
