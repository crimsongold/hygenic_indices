"""
Data models for the S&P Carbon Aware Index Series.

These models represent the inputs required to run the index methodology
described in "S&P Carbon Aware Index Series Methodology" (March 2026).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class UNGCStatus(str, Enum):
    """
    Sustainalytics Global Standards Screening (GSS) classification.
    Ref: Methodology §Exclusions Based on Sustainalytics' Global Standards Screening
    """

    COMPLIANT = "Compliant"
    WATCHLIST = "Watchlist"
    NON_COMPLIANT = "Non-Compliant"
    NO_COVERAGE = "No Coverage"


# ---------------------------------------------------------------------------
# Business activity exposure thresholds
# Ref: Methodology §Exclusions Based on Business Activities (pp. 5-6)
# ---------------------------------------------------------------------------

BUSINESS_ACTIVITY_THRESHOLDS: dict[str, dict[str, tuple[float, Optional[float]]]] = {
    # (level_of_involvement_threshold, significant_ownership_threshold)
    # None means the criterion does not apply for that column.
    # Revenue-based: exclude if revenue % > threshold
    # Ownership-based: exclude if ownership % >= threshold
    "controversial_weapons": {
        "tailor_made_essential": (0.0, 10.0),
        "non_tailor_made": (0.0, 10.0),
    },
    "tobacco": {
        "production": (0.0, None),
        "related_products_services": (5.0, None),
        "retail": (5.0, None),
    },
    "thermal_coal": {
        "extraction": (0.0, None),
        "power_generation": (0.0, None),
    },
    "oil_sands": {
        "extraction": (0.0, None),
    },
    "shale_energy": {
        "extraction": (0.0, None),
    },
    "arctic_oil_gas": {
        "extraction": (0.0, None),
    },
    "oil_gas": {
        "production": (0.0, None),
        "generation": (0.0, None),
        "supporting_products_services": (10.0, None),
    },
    "gambling": {
        "operations": (5.0, None),
        "specialized_equipment": (10.0, None),
        "supporting_products_services": (10.0, None),
    },
    "adult_entertainment": {
        "production": (0.0, None),
        "distribution": (5.0, None),
    },
    "alcoholic_beverages": {
        "production": (5.0, None),
        "retail": (10.0, None),
        "related_products_services": (10.0, None),
    },
}


class BusinessActivityExposures(BaseModel):
    """
    Revenue exposure percentages and ownership percentages for each screened activity.
    All values are expressed as percentages (0–100).

    Ref: Methodology §Exclusions Based on Business Activities (pp. 5-6)
    """

    # Controversial Weapons
    controversial_weapons_tailor_made_essential_pct: float = 0.0
    controversial_weapons_tailor_made_essential_ownership_pct: float = 0.0
    controversial_weapons_non_tailor_made_pct: float = 0.0
    controversial_weapons_non_tailor_made_ownership_pct: float = 0.0

    # Tobacco (revenue-based)
    tobacco_production_revenue_pct: float = 0.0
    tobacco_related_revenue_pct: float = 0.0
    tobacco_retail_revenue_pct: float = 0.0

    # Thermal Coal (revenue-based)
    thermal_coal_extraction_revenue_pct: float = 0.0
    thermal_coal_power_generation_revenue_pct: float = 0.0

    # Oil Sands (revenue-based)
    oil_sands_extraction_revenue_pct: float = 0.0

    # Shale Energy (revenue-based)
    shale_energy_extraction_revenue_pct: float = 0.0

    # Arctic Oil & Gas (revenue-based)
    arctic_oil_gas_extraction_revenue_pct: float = 0.0

    # Oil & Gas (revenue-based)
    oil_gas_production_revenue_pct: float = 0.0
    oil_gas_generation_revenue_pct: float = 0.0
    oil_gas_supporting_revenue_pct: float = 0.0

    # Gambling (revenue-based)
    gambling_operations_revenue_pct: float = 0.0
    gambling_equipment_revenue_pct: float = 0.0
    gambling_supporting_revenue_pct: float = 0.0

    # Adult Entertainment (revenue-based)
    adult_entertainment_production_revenue_pct: float = 0.0
    adult_entertainment_distribution_revenue_pct: float = 0.0

    # Alcoholic Beverages (revenue-based)
    alcoholic_beverages_production_revenue_pct: float = 0.0
    alcoholic_beverages_retail_revenue_pct: float = 0.0
    alcoholic_beverages_related_revenue_pct: float = 0.0

    # Sustainalytics coverage flag
    has_sustainalytics_coverage: bool = True


class Stock(BaseModel):
    """
    Represents a single security in the index universe with all data required
    to apply the S&P Carbon Aware eligibility and optimization logic.
    """

    ticker: str
    company_name: str
    country: str
    gics_sector: str
    gics_industry_group: str

    # Market data (used to derive underlying index weight)
    market_cap_usd: float  # total market cap in USD
    float_ratio: float  # fraction of shares in free float (0–1)

    # ESG data
    esg_score: Optional[float] = None  # S&P Global ESG Score (0–100), None if no coverage
    has_esg_coverage: bool = True

    # Carbon data (Trucost)
    # tCO2e per million USD revenue; None if not in Trucost coverage
    carbon_intensity: Optional[float] = None

    # Sustainalytics GSS status
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT

    # Business activity exposures
    business_activities: BusinessActivityExposures = Field(
        default_factory=BusinessActivityExposures
    )

    # MSA flag (set by Index Committee review)
    msa_flagged: bool = False

    @property
    def float_adjusted_market_cap(self) -> float:
        return self.market_cap_usd * self.float_ratio


class IndexUniverse(BaseModel):
    """
    The full universe of stocks eligible for consideration, together with
    their weights in the underlying index.
    """

    stocks: list[Stock]
    # Underlying index weights (float-adjusted market cap weighted), keyed by ticker.
    # Computed automatically from stocks if not provided.
    underlying_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_weights_if_missing(self) -> IndexUniverse:
        if not self.underlying_weights:
            self._compute_underlying_weights()
        return self

    def _compute_underlying_weights(self) -> None:
        total_float_cap = sum(s.float_adjusted_market_cap for s in self.stocks)
        if total_float_cap == 0:
            raise ValueError("Total float-adjusted market cap is zero.")
        self.underlying_weights = {
            s.ticker: s.float_adjusted_market_cap / total_float_cap
            for s in self.stocks
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return stocks as a DataFrame with underlying weights attached."""
        rows = []
        for s in self.stocks:
            ba = s.business_activities
            rows.append(
                {
                    "ticker": s.ticker,
                    "company_name": s.company_name,
                    "country": s.country,
                    "gics_sector": s.gics_sector,
                    "gics_industry_group": s.gics_industry_group,
                    "market_cap_usd": s.market_cap_usd,
                    "float_ratio": s.float_ratio,
                    "underlying_weight": self.underlying_weights[s.ticker],
                    "esg_score": s.esg_score,
                    "has_esg_coverage": s.has_esg_coverage,
                    "carbon_intensity": s.carbon_intensity,
                    "ungc_status": s.ungc_status.value,
                    "msa_flagged": s.msa_flagged,
                    "has_sustainalytics_coverage": ba.has_sustainalytics_coverage,
                    # Business activities
                    "controversial_weapons_tailor_made_essential_pct": ba.controversial_weapons_tailor_made_essential_pct,
                    "controversial_weapons_tailor_made_essential_ownership_pct": ba.controversial_weapons_tailor_made_essential_ownership_pct,
                    "controversial_weapons_non_tailor_made_pct": ba.controversial_weapons_non_tailor_made_pct,
                    "controversial_weapons_non_tailor_made_ownership_pct": ba.controversial_weapons_non_tailor_made_ownership_pct,
                    "tobacco_production_revenue_pct": ba.tobacco_production_revenue_pct,
                    "tobacco_related_revenue_pct": ba.tobacco_related_revenue_pct,
                    "tobacco_retail_revenue_pct": ba.tobacco_retail_revenue_pct,
                    "thermal_coal_extraction_revenue_pct": ba.thermal_coal_extraction_revenue_pct,
                    "thermal_coal_power_generation_revenue_pct": ba.thermal_coal_power_generation_revenue_pct,
                    "oil_sands_extraction_revenue_pct": ba.oil_sands_extraction_revenue_pct,
                    "shale_energy_extraction_revenue_pct": ba.shale_energy_extraction_revenue_pct,
                    "arctic_oil_gas_extraction_revenue_pct": ba.arctic_oil_gas_extraction_revenue_pct,
                    "oil_gas_production_revenue_pct": ba.oil_gas_production_revenue_pct,
                    "oil_gas_generation_revenue_pct": ba.oil_gas_generation_revenue_pct,
                    "oil_gas_supporting_revenue_pct": ba.oil_gas_supporting_revenue_pct,
                    "gambling_operations_revenue_pct": ba.gambling_operations_revenue_pct,
                    "gambling_equipment_revenue_pct": ba.gambling_equipment_revenue_pct,
                    "gambling_supporting_revenue_pct": ba.gambling_supporting_revenue_pct,
                    "adult_entertainment_production_revenue_pct": ba.adult_entertainment_production_revenue_pct,
                    "adult_entertainment_distribution_revenue_pct": ba.adult_entertainment_distribution_revenue_pct,
                    "alcoholic_beverages_production_revenue_pct": ba.alcoholic_beverages_production_revenue_pct,
                    "alcoholic_beverages_retail_revenue_pct": ba.alcoholic_beverages_retail_revenue_pct,
                    "alcoholic_beverages_related_revenue_pct": ba.alcoholic_beverages_related_revenue_pct,
                }
            )
        return pd.DataFrame(rows)


class RebalanceResult(BaseModel):
    """Output of the optimization: final constituent weights and diagnostics."""

    optimized_weights: dict[str, float]  # ticker -> weight
    eligible_tickers: list[str]
    excluded_tickers: dict[str, str]  # ticker -> reason for exclusion
    weighted_avg_carbon_intensity: float
    underlying_weighted_avg_carbon_intensity: float
    solver_status: str
    relaxation_level: int  # 0 = no relaxation; higher = more constraints relaxed
