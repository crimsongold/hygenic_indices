"""
Data models for the S&P ESG (Scored & Screened) Index Series.

The S&P ESG Index Series applies ESG score screening and business activity
exclusions to a parent S&P benchmark (e.g. S&P 500, S&P Global 1200), then
re-weights survivors by float-adjusted market capitalisation. The goal is to
retain approximately 75% of each GICS Industry Group's float-adjusted market
cap while excluding companies that fail sustainability screens.

Model hierarchy:
  - UNGCStatus: enum for UN Global Compact compliance classification
  - BusinessActivityExposures: revenue/ownership percentages for screened activities
  - Stock: single security with all fields needed for eligibility determination
  - IndexUniverse: full universe with auto-computed float-cap weights
  - RebalanceResult: output of the rebalancing pipeline

Design note: Pydantic models are used throughout so that type validation
happens at construction time. Invalid data (e.g. a negative market cap or
a string in a float field) raises immediately rather than causing subtle
bugs downstream in the eligibility or weighting logic.

Ref: "S&P ESG Index Series Methodology" (S&P Dow Jones Indices)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# UN Global Compact compliance status
# Ref: §UN Global Compact Compliance
#
# Sustainalytics' Global Standards Screening classifies companies into one of
# four categories. The index excludes Non-Compliant and No Coverage; Watchlist
# companies are retained at rebalancing but may be removed by the quarterly
# UNGC review conducted between rebalancings.
# ---------------------------------------------------------------------------


class UNGCStatus(str, Enum):
    """
    UN Global Compact compliance classification from Sustainalytics GSS.

    Values match the string representations used in the CSV data files so
    that parsing is a direct enum lookup (UNGCStatus(raw_string)).

    Ref: §UN Global Compact Compliance
    """

    COMPLIANT = "Compliant"
    WATCHLIST = "Watchlist"
    NON_COMPLIANT = "Non-Compliant"
    NO_COVERAGE = "No Coverage"


# ---------------------------------------------------------------------------
# Business activity exposures
# Ref: §Business Activity Exclusions
#
# The ESG index screens four business activity categories. Each has one or two
# sub-categories tracked as separate percentage fields. The thresholds are
# applied in eligibility.py; this model only stores the raw data.
#
# Controversial Weapons: revenue + ownership (two triggers)
# Tobacco:              production + retail
# Thermal Coal:         extraction + power generation
# Small Arms:           manufacture + retail
# ---------------------------------------------------------------------------


class BusinessActivityExposures(BaseModel):
    """
    Revenue and ownership exposures for the business activities screened by the
    S&P ESG Index Series.

    All values are expressed as percentages (0-100). A value of 0.0 means no
    known involvement; NaN values from the CSV are converted to 0.0 during
    loading (missing data = no known involvement, which is the safe default).

    The field names match the CSV column names exactly so that the CSV-to-model
    mapping in the rebalancer can use a simple dict comprehension.

    Ref: §Business Activity Exclusions
    """

    # Controversial Weapons (cluster munitions, anti-personnel mines, BWC/CWC/nuclear)
    # Revenue > 0% triggers exclusion; ownership >= 10% also triggers.
    controversial_weapons_revenue_pct: float = 0.0
    controversial_weapons_ownership_pct: float = 0.0

    # Tobacco Products
    # Production > 0% triggers exclusion; retail >= 10% triggers.
    tobacco_production_revenue_pct: float = 0.0
    tobacco_retail_revenue_pct: float = 0.0

    # Thermal Coal
    # Extraction > 5% triggers exclusion; power generation > 25% triggers.
    # These are higher thresholds than the Carbon Aware index (which uses > 0%)
    # because the ESG index permits some diversified exposure.
    thermal_coal_extraction_revenue_pct: float = 0.0
    thermal_coal_power_revenue_pct: float = 0.0

    # Small Arms (civilian-facing firearms)
    # Manufacture >= 5% triggers exclusion; retail >= 10% triggers.
    # Only civilian markets are screened -- military/law-enforcement contracts
    # are not captured by these fields.
    small_arms_manufacture_revenue_pct: float = 0.0
    small_arms_retail_revenue_pct: float = 0.0


# ---------------------------------------------------------------------------
# Stock model
# Ref: §Eligibility Criteria
# ---------------------------------------------------------------------------


class Stock(BaseModel):
    """
    Single security with all fields required for S&P ESG Index eligibility
    determination and weight calculation.

    Core identification fields:
      - ticker, company_name, country: basic security identity
      - gics_sector, gics_industry_group: GICS classification used for the
        per-industry-group ESG quartile screen

    Weight calculation fields:
      - market_cap_usd: total market capitalisation in USD
      - float_ratio: fraction of shares available for public trading (0-1).
        float_adjusted_market_cap = market_cap_usd * float_ratio
        This drives the underlying index weights.

    ESG screening fields:
      - esg_score: S&P Global ESG score (None if uncovered)
      - has_esg_coverage: whether S&P Global has assessed the company

    Compliance fields:
      - ungc_status: UN Global Compact classification from Sustainalytics
      - msa_flagged: True if the Index Committee has acted on an MSA alert

    Business activity exposures:
      - business_activities: revenue/ownership percentages for screened activities
    """

    ticker: str
    company_name: str
    country: str
    gics_sector: str
    gics_industry_group: str

    market_cap_usd: float
    float_ratio: float

    # ESG score -- None means the company has not been assessed or score is
    # unavailable. This is distinct from a score of 0.0 (which would mean
    # "assessed and scored zero").
    esg_score: Optional[float] = None
    has_esg_coverage: bool = True

    # UNGC compliance defaults to Compliant so that test fixtures only need
    # to set this for companies that are non-compliant or on the watchlist.
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT
    # MSA flag defaults to False (no controversy action taken).
    msa_flagged: bool = False

    business_activities: BusinessActivityExposures = Field(
        default_factory=BusinessActivityExposures
    )

    @property
    def float_adjusted_market_cap(self) -> float:
        """
        Compute float-adjusted market cap: market_cap_usd * float_ratio.

        This is the basis for the stock's weight in the parent index. Stocks
        with a lower float ratio (e.g. closely-held companies with a 30% free
        float) receive proportionally less weight than their total market cap
        would suggest.
        """
        return self.market_cap_usd * self.float_ratio


# ---------------------------------------------------------------------------
# Index universe
# Ref: §Index Construction
# ---------------------------------------------------------------------------


class IndexUniverse(BaseModel):
    """
    Full universe of stocks with automatically computed underlying
    float-cap weights.

    The underlying_weights dict maps ticker -> weight (summing to ~1.0).
    If not supplied at construction time, weights are computed from each
    stock's float_adjusted_market_cap as a proportion of the universe total.

    Design note: weights are stored as a dict rather than on each Stock so
    that the universe can be easily serialised and the weight calculation
    stays centralised in one place.
    """

    stocks: list[Stock]
    underlying_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_weights_if_missing(self) -> IndexUniverse:
        """Auto-compute float-cap weights if none were provided."""
        if not self.underlying_weights:
            self._compute_underlying_weights()
        return self

    def _compute_underlying_weights(self) -> None:
        """
        Compute each stock's weight as its share of total float-adjusted
        market cap.

        weight_i = float_adjusted_market_cap_i / sum(float_adjusted_market_cap)

        Raises ValueError if the total is zero (e.g. empty universe or all
        stocks have zero market cap), since weights would be undefined.
        """
        total = sum(s.float_adjusted_market_cap for s in self.stocks)
        if total == 0:
            raise ValueError("Total float-adjusted market cap is zero.")
        self.underlying_weights = {
            s.ticker: s.float_adjusted_market_cap / total for s in self.stocks
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten all stocks to a pandas DataFrame with underlying weights
        and business activity exposures unpacked into top-level columns.

        This is the primary interface between the Pydantic model layer and
        the pandas-based eligibility filtering logic. Each row contains all
        the fields needed by the exclusion checks in eligibility.py, so the
        filters can operate on plain column references without needing to
        access nested Pydantic objects.

        Column layout:
          - Stock identity: ticker, company_name, country, gics_sector,
            gics_industry_group
          - Weight/size: market_cap_usd, float_ratio, underlying_weight
          - ESG: esg_score, has_esg_coverage
          - Compliance: ungc_status (string value), msa_flagged (bool)
          - Business activities: all BusinessActivityExposures fields
        """
        rows = []
        for s in self.stocks:
            ba = s.business_activities
            rows.append({
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
                # Store as string value so eligibility filters can use
                # .isin(["Non-Compliant", "No Coverage"]) directly
                "ungc_status": s.ungc_status.value,
                "msa_flagged": s.msa_flagged,
                # Unpack all business activity fields into top-level columns
                "controversial_weapons_revenue_pct": ba.controversial_weapons_revenue_pct,
                "controversial_weapons_ownership_pct": ba.controversial_weapons_ownership_pct,
                "tobacco_production_revenue_pct": ba.tobacco_production_revenue_pct,
                "tobacco_retail_revenue_pct": ba.tobacco_retail_revenue_pct,
                "thermal_coal_extraction_revenue_pct": ba.thermal_coal_extraction_revenue_pct,
                "thermal_coal_power_revenue_pct": ba.thermal_coal_power_revenue_pct,
                "small_arms_manufacture_revenue_pct": ba.small_arms_manufacture_revenue_pct,
                "small_arms_retail_revenue_pct": ba.small_arms_retail_revenue_pct,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalance result
# ---------------------------------------------------------------------------


class RebalanceResult(BaseModel):
    """
    Output of the S&P ESG Index rebalance pipeline.

    Unlike the Carbon Aware index result, the ESG result does not include
    solver status or relaxation level because the ESG index uses simple
    float-cap rescaling rather than convex optimization.

    Fields:
      - rebalanced_weights: ticker -> final weight (sums to ~1.0). Empty
        dict if all stocks were excluded.
      - eligible_tickers: list of tickers in the final portfolio.
      - excluded_tickers: ticker -> first exclusion reason for every stock
        that was removed by eligibility filters.
    """

    rebalanced_weights: dict[str, float]  # ticker -> final weight
    eligible_tickers: list[str]
    excluded_tickers: dict[str, str]  # ticker -> first exclusion reason
