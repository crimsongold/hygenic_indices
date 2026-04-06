"""
Data models for the Dow Jones Sustainability Diversified Indices.

Provides Pydantic models that represent the full data pipeline:

  Stock:
    A single security with all fields required for eligibility screening
    and best-in-class selection — identifiers, financials, ESG score,
    UNGC compliance status, MSA flag, and business activity revenue exposures.

  BusinessActivityExposures:
    Revenue and ownership percentages for each of the seven screened activity
    categories. These map directly to the exclusion thresholds defined in the
    methodology (pp. 5-6).

  IndexUniverse:
    The complete candidate universe for a given rebalancing date. Wraps a list
    of Stocks and auto-computes float-adjusted market-cap weights. The
    to_dataframe() method flattens the typed models into a pandas DataFrame
    for vectorised eligibility and selection operations.

  RebalanceResult:
    Output of the rebalancing pipeline — final capped weights, selected
    tickers, exclusion log, and which tickers were capped.

The DJSI Diversified Select methodology groups companies by GICS Sector x
Region (North America, EMEA, Asia/Pacific) and selects the highest-scoring
companies by S&P Corporate Sustainability Assessment (CSA) within each group,
targeting approximately 50% of each group's float-adjusted market cap.
Selected companies are weighted by float-adjusted market cap, with a 10%
single-stock cap and iterative redistribution of excess weight.

Rebalancing occurs semi-annually in March and September.

Ref: "Dow Jones Sustainability Diversified Indices Methodology" (S&P Dow Jones Indices)
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class UNGCStatus(str, Enum):
    """
    UN Global Compact compliance classification from Sustainalytics GSS.

    Sustainalytics' Global Standards Screening (GSS) assesses each company's
    adherence to the ten principles of the UN Global Compact. The four
    possible outcomes determine eligibility:

      COMPLIANT     — no known violations; eligible for the index.
      WATCHLIST     — potential violation under investigation; retained at
                      rebalancing but may be removed at a quarterly review.
      NON_COMPLIANT — confirmed violation; excluded from the index.
      NO_COVERAGE   — not assessed by Sustainalytics; excluded as a
                      precaution since compliance cannot be verified.

    Ref: §UN Global Compact Compliance
    """

    COMPLIANT = "Compliant"
    WATCHLIST = "Watchlist"
    NON_COMPLIANT = "Non-Compliant"
    NO_COVERAGE = "No Coverage"


# ---------------------------------------------------------------------------
# Business Activity Exposures
# Ref: §Exclusions Based on Business Activities (pp. 5-6)
#
# Each field represents a revenue or ownership percentage (0-100) for a
# specific screened activity. The field names encode both the activity
# category and the sub-category (e.g. production vs retail).
#
# Default values are 0.0, meaning "no involvement" — this is the safe
# default for companies where Sustainalytics data is unavailable or the
# CSV column is missing.
# ---------------------------------------------------------------------------


class BusinessActivityExposures(BaseModel):
    """
    Revenue and ownership exposures for activities screened by the DJSI
    Diversified Select methodology. All values are expressed as percentages
    (0-100).

    The seven screened categories and their exclusion thresholds:

      Controversial Weapons: >0% revenue OR >=10% ownership
        - Zero-tolerance on direct revenue; ownership captures indirect
          participation through subsidiary stakes.

      Tobacco: production >0%
        - Only manufacturing is screened; retail/distribution are not excluded.

      Adult Entertainment: production >=5%, retail >=5%
        - Materiality threshold allows incidental exposure (< 5%) to pass.

      Alcohol: production >0%
        - Zero-tolerance on manufacturing; retail is not screened.

      Gambling: operations >0%, equipment >0%
        - Both sub-categories use zero-tolerance thresholds.

      Military Contracting: integral weapons >=5%, weapon related >=5%
        - Materiality threshold allows diversified defence companies with
          small weapons-related revenue to remain eligible.

      Small Arms: civilian >=5%, key components >=5%, non-civilian >=5%,
                  retail >=5%
        - All four sub-categories use the same 5% materiality threshold
          to cover the full firearms value chain.

    Ref: Methodology §Exclusions Based on Business Activities (pp. 5-6)
    """

    # --- Controversial Weapons ---
    # Direct revenue from manufacturing or supplying controversial weapons.
    # Threshold: > 0% (zero-tolerance)
    controversial_weapons_revenue_pct: float = 0.0
    # Ownership stake in a subsidiary involved in controversial weapons.
    # Threshold: >= 10% (significant ownership)
    controversial_weapons_ownership_pct: float = 0.0

    # --- Tobacco ---
    # Revenue from manufacturing tobacco products.
    # Threshold: > 0% (zero-tolerance on production)
    tobacco_production_revenue_pct: float = 0.0

    # --- Adult Entertainment ---
    # Revenue from producing adult content or operating adult venues.
    # Threshold: >= 5% (materiality threshold)
    adult_entertainment_production_revenue_pct: float = 0.0
    # Revenue from distributing or retailing adult entertainment.
    # Threshold: >= 5% (materiality threshold)
    adult_entertainment_retail_revenue_pct: float = 0.0

    # --- Alcohol ---
    # Revenue from manufacturing alcoholic beverages.
    # Threshold: > 0% (zero-tolerance on production)
    alcohol_production_revenue_pct: float = 0.0

    # --- Gambling ---
    # Revenue from owning/operating gambling establishments.
    # Threshold: > 0% (zero-tolerance)
    gambling_operations_revenue_pct: float = 0.0
    # Revenue from manufacturing gambling-specific equipment.
    # Threshold: > 0% (zero-tolerance)
    gambling_equipment_revenue_pct: float = 0.0

    # --- Military Contracting ---
    # Revenue from manufacturing integral/complete weapons systems.
    # Threshold: >= 5% (materiality — allows diversified defence companies)
    military_integral_weapons_revenue_pct: float = 0.0
    # Revenue from weapon-related products/services (guidance, ammo, etc.).
    # Threshold: >= 5% (materiality — same rationale as integral weapons)
    military_weapon_related_revenue_pct: float = 0.0

    # --- Small Arms ---
    # Revenue from manufacturing civilian firearms (hunting, sporting, etc.).
    # Threshold: >= 5%
    small_arms_civilian_production_revenue_pct: float = 0.0
    # Revenue from manufacturing key firearm components (barrels, receivers).
    # Threshold: >= 5%
    small_arms_key_components_revenue_pct: float = 0.0
    # Revenue from manufacturing military/law-enforcement firearms.
    # Threshold: >= 5%
    small_arms_noncivilian_production_revenue_pct: float = 0.0
    # Revenue from distributing/retailing small arms to consumers.
    # Threshold: >= 5%
    small_arms_retail_revenue_pct: float = 0.0


# ---------------------------------------------------------------------------
# Stock model
# ---------------------------------------------------------------------------


class Stock(BaseModel):
    """
    Single security with all fields required for DJSI Diversified eligibility
    screening, best-in-class selection, and weight computation.

    Design notes:
        - `region` is used alongside `gics_sector` to form the Sector x Region
          groups for best-in-class selection. Valid regions are "North America",
          "EMEA", and "Asia/Pacific".
        - `esg_score` is Optional because some companies lack S&P Global ESG
          coverage. A None value causes exclusion at the ESG coverage screen.
        - `float_adjusted_market_cap` is a computed property (market_cap_usd *
          float_ratio) used for both selection targeting and weight computation.
        - Business activity exposures are stored in a nested model to keep the
          Stock model clean while providing all 14 revenue/ownership fields
          needed for the seven exclusion screens.
    """

    ticker: str
    company_name: str
    country: str
    # Region used for selection grouping (e.g. "North America", "EMEA", "Asia/Pacific")
    region: str
    gics_sector: str
    gics_industry_group: str

    # Raw total market capitalization in USD
    market_cap_usd: float
    # Fraction of shares available for public trading (0.0 to 1.0)
    float_ratio: float

    # S&P Global ESG Score (0-100), derived from the Corporate Sustainability
    # Assessment (CSA). None means the company has not been assessed.
    esg_score: Optional[float] = None
    # Whether S&P Global has ESG coverage for this company
    has_esg_coverage: bool = True

    # UN Global Compact compliance status from Sustainalytics GSS
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT
    # Whether the Index Committee has flagged this company via the MSA overlay
    msa_flagged: bool = False

    # Revenue/ownership exposures for all screened business activities
    business_activities: BusinessActivityExposures = Field(
        default_factory=BusinessActivityExposures
    )

    @property
    def float_adjusted_market_cap(self) -> float:
        """
        Float-adjusted market cap = total market cap * float ratio.

        This is the investable market cap — the portion available for public
        trading. Used for both the 50% FMC targeting in best-in-class selection
        and for computing initial portfolio weights before capping.
        """
        return self.market_cap_usd * self.float_ratio


# ---------------------------------------------------------------------------
# Index Universe
# ---------------------------------------------------------------------------


class IndexUniverse(BaseModel):
    """
    Full universe of DJSI Diversified candidates with float-cap weights.

    Wraps a list of Stock models and computes underlying float-adjusted
    market-cap weights automatically via a Pydantic model validator. The
    weights represent each stock's share of the total investable market cap
    before any eligibility screening or selection.

    The to_dataframe() method flattens the typed Pydantic models into a
    pandas DataFrame for vectorised operations in the eligibility and
    selection modules. Every Stock field and BusinessActivityExposures field
    is expanded into its own column.

    Design note:
        - underlying_weights is computed automatically if not provided.
          This allows callers to either supply custom weights (for testing)
          or let the model derive them from market_cap_usd * float_ratio.
    """

    stocks: list[Stock]
    # Ticker -> underlying float-adjusted market-cap weight (sums to 1.0)
    underlying_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_weights_if_missing(self) -> IndexUniverse:
        """Auto-compute float-adjusted market-cap weights if not provided."""
        if not self.underlying_weights:
            self._compute_underlying_weights()
        return self

    def _compute_underlying_weights(self) -> None:
        """
        Compute each stock's weight as its float-adjusted market cap divided
        by the total float-adjusted market cap of all stocks in the universe.

        Raises ValueError if total FMC is zero — this would indicate empty or
        degenerate input data that cannot produce meaningful weights.
        """
        total = sum(s.float_adjusted_market_cap for s in self.stocks)
        if total == 0:
            raise ValueError("Total float-adjusted market cap is zero.")
        self.underlying_weights = {
            s.ticker: s.float_adjusted_market_cap / total for s in self.stocks
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten the universe into a pandas DataFrame for vectorised operations.

        Each row represents one stock. Columns include:
          - Identifiers: ticker, company_name, country, region, gics_sector,
            gics_industry_group
          - Financials: market_cap_usd, float_ratio, underlying_weight,
            float_adjusted_market_cap
          - ESG: esg_score, has_esg_coverage
          - Compliance: ungc_status (string value), msa_flagged
          - Business activities: all 14 revenue/ownership percentage columns
            from BusinessActivityExposures, expanded as top-level columns

        The BusinessActivityExposures fields are expanded into top-level columns
        (rather than kept as a nested object) so that the eligibility filters
        can access them directly via df["column_name"] without needing to
        unpack nested structures.
        """
        rows = []
        for s in self.stocks:
            ba = s.business_activities
            rows.append({
                "ticker": s.ticker,
                "company_name": s.company_name,
                "country": s.country,
                "region": s.region,
                "gics_sector": s.gics_sector,
                "gics_industry_group": s.gics_industry_group,
                "market_cap_usd": s.market_cap_usd,
                "float_ratio": s.float_ratio,
                "underlying_weight": self.underlying_weights[s.ticker],
                "float_adjusted_market_cap": s.float_adjusted_market_cap,
                "esg_score": s.esg_score,
                "has_esg_coverage": s.has_esg_coverage,
                "ungc_status": s.ungc_status.value,
                "msa_flagged": s.msa_flagged,
                # --- Business activity columns (expanded from nested model) ---
                "controversial_weapons_revenue_pct": ba.controversial_weapons_revenue_pct,
                "controversial_weapons_ownership_pct": ba.controversial_weapons_ownership_pct,
                "tobacco_production_revenue_pct": ba.tobacco_production_revenue_pct,
                "adult_entertainment_production_revenue_pct": ba.adult_entertainment_production_revenue_pct,
                "adult_entertainment_retail_revenue_pct": ba.adult_entertainment_retail_revenue_pct,
                "alcohol_production_revenue_pct": ba.alcohol_production_revenue_pct,
                "gambling_operations_revenue_pct": ba.gambling_operations_revenue_pct,
                "gambling_equipment_revenue_pct": ba.gambling_equipment_revenue_pct,
                "military_integral_weapons_revenue_pct": ba.military_integral_weapons_revenue_pct,
                "military_weapon_related_revenue_pct": ba.military_weapon_related_revenue_pct,
                "small_arms_civilian_production_revenue_pct": ba.small_arms_civilian_production_revenue_pct,
                "small_arms_key_components_revenue_pct": ba.small_arms_key_components_revenue_pct,
                "small_arms_noncivilian_production_revenue_pct": ba.small_arms_noncivilian_production_revenue_pct,
                "small_arms_retail_revenue_pct": ba.small_arms_retail_revenue_pct,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalance result
# ---------------------------------------------------------------------------


class RebalanceResult(BaseModel):
    """
    Output of the DJSI Diversified rebalance pipeline.

    Captures everything needed to understand the rebalancing outcome:
      - rebalanced_weights: the final portfolio (ticker -> weight after capping)
      - selected_tickers: which companies were chosen by best-in-class selection
      - excluded_tickers: which companies were excluded and why (first reason)
      - capped_tickers: which companies had their weight reduced by the 10% cap

    The separation of selected_tickers and capped_tickers allows downstream
    reporting to distinguish between "selected but capped" and "selected and
    uncapped" constituents.
    """

    # Final capped weights: ticker -> weight (sums to ~1.0)
    rebalanced_weights: dict[str, float]
    # Tickers chosen by the best-in-class selection algorithm
    selected_tickers: list[str]
    # Tickers removed during hard exclusion screening: ticker -> first reason
    excluded_tickers: dict[str, str]
    # Tickers whose weights were reduced by the single-stock cap redistribution
    capped_tickers: list[str]
