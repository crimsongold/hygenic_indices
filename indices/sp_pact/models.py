"""
Data models for the S&P PAB ESG & S&P CTB Indices.

Two index variants share the same architecture but with different constraint
thresholds and exclusion scopes:

  - CTB (Climate Transition Benchmark): 30% WACI reduction target, broader
    inclusion with fewer business-activity exclusions. Designed to be a
    minimum-standard climate index suitable as a benchmark for portfolios
    in transition toward lower carbon intensity.

  - PAB (Paris-Aligned Benchmark): 50% WACI reduction target, stricter
    fossil fuel and sin-stock exclusions (small arms, military contracting,
    gambling, alcohol). Aligned with the Paris Agreement's 1.5 C warming
    target and intended for portfolios seeking aggressive decarbonization.

Both variants require Scope 1+2+3 carbon data for all constituents and share
a 7% annual decarbonization trajectory after the anchor year.

Design note:
    The Variant enum is threaded through eligibility filters and optimization
    constraints so that a single codebase handles both CTB and PAB without
    duplication. Threshold differences are parameterized rather than hard-coded
    in separate branches.

Ref: "S&P PAB ESG and S&P CTB Indices Methodology" (S&P Dow Jones Indices)
     pp. 8-16
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Index variant enum
# ---------------------------------------------------------------------------


class Variant(str, Enum):
    """
    Index variant — controls which exclusion screens and constraint thresholds
    apply throughout the pipeline.

    CTB uses lenient thresholds (e.g. 30% WACI reduction, tobacco retail at
    >=10%) while PAB uses strict thresholds (e.g. 50% WACI reduction, tobacco
    retail at >=5%, plus additional fossil-fuel and sin-stock screens).

    Ref: §Index Variants (p. 8)
    """

    CTB = "ctb"
    PAB = "pab"


# ---------------------------------------------------------------------------
# UNGC Global Standards Screening classification
# Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 10)
# ---------------------------------------------------------------------------


class UNGCStatus(str, Enum):
    """
    Sustainalytics Global Standards Screening (GSS) classification.

    Sustainalytics assesses each company's adherence to the ten principles of
    the UN Global Compact (human rights, labour, environment, anti-corruption):

      COMPLIANT     — no known violations; eligible for the index.
      WATCHLIST     — potential violation under investigation; retained at
                      rebalancing but subject to quarterly review.
      NON_COMPLIANT — confirmed violation of UNGC principles; EXCLUDED.
      NO_COVERAGE   — Sustainalytics has not assessed the company; EXCLUDED
                      as a precaution (cannot confirm compliance).

    Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 10)
    """

    COMPLIANT = "Compliant"
    WATCHLIST = "Watchlist"
    NON_COMPLIANT = "Non-Compliant"
    NO_COVERAGE = "No Coverage"


# ---------------------------------------------------------------------------
# Business activity exposure model
# Ref: §Exclusions Based on Business Activities (pp. 9-10)
#
# Each field stores a revenue percentage (0-100) or ownership percentage
# for a specific screened activity. The comments note which variant uses
# each field and the exclusion threshold that applies.
#
# CTB and PAB share core exclusions (controversial weapons, tobacco, UNGC,
# MSA). PAB adds screens for small arms, military contracting, fossil fuels,
# thermal coal, oil sands, shale, gambling, and alcohol.
# ---------------------------------------------------------------------------


class BusinessActivityExposures(BaseModel):
    """
    Revenue / ownership exposures for activities screened by PACT methodology.
    All values are percentages (0-100).

    Ref: §Exclusions Based on Business Activities (pp. 9-10)
    """

    # --- Controversial Weapons (CTB & PAB) ---
    # Covers cluster munitions, anti-personnel mines, biological/chemical/
    # nuclear weapons. Any revenue or significant ownership triggers exclusion.
    # Thresholds: revenue > 0%, ownership >= 25%
    controversial_weapons_revenue_pct: float = 0.0
    controversial_weapons_ownership_pct: float = 0.0

    # --- Tobacco (CTB & PAB) ---
    # Production of tobacco products — any involvement excludes (> 0%).
    tobacco_production_revenue_pct: float = 0.0   # >0%
    # Tobacco-related products/services to manufacturers (>= 10%).
    tobacco_related_revenue_pct: float = 0.0       # >=10%
    # Tobacco retail — threshold differs by variant:
    #   PAB: >= 5% (stricter — a retailer with moderate tobacco exposure excluded)
    #   CTB: >= 10% (only retailers heavily dependent on tobacco sales excluded)
    tobacco_retail_revenue_pct: float = 0.0        # >=5% PAB, >=10% CTB

    # --- Small Arms (PAB only) ---
    # All sub-categories use a zero-tolerance threshold (> 0%).
    # The PAB methodology excludes any involvement in the manufacture,
    # distribution, or retail of firearms intended for civilian or military use.
    small_arms_civilian_revenue_pct: float = 0.0   # >0%
    small_arms_noncivilian_revenue_pct: float = 0.0  # >0%
    small_arms_key_components_revenue_pct: float = 0.0  # >0%
    small_arms_retail_revenue_pct: float = 0.0     # >0%

    # --- Military Contracting (PAB only) ---
    # Integral weapons systems: any involvement excludes (> 0%).
    # Weapon-related products/services: material involvement excludes (>= 5%).
    military_integral_weapons_revenue_pct: float = 0.0  # >0%
    military_weapon_related_revenue_pct: float = 0.0    # >=5%

    # --- Thermal Coal (PAB only) ---
    # Power generation from thermal coal >= 5% of revenue.
    # Unlike the Carbon Aware index (which uses > 0%), the PACT PAB variant
    # allows de minimis coal exposure below 5%.
    thermal_coal_generation_revenue_pct: float = 0.0

    # --- Oil Sands / Tar Sands (PAB only) ---
    # Extraction of oil sands >= 5% of revenue.
    oil_sands_extraction_revenue_pct: float = 0.0

    # --- Shale Oil & Gas (PAB only) ---
    # Exploration/production via hydraulic fracturing >= 5% of revenue.
    shale_oil_gas_extraction_revenue_pct: float = 0.0

    # --- Gambling (PAB only) ---
    # Operating gambling establishments >= 10% of revenue.
    gambling_operations_revenue_pct: float = 0.0

    # --- Alcohol (PAB only) ---
    # Production >= 5%, related products/services >= 10%, retail >= 10%.
    alcohol_production_revenue_pct: float = 0.0
    alcohol_related_revenue_pct: float = 0.0
    alcohol_retail_revenue_pct: float = 0.0

    # --- Sustainalytics coverage flag ---
    # Without coverage, business activity screens cannot be applied, so
    # uncovered companies are excluded as a precaution.
    has_sustainalytics_coverage: bool = True

    # --- Fossil Fuel Revenue (PAB only) ---
    # Ref: §Exclusions Based on Revenue Thresholds in Fossil Fuel Operations
    #
    # These thresholds are tiered by fuel type to reflect each fuel's
    # relative carbon intensity:
    #   Coal:  >= 1%  (most carbon-intensive fuel; near-zero tolerance)
    #   Oil:   >= 10% (high-emission but more widely used in transition)
    #   Gas:   >= 50% (lower-emission fossil fuel; higher tolerance)
    #   Power: >= 50% (fossil-fuel-fired electricity generation)
    coal_revenue_pct: float = 0.0
    oil_revenue_pct: float = 0.0
    natural_gas_revenue_pct: float = 0.0
    power_generation_revenue_pct: float = 0.0


# ---------------------------------------------------------------------------
# Stock model
# ---------------------------------------------------------------------------


class Stock(BaseModel):
    """
    Single security with all fields required for PACT index construction.

    Each Stock carries identification fields (ticker, company_name, country,
    GICS classification), weighting inputs (market_cap_usd, float_ratio),
    ESG and carbon data, and the BusinessActivityExposures used by the
    eligibility filters.

    Design note:
        The total_carbon_intensity property combines Scope 1+2 and Scope 3
        into a single figure because both CTB and PAB methodologies require
        Scope 1+2+3 for the WACI calculation and decarbonization trajectory.
        If Scope 1+2 data is missing, total_carbon_intensity returns None,
        and the stock may be excluded by the carbon coverage check or have
        its intensity imputed during optimization.

    Ref: §Eligibility Criteria (p. 8), §Carbon Emissions Data (p. 11)
    """

    ticker: str
    company_name: str
    country: str
    gics_sector: str
    gics_industry_group: str

    market_cap_usd: float
    float_ratio: float

    # S&P Global ESG Score (0-100). Used in the optimization constraint
    # requiring the optimized portfolio's weighted-average ESG score to
    # meet or exceed the underlying index's score.
    esg_score: Optional[float] = None
    has_esg_coverage: bool = True

    # Trucost carbon intensity data (tCO2e per $M revenue).
    # Scope 1+2 covers direct emissions and purchased energy.
    # Scope 3 covers upstream/downstream value chain emissions.
    # Both scopes are required for the WACI target and decarbonization path.
    scope_1_2_carbon_intensity: Optional[float] = None  # tCO2e / $M revenue
    scope_3_carbon_intensity: Optional[float] = None
    has_carbon_coverage: bool = True

    # Science Based Targets initiative (SBTI) commitment flag.
    # The optimization requires that the weight of SBTI-committed companies
    # in the optimized index >= 120% of their weight in the underlying index.
    # Ref: §SBTI Weight Constraint (p. 15)
    has_sbti_target: bool = False

    # Sustainalytics UNGC compliance status
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT

    # Media and Stakeholder Analysis flag — set by the Index Committee
    # when an MSA alert warrants removal from the index.
    msa_flagged: bool = False

    # Business activity revenue/ownership exposures for exclusion screening
    business_activities: BusinessActivityExposures = Field(
        default_factory=BusinessActivityExposures
    )

    @property
    def float_adjusted_market_cap(self) -> float:
        """
        Float-adjusted market cap = total market cap x free-float ratio.

        Used as the weighting basis for the underlying index. Companies with
        lower free-float ratios (e.g. large insider/government holdings)
        receive proportionally less weight.
        """
        return self.market_cap_usd * self.float_ratio

    @property
    def total_carbon_intensity(self) -> Optional[float]:
        """
        Total carbon intensity = Scope 1+2 + Scope 3 (tCO2e / $M revenue).

        Returns None if Scope 1+2 data is missing (the minimum required data).
        If Scope 3 is missing but Scope 1+2 is available, Scope 3 defaults to
        0.0 so the stock is not penalised for missing value-chain data — the
        imputation in the optimizer handles the rest.

        Ref: §Carbon Emissions Data (p. 11)
        """
        if self.scope_1_2_carbon_intensity is None:
            return None
        s3 = self.scope_3_carbon_intensity or 0.0
        return self.scope_1_2_carbon_intensity + s3


# ---------------------------------------------------------------------------
# Index universe model
# ---------------------------------------------------------------------------


class IndexUniverse(BaseModel):
    """
    Full universe of PACT index candidates.

    Contains the list of all stocks eligible for consideration (before
    exclusion filters) and the underlying benchmark weights derived from
    float-adjusted market capitalisation.

    Design note:
        The model_validator auto-computes underlying_weights from market-cap
        data when no explicit weights are provided. This means callers can
        construct an IndexUniverse with just a list of Stock objects and get
        correct benchmark weights automatically.

    Ref: §Index Construction — Underlying Index (p. 13)
    """

    stocks: list[Stock]
    underlying_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_weights_if_missing(self) -> IndexUniverse:
        """Auto-compute float-adjusted market-cap weights if not provided."""
        if not self.underlying_weights:
            self._compute_underlying_weights()
        return self

    def _compute_underlying_weights(self) -> None:
        """
        Derive benchmark weights from float-adjusted market capitalisation.

        Each stock's weight = its float-adjusted market cap / total float-
        adjusted market cap across all stocks in the universe. This produces
        a capitalisation-weighted benchmark that the optimizer will track.
        """
        total = sum(s.float_adjusted_market_cap for s in self.stocks)
        if total == 0:
            raise ValueError("Total float-adjusted market cap is zero.")
        self.underlying_weights = {
            s.ticker: s.float_adjusted_market_cap / total for s in self.stocks
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten the universe into a pandas DataFrame for vectorised
        eligibility filtering and optimization.

        Each row represents one stock. Business activity exposures are
        unpacked from the nested model into top-level columns so that
        the eligibility module can apply simple boolean masks without
        accessing nested attributes.

        The resulting DataFrame has one column per field needed by
        eligibility.py (exclusion screens) and optimization.py
        (carbon, ESG, SBTI, country, sector data).
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
                "float_adjusted_market_cap": s.float_adjusted_market_cap,
                "esg_score": s.esg_score,
                "has_esg_coverage": s.has_esg_coverage,
                "scope_1_2_carbon_intensity": s.scope_1_2_carbon_intensity,
                "scope_3_carbon_intensity": s.scope_3_carbon_intensity,
                "total_carbon_intensity": s.total_carbon_intensity,
                "has_carbon_coverage": s.has_carbon_coverage,
                "has_sbti_target": s.has_sbti_target,
                "ungc_status": s.ungc_status.value,
                "msa_flagged": s.msa_flagged,
                # Business activity columns — flattened for vectorised filtering
                "has_sustainalytics_coverage": ba.has_sustainalytics_coverage,
                "controversial_weapons_revenue_pct": ba.controversial_weapons_revenue_pct,
                "controversial_weapons_ownership_pct": ba.controversial_weapons_ownership_pct,
                "tobacco_production_revenue_pct": ba.tobacco_production_revenue_pct,
                "tobacco_related_revenue_pct": ba.tobacco_related_revenue_pct,
                "tobacco_retail_revenue_pct": ba.tobacco_retail_revenue_pct,
                "small_arms_civilian_revenue_pct": ba.small_arms_civilian_revenue_pct,
                "small_arms_noncivilian_revenue_pct": ba.small_arms_noncivilian_revenue_pct,
                "small_arms_key_components_revenue_pct": ba.small_arms_key_components_revenue_pct,
                "small_arms_retail_revenue_pct": ba.small_arms_retail_revenue_pct,
                "military_integral_weapons_revenue_pct": ba.military_integral_weapons_revenue_pct,
                "military_weapon_related_revenue_pct": ba.military_weapon_related_revenue_pct,
                "thermal_coal_generation_revenue_pct": ba.thermal_coal_generation_revenue_pct,
                "oil_sands_extraction_revenue_pct": ba.oil_sands_extraction_revenue_pct,
                "shale_oil_gas_extraction_revenue_pct": ba.shale_oil_gas_extraction_revenue_pct,
                "gambling_operations_revenue_pct": ba.gambling_operations_revenue_pct,
                "alcohol_production_revenue_pct": ba.alcohol_production_revenue_pct,
                "alcohol_related_revenue_pct": ba.alcohol_related_revenue_pct,
                "alcohol_retail_revenue_pct": ba.alcohol_retail_revenue_pct,
                "coal_revenue_pct": ba.coal_revenue_pct,
                "oil_revenue_pct": ba.oil_revenue_pct,
                "natural_gas_revenue_pct": ba.natural_gas_revenue_pct,
                "power_generation_revenue_pct": ba.power_generation_revenue_pct,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalance result model
# ---------------------------------------------------------------------------


class RebalanceResult(BaseModel):
    """
    Output of the PACT index optimization pipeline.

    Contains everything needed to reconstruct the rebalanced index:
      - optimized_weights: final ticker -> weight mapping (sums to ~1.0)
      - eligible_tickers:  stocks that passed all exclusion screens
      - excluded_tickers:  ticker -> first exclusion reason for rejected stocks
      - weighted_avg_carbon_intensity:  WACI of the optimized portfolio
      - underlying_weighted_avg_carbon_intensity:  WACI of the parent index
      - waci_reduction_pct:  percentage WACI reduction achieved
      - solver_status:  cvxpy status string (e.g. 'optimal', 'infeasible')
      - relaxation_level:  0 = base constraints; higher = more relaxation applied
      - variant:  'ctb' or 'pab' — which index variant was optimized

    Ref: §Index Construction (pp. 13-16)
    """

    optimized_weights: dict[str, float]
    eligible_tickers: list[str]
    excluded_tickers: dict[str, str]
    weighted_avg_carbon_intensity: float
    underlying_weighted_avg_carbon_intensity: float
    waci_reduction_pct: float
    solver_status: str
    relaxation_level: int
    variant: str
