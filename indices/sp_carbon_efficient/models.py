"""
Data models for the S&P Global Carbon Efficient Index Series.

The Carbon Efficient index uses a tilt-based approach rather than exclusions:
every benchmark constituent remains in the index, but each stock's weight is
adjusted according to its carbon intensity relative to its GICS Industry Group
peers. Companies with lower carbon intensity receive a higher weight; companies
with higher carbon intensity receive a lower weight.

This is fundamentally different from the S&P Carbon Aware Index Series, which
hard-excludes companies based on ESG scores, business activities, and UNGC
compliance. Here, no stock is removed — the worst emitters are simply
underweighted rather than excluded entirely.

Key models:
  - Stock: one security with identifiers, market data, and carbon intensity.
  - IndexUniverse: the full benchmark universe with float-cap-weighted positions.
  - RebalanceResult: output container holding tilted weights and WACI diagnostics.

Ref: "S&P Global Carbon Efficient Index Series Methodology" (S&P Dow Jones Indices),
     §Index Construction, §Carbon-to-Revenue Footprint
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Stock model
# Ref: §Constituent Data Requirements
# ---------------------------------------------------------------------------


class Stock(BaseModel):
    """
    Single security with all fields required for the carbon efficiency tilt.

    Each stock carries:
      - Identifiers (ticker, company name, country, GICS classification)
      - Market data (market cap, float ratio) used to compute benchmark weights
      - Carbon intensity from Trucost (optional — None means no coverage)

    The GICS Industry Group is the key grouping dimension: all carbon z-score
    calculations and weight normalisations happen within industry groups to
    ensure the index remains industry-group-neutral.

    Ref: §Carbon-to-Revenue Footprint, §Index Construction
    """

    ticker: str
    company_name: str
    country: str
    gics_sector: str
    gics_industry_group: str

    market_cap_usd: float
    float_ratio: float

    # ---------------------------------------------------------------------------
    # Carbon intensity: tCO2e per USD million of revenue, sourced from Trucost.
    #
    # None indicates the company is not in Trucost coverage. The methodology
    # treats non-disclosed companies as neutral (equivalent to the 4th-7th
    # decile "Not-disclosed" category with a 0% weight adjustment), so they
    # receive a CEF of 1.0 — their underlying weight passes through unchanged.
    #
    # Ref: §Carbon-to-Revenue Footprint, §Decile-Based Carbon Weight Adjustment
    # ---------------------------------------------------------------------------
    carbon_intensity: Optional[float] = None

    @property
    def float_adjusted_market_cap(self) -> float:
        """
        Float-adjusted market capitalisation used for benchmark weight calculation.

        Formula: market_cap_usd * float_ratio

        The float ratio represents the fraction of shares available for public
        trading. Multiplying by market cap gives the investable (free-float)
        capitalisation, which determines each stock's weight in the underlying
        benchmark index.

        Ref: §Underlying Index Weighting
        """
        return self.market_cap_usd * self.float_ratio


# ---------------------------------------------------------------------------
# Universe model
# Ref: §Index Construction — Underlying Index
# ---------------------------------------------------------------------------


class IndexUniverse(BaseModel):
    """
    Full universe of benchmark stocks with float-cap weights.

    The universe represents the underlying index (e.g. S&P Global LargeMidCap)
    before any carbon tilt is applied. Each stock's underlying weight is derived
    from its float-adjusted market capitalisation relative to the total.

    If underlying_weights are not supplied at construction time, they are
    automatically computed from the stocks' float-adjusted market caps via the
    model_validator. This allows callers to either:
      (a) supply pre-computed weights (e.g. from a benchmark data feed), or
      (b) let the model derive them from market_cap_usd and float_ratio.

    Design note:
        Using a Pydantic model_validator (mode="after") ensures weights are
        always available after construction, regardless of which path was taken.
        This avoids None-check boilerplate throughout the weighting and
        rebalancing modules.

    Ref: §Underlying Index Weighting
    """

    stocks: list[Stock]
    underlying_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def compute_weights_if_missing(self) -> IndexUniverse:
        """
        Auto-compute float-cap weights if none were provided.

        Only runs when underlying_weights is empty (default). If the caller
        supplies explicit weights, they are used as-is without validation —
        this supports testing and scenarios where benchmark weights come from
        an external source.
        """
        if not self.underlying_weights:
            self._compute_underlying_weights()
        return self

    def _compute_underlying_weights(self) -> None:
        """
        Derive each stock's underlying weight from float-adjusted market cap.

        Formula:  W_i = FAM_i / sum(FAM_j)  for all stocks j in the universe

        Raises ValueError if the total float-adjusted market cap is zero, which
        would indicate either an empty universe or corrupted data (all stocks
        have zero market cap or zero float ratio).
        """
        total = sum(s.float_adjusted_market_cap for s in self.stocks)
        if total == 0:
            raise ValueError("Total float-adjusted market cap is zero.")
        self.underlying_weights = {
            s.ticker: s.float_adjusted_market_cap / total for s in self.stocks
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten stocks to a DataFrame with underlying weights attached.

        Returns a DataFrame with one row per stock, containing all Stock fields
        plus the computed underlying_weight. This is the standard input format
        for the weighting module's compute_carbon_efficiency_factors() and
        apply_tilt() functions.

        The underlying_weight column is included so that downstream code can
        access both stock metadata and weights from a single DataFrame without
        needing a separate lookup into the underlying_weights dict.
        """
        rows = []
        for s in self.stocks:
            rows.append({
                "ticker": s.ticker,
                "company_name": s.company_name,
                "country": s.country,
                "gics_sector": s.gics_sector,
                "gics_industry_group": s.gics_industry_group,
                "market_cap_usd": s.market_cap_usd,
                "float_ratio": s.float_ratio,
                "underlying_weight": self.underlying_weights[s.ticker],
                "carbon_intensity": s.carbon_intensity,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rebalance result
# Ref: §Index Construction — Output
# ---------------------------------------------------------------------------


class RebalanceResult(BaseModel):
    """
    Output of the carbon efficiency tilt rebalance.

    Contains:
      - tilted_weights: the final carbon-adjusted portfolio weights (ticker -> weight).
        These sum to 1.0 and represent the index after the carbon tilt has been applied.
      - weighted_avg_carbon_intensity: the post-tilt WACI, measuring how carbon-
        intensive the tilted portfolio is.
      - underlying_waci: the pre-tilt WACI of the original benchmark, used as
        the baseline for measuring the tilt's effectiveness.
      - waci_reduction_pct: the percentage reduction in WACI achieved by the tilt,
        computed as (1 - post_waci / pre_waci) * 100. Positive values mean the
        tilt successfully lowered carbon intensity.

    Ref: §Index Construction, §WACI Reporting
    """

    tilted_weights: dict[str, float]       # ticker -> final carbon-tilted weight
    weighted_avg_carbon_intensity: float    # post-tilt WACI
    underlying_waci: float                 # pre-tilt WACI (benchmark)
    waci_reduction_pct: float              # (1 - post/pre) * 100
