"""
Eligibility filtering for the S&P ESG (Scored & Screened) Index Series.

Each public function takes the universe DataFrame (from IndexUniverse.to_dataframe())
and returns a boolean Series where True means the stock should be EXCLUDED.

Filters are applied in a fixed sequence (see EXCLUSION_CHECKS). A stock is
recorded against the FIRST filter it fails -- subsequent filters are never
evaluated for that stock. This means the order of EXCLUSION_CHECKS matters both
for performance and for the reason label stored in the exclusion log.

The S&P ESG Index targets approximately 75% of the float-adjusted market
capitalisation of each GICS Industry Group from the parent index, after removing
companies that fail ESG score screens, business activity screens, UNGC
compliance checks, and MSA controversy flags.

Ref: "S&P ESG Index Series Methodology" (S&P Dow Jones Indices), §Eligibility Criteria
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# ESG Score Exclusions
# Ref: §ESG Score Eligibility
#
# The index removes two categories of companies based on ESG scores:
#
#   1. Companies with no S&P Global ESG score at all (uncovered or NaN).
#      These must be removed first so they do not distort the percentile
#      threshold calculation in the next step.
#
#   2. Companies whose ESG score falls in the bottom 25% of their GICS
#      Industry Group. The percentile is computed per industry group so the
#      screen is sector-relative -- a score of 40 might be acceptable in
#      Energy but poor in Technology.
#
# Together these two screens drive the index toward the target of retaining
# ~75% of each industry group's float-adjusted market capitalisation.
# ---------------------------------------------------------------------------


def exclude_missing_esg(df: pd.DataFrame) -> pd.Series:
    """
    Exclude all companies without an S&P Global ESG score.

    A company is considered uncovered if either:
      - has_esg_coverage is False (S&P Global has not assessed it), OR
      - esg_score is NaN (assessed but score is unavailable)

    This check runs first in the pipeline, before the quartile screen, so that
    companies with no score don't distort the percentile threshold calculation.

    Ref: §ESG Score Eligibility -- companies without an S&P ESG Score are removed.
    """
    # Missing coverage flag OR null score value -> exclude
    return ~df["has_esg_coverage"] | df["esg_score"].isna()


def exclude_bottom_quartile_esg(
    df: pd.DataFrame,
    universe_type: str = "standard",
) -> pd.Series:
    """
    Exclude companies whose ESG score falls strictly below the 25th percentile
    of their GICS Industry Group.

    The percentile threshold is computed independently per industry group so
    that the screen is sector-relative -- a score of 40 may be adequate in
    Energy but poor in Technology. This design ensures each GICS Industry Group
    contributes proportionally to the final index rather than penalising
    inherently lower-scoring sectors.

    We use a strict less-than comparison (score < threshold) rather than <=
    so that a stock sitting exactly on the 25th percentile boundary is
    retained. This also prevents single-stock industry groups from being
    self-excluded (a stock is never strictly below its own score).

    Companies with missing ESG coverage were already removed by
    exclude_missing_esg earlier in the pipeline, so they do not appear here
    and cannot influence the percentile calculation.

    Ref: §ESG Score Eligibility -- remove bottom 25% per GICS Industry Group.
    """
    # Start with no exclusions; we'll flag rows as we process each group
    exclude = pd.Series(False, index=df.index)

    for _group, group_df in df.groupby("gics_industry_group"):
        # Only consider companies that have ESG coverage and a valid score
        scored = group_df.loc[group_df["has_esg_coverage"], "esg_score"].dropna()
        if scored.empty:
            # No scored companies in this group -- nothing to exclude
            continue

        # 25th percentile of the scored companies in this industry group
        threshold = scored.quantile(0.25)
        # Strictly below the threshold -> worst 25%
        bottom_25 = scored.index[scored < threshold]
        exclude.loc[bottom_25] = True

    return exclude


# ---------------------------------------------------------------------------
# Business Activity Exclusions
# Ref: §Business Activity Exclusions
#
# The methodology screens companies by their revenue exposure (and in one
# case, ownership exposure) to specific business activities. Different
# activities carry different thresholds reflecting the index's view of the
# severity and directness of the involvement:
#
#   Controversial Weapons:
#     - Revenue > 0%            (any direct involvement triggers exclusion)
#     - Ownership >= 10%        (indirect involvement via subsidiary)
#
#   Tobacco:
#     - Production > 0%         (any manufacturing revenue triggers exclusion)
#     - Retail >= 10%           (only material retail exposure triggers)
#
#   Thermal Coal:
#     - Extraction > 5%         (allows minor legacy mining exposure)
#     - Power Generation > 25%  (allows diversified utilities with some coal)
#
#   Small Arms (Civilian):
#     - Manufacture >= 5%       (material firearms production triggers)
#     - Retail >= 10%           (only significant retail exposure triggers)
#
# The tiered thresholds reflect the principle that direct production of a
# harmful product is treated more strictly than retail or support activities,
# and that some activities (controversial weapons) warrant zero tolerance
# while others (thermal coal power) permit some diversified exposure.
# ---------------------------------------------------------------------------


def exclude_controversial_weapons(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any involvement in controversial weapons
    (cluster munitions, anti-personnel mines, biological/chemical/nuclear
    weapons for non-treaty states, depleted uranium, blinding laser).

    The methodology applies zero tolerance for direct revenue involvement
    and a 10% ownership threshold for indirect involvement:

    Direct Revenue Exposure:
        Any revenue at all -> excluded  (> 0%)
        Rationale: direct manufacture of or contribution to controversial
        weapons is unambiguously excluded regardless of scale.

    Significant Ownership:
        Owning >= 10% of a company with direct involvement -> excluded
        Rationale: material ownership of a weapons-involved subsidiary
        indicates the parent company profits meaningfully from the activity.

    Ref: §Business Activity Exclusions -- Controversial Weapons
    """
    # Any direct revenue from controversial weapons -> exclude immediately
    revenue = df["controversial_weapons_revenue_pct"] > 0.0
    # Owning >= 10% of an involved subsidiary -> exclude
    ownership = df["controversial_weapons_ownership_pct"] >= 10.0
    return revenue | ownership


def exclude_tobacco(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in tobacco products.

    The methodology applies different thresholds by sub-category to reflect
    that manufacturing tobacco is a more direct activity than retailing it:

    Production (manufacturing tobacco products):
        Any revenue at all -> excluded  (> 0%)
        Rationale: direct manufacture is the core harmful activity; no
        level of involvement is considered acceptable.

    Retail (distributing/selling tobacco products to consumers):
        Material revenue -> excluded  (>= 10%)
        Rationale: a general retailer with a small tobacco section is not
        excluded, but one deriving 10%+ of revenue from tobacco sales is
        considered materially linked to the industry.

    Note: unlike the Carbon Aware index, the ESG index does not screen for
    tobacco-related products/services as a separate sub-category.

    Ref: §Business Activity Exclusions -- Tobacco Products
    """
    # Any tobacco manufacturing revenue -> exclude immediately
    production = df["tobacco_production_revenue_pct"] > 0.0
    # Tobacco retail: >= 10% revenue threshold
    retail = df["tobacco_retail_revenue_pct"] >= 10.0
    return production | retail


def exclude_thermal_coal(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material thermal coal revenue.

    Unlike the Carbon Aware index which applies zero-tolerance to thermal
    coal, the ESG index uses materiality thresholds that permit diversified
    companies with minor legacy coal exposure:

    Extraction (mining or quarrying thermal coal):
        Revenue > 5% from coal mining -> excluded
        Rationale: a diversified mining company with a small thermal coal
        segment is permitted, but one deriving more than 5% of revenue
        from coal extraction is considered materially involved.

    Power Generation (burning thermal coal to produce electricity):
        Revenue > 25% from coal-fired power -> excluded
        Rationale: the higher threshold reflects that many utilities are
        in multi-year transition plans away from coal. A utility generating
        25% or less of revenue from coal power is still considered to be
        transitioning; above that it is materially coal-dependent.

    Ref: §Business Activity Exclusions -- Thermal Coal
    """
    # More than 5% of revenue from thermal coal mining -> exclude
    extraction = df["thermal_coal_extraction_revenue_pct"] > 5.0
    # More than 25% of revenue from coal-fired electricity -> exclude
    power = df["thermal_coal_power_revenue_pct"] > 25.0
    return extraction | power


def exclude_small_arms(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material civilian small-arms involvement.

    This screen targets firearms manufactured for or sold to civilian
    markets (not military or law-enforcement contracts). The distinction
    matters because civilian firearms carry higher societal harm risk
    without the institutional oversight of military procurement:

    Manufacture (producing civilian firearms):
        Revenue >= 5% from civilian firearms production -> excluded
        Rationale: a company deriving 5%+ of revenue from making civilian
        guns is considered a material arms manufacturer.

    Retail (selling civilian firearms to end consumers):
        Revenue >= 10% from civilian firearms retail -> excluded
        Rationale: the higher threshold reflects that retailers are one
        step removed from production; a general sporting-goods store with
        a small firearms counter is not excluded, but a company where
        firearms are 10%+ of revenue is materially dependent on the activity.

    Ref: §Business Activity Exclusions -- Small Arms (Civilian)
    """
    # Material firearms manufacturing: >= 5% threshold
    manufacture = df["small_arms_manufacture_revenue_pct"] >= 5.0
    # Material firearms retail: >= 10% threshold
    retail = df["small_arms_retail_revenue_pct"] >= 10.0
    return manufacture | retail


# ---------------------------------------------------------------------------
# UNGC / Global Standards Screening Exclusions
# Ref: §UN Global Compact Compliance
#
# Sustainalytics' Global Standards Screening (GSS) assesses each company's
# adherence to the ten principles of the UN Global Compact (UNGC), which cover
# human rights, labour standards, environmental responsibility, and
# anti-corruption.
#
# Three possible classifications:
#   Non-Compliant -- confirmed violation of UNGC principles   -> EXCLUDED
#   Watchlist     -- potential violation under investigation    -> NOT excluded
#                    (may be removed between rebalancings by quarterly review)
#   Compliant     -- no known violations                       -> eligible
#   No Coverage   -- Sustainalytics has not assessed the company -> EXCLUDED
# ---------------------------------------------------------------------------


def exclude_ungc_non_compliant(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies classified as Non-Compliant with UNGC principles,
    and those for which Sustainalytics provides no GSS coverage.

    Watchlist companies are retained at the semi-annual rebalancing -- they
    may be removed by the separate quarterly UNGC eligibility review instead.

    Companies with no coverage are excluded as a precaution: without GSS
    assessment, the index cannot determine whether the company violates
    UNGC principles.

    Ref: §UN Global Compact Compliance
    """
    # "Non-Compliant" = confirmed violation; "No Coverage" = unknown -> treat
    # both as ineligible for the index
    return df["ungc_status"].isin(["Non-Compliant", "No Coverage"])


# ---------------------------------------------------------------------------
# MSA Controversy Exclusions
# Ref: §Controversies / Media and Stakeholder Analysis
#
# S&P Global Sustainable1 continuously monitors news and stakeholder sources
# for ESG-related incidents (fraud, environmental disasters, labour disputes,
# human rights violations, etc.). When a risk is identified, they issue a
# Media and Stakeholder Analysis (MSA) report.
#
# The Index Committee reviews each MSA and decides whether to remove the
# company. If removed, the company cannot re-enter the index for at least
# one full calendar year from the following rebalancing.
#
# In this implementation we represent the Index Committee's decision via the
# binary `msa_flagged` field on each stock -- True means the committee has
# decided to exclude the company.
# ---------------------------------------------------------------------------


def exclude_msa_flagged(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies where the Index Committee has acted on an MSA alert.

    The MSA overlay is the final safety net in the eligibility pipeline: even
    if a company passes all quantitative screens, a serious real-world
    controversy can trigger removal at the committee's discretion.

    Ref: §Controversies / Media and Stakeholder Analysis
    """
    return df["msa_flagged"].astype(bool)


# ---------------------------------------------------------------------------
# Combined eligibility pipeline
# ---------------------------------------------------------------------------

# Ordered list of (human-readable reason, filter function) pairs.
# The ORDER IS SIGNIFICANT: each stock is recorded against the first filter it
# fails. If a stock would fail multiple screens, only the earliest one is logged.
# Callers can rely on reason strings being stable identifiers.
#
# The ordering follows the methodology's logical flow:
#   1. ESG score screens (no score, then bottom quartile)
#   2. Business activity screens (weapons, tobacco, coal, small arms)
#   3. Global standards / UNGC compliance
#   4. MSA controversy overlay
EXCLUSION_CHECKS: list[tuple[str, callable]] = [
    ("Missing ESG score", exclude_missing_esg),
    ("Bottom 25% ESG in industry group", exclude_bottom_quartile_esg),
    ("Controversial weapons", exclude_controversial_weapons),
    ("Tobacco", exclude_tobacco),
    ("Thermal coal", exclude_thermal_coal),
    ("Small arms (civilian)", exclude_small_arms),
    ("UNGC non-compliant / no coverage", exclude_ungc_non_compliant),
    ("MSA flagged", exclude_msa_flagged),
]


def apply_eligibility_filters(
    df: pd.DataFrame,
    universe_type: str = "standard",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Apply all eligibility filters in sequence, returning eligible stocks and
    a log of which stocks were excluded and why.

    Parameters
    ----------
    df:
        Full universe DataFrame from IndexUniverse.to_dataframe().
    universe_type:
        'standard' -- default ESG index behaviour. Passed through to the
        ESG quartile screen for potential variant-specific logic.

    Returns
    -------
    eligible_df:
        Subset of df containing only stocks that passed every filter.
    excluded:
        Dict mapping ticker -> first exclusion reason for every excluded stock.

    Design notes:
        - `remaining` shrinks after each filter so later filters only evaluate
          stocks that have not yet been excluded. This is intentional: it keeps
          the logic clean and ensures one reason per ticker.
        - The `exclude_bottom_quartile_esg` function requires the universe_type
          argument, so it receives special treatment in the loop. All other
          filters take only the DataFrame.

    Ref: §Eligibility Criteria
    """
    excluded: dict[str, str] = {}
    # Work on a copy so we can safely drop rows without mutating the caller's data
    remaining = df.copy()

    for reason, check_fn in EXCLUSION_CHECKS:
        if remaining.empty:
            # No stocks left to screen -- short-circuit the remaining checks
            break

        # Run the filter. The ESG quartile check needs an extra argument;
        # all other checks only need the DataFrame.
        if check_fn is exclude_bottom_quartile_esg:
            mask = check_fn(remaining, universe_type=universe_type)
        else:
            mask = check_fn(remaining)

        # Record every newly-excluded ticker against this reason
        for ticker in remaining.loc[mask, "ticker"].tolist():
            excluded[ticker] = reason

        # Remove excluded stocks so they don't appear in subsequent filter calls
        remaining = remaining.loc[~mask].copy()

    # `remaining` now contains only stocks that passed every filter
    return remaining, excluded
