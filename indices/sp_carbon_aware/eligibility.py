"""
Eligibility filtering for the S&P Carbon Aware Index Series.

Each public function takes the universe DataFrame (from IndexUniverse.to_dataframe())
and returns a boolean Series where True means the stock should be EXCLUDED.

Filters are applied in a fixed sequence (see EXCLUSION_CHECKS). A stock is
recorded against the FIRST filter it fails — subsequent filters are never
evaluated for that stock. This means the order of EXCLUSION_CHECKS matters both
for performance and for the reason label stored in the exclusion log.

Ref: "S&P Carbon Aware Index Series Methodology" (March 2026), §Eligibility Criteria
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# ESG Score Exclusions
# Ref: §Exclusions Based on S&P Global ESG Score (p. 5)
# ---------------------------------------------------------------------------


def exclude_missing_esg(df: pd.DataFrame) -> pd.Series:
    """
    Exclude all companies without an S&P Global ESG score.

    A company is considered uncovered if either:
      - has_esg_coverage is False (Sustainalytics has not assessed it), OR
      - esg_score is NaN (assessed but score is unavailable)

    This check runs first in the pipeline, before the quartile screen, so that
    companies with no score don't distort the percentile threshold calculation.

    Applies to both Developed and Emerging variants.
    Ref: §Exclusions Based on S&P Global ESG Score (p. 5)
    """
    # Missing coverage flag OR null score value → exclude
    return ~df["has_esg_coverage"] | df["esg_score"].isna()


def exclude_bottom_quartile_esg(
    df: pd.DataFrame,
    universe_type: str = "developed",
) -> pd.Series:
    """
    Exclude companies whose ESG score falls strictly below the 25th percentile
    of their GICS Industry Group.

    The percentile threshold is computed independently per industry group so
    that the screen is sector-relative — a score of 40 may be adequate in
    Energy but poor in Technology.

    Developed variant:
        The 25th percentile is ideally computed against the full global
        universe (S&P Global LargeMidCap + S&P Global 1200). In this
        implementation we approximate using only the stocks present in the
        input DataFrame. Companies with no ESG coverage were already removed
        by exclude_missing_esg, so they do not appear here.

    Emerging variant:
        The 25th percentile is computed within the underlying index's own
        GICS Industry Group. The methodology explicitly states that companies
        without ESG scores are treated as worst performers and are always
        excluded — so they are flagged unconditionally before the percentile
        threshold is computed for the remaining scored stocks.

    We use a strict less-than comparison (score < threshold) rather than <=
    so that a stock sitting exactly on the 25th percentile boundary is
    retained. This also prevents single-stock industry groups from being
    self-excluded (a stock is never strictly below its own score).

    Ref: §Exclusions Based on S&P Global ESG Score (p. 5)
    """
    # Start with no exclusions; we'll flag rows as we process each group
    exclude = pd.Series(False, index=df.index)

    for group, group_df in df.groupby("gics_industry_group"):
        if universe_type == "emerging":
            # Step 1 (Emerging only): unconditionally exclude companies with no
            # ESG score. The methodology says these count as the worst performers
            # in their group regardless of how few or many scored peers exist.
            no_score_mask = ~group_df["has_esg_coverage"] | group_df["esg_score"].isna()
            exclude.loc[group_df.index[no_score_mask]] = True

            # Step 2: compute threshold only from the companies that do have scores
            scored = group_df.loc[~no_score_mask, "esg_score"]
            if scored.empty:
                # Every company in this group had no score → all already flagged
                continue

            threshold = scored.quantile(0.25)
            # Strictly below the 25th-percentile value → bottom quartile
            bottom_25 = scored.index[scored < threshold]

        else:  # developed
            # For the Developed variant, companies with missing scores were
            # already removed by exclude_missing_esg, so we only see scored stocks here.
            scored = group_df.loc[group_df["has_esg_coverage"], "esg_score"].dropna()
            if scored.empty:
                continue

            # 25th percentile of the scored companies in this industry group
            threshold = scored.quantile(0.25)
            # Strictly below the threshold → worst 25%
            bottom_25 = scored.index[scored < threshold]

        exclude.loc[bottom_25] = True

    return exclude


# ---------------------------------------------------------------------------
# Business Activity Exclusions
# Ref: §Exclusions Based on Business Activities (pp. 5-6)
#
# The methodology distinguishes two exclusion triggers:
#
#   1. "Level of Involvement" (direct exposure) — the company's own revenue %
#      from the activity. Thresholds vary by activity and sub-category:
#        > 0%  means ANY revenue from that activity triggers exclusion
#        ≥ 5%  means the company must derive at least 5% of revenue
#        ≥10%  means at least 10% of revenue
#
#   2. "Significant Ownership" — the company indirectly participates by owning
#      ≥10% of a subsidiary that has direct involvement. Only Controversial
#      Weapons uses this trigger; all other screens are revenue-only.
#
# _exceeds_threshold implements the > 0% / > X% checks (strictly greater than).
# _meets_or_exceeds implements the ≥ 5% / ≥10% checks (greater than or equal).
# ---------------------------------------------------------------------------


def _exceeds_threshold(series: pd.Series, threshold: float) -> pd.Series:
    """
    Return True where a revenue/involvement percentage STRICTLY EXCEEDS the threshold.

    Used for:
      - "> 0%" rules — any involvement at all triggers exclusion (e.g. thermal coal,
        oil & gas production, controversial weapons direct involvement)
    """
    return series > threshold


def _meets_or_exceeds(series: pd.Series, threshold: float) -> pd.Series:
    """
    Return True where a revenue/involvement percentage is >= threshold.

    Used for:
      - "≥ 5%" and "≥ 10%" rules — only material involvement triggers exclusion
        (e.g. tobacco retail, gambling operations, alcoholic beverages production)
    """
    return series >= threshold


def exclude_no_sustainalytics_coverage(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies that Sustainalytics has not assessed for business activities.

    Without coverage we cannot determine whether a company is involved in any
    of the screened activities, so it is excluded from the index as a precaution.
    This is listed as a standalone bullet in the methodology before the individual
    activity thresholds.

    Ref: §Exclusions Based on Business Activities (p. 5) — 'companies without coverage'
    """
    # has_sustainalytics_coverage == False → exclude
    return ~df["has_sustainalytics_coverage"]


def exclude_controversial_weapons(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any involvement in controversial weapons
    (cluster munitions, anti-personnel mines, biological/chemical/nuclear/
    depleted uranium weapons, and white phosphorus weapons).

    The methodology splits involvement into two categories, each with its own
    triggers for direct revenue exposure and indirect ownership:

    Tailor-Made and Essential:
        The company makes the core weapon system or tailor-made components.
        Excluded if: direct revenue > 0%  OR  significant ownership ≥ 10%

    Non Tailor-Made or Non-Essential:
        The company supplies non-essential components or generic services.
        Excluded if: direct revenue > 0%  OR  significant ownership ≥ 10%

    Both categories share the same thresholds, but are tracked separately in
    Sustainalytics data so we test them independently and OR the results.

    Ref: §Exclusions Based on Business Activities, Controversial Weapons row (pp. 5-6)
    """
    # Tailor-made/essential: any direct revenue exposure triggers exclusion
    tailor_made_direct = _exceeds_threshold(
        df["controversial_weapons_tailor_made_essential_pct"], 0.0
    )
    # Tailor-made/essential: owning ≥10% of a company with involvement also excludes
    tailor_made_ownership = _meets_or_exceeds(
        df["controversial_weapons_tailor_made_essential_ownership_pct"], 10.0
    )
    # Non-tailor-made: same thresholds, tested on the separate non-tailor column
    non_tailor_direct = _exceeds_threshold(
        df["controversial_weapons_non_tailor_made_pct"], 0.0
    )
    non_tailor_ownership = _meets_or_exceeds(
        df["controversial_weapons_non_tailor_made_ownership_pct"], 10.0
    )

    # Excluded if ANY of the four conditions is met
    return (
        tailor_made_direct
        | tailor_made_ownership
        | non_tailor_direct
        | non_tailor_ownership
    )


def exclude_tobacco(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in tobacco products.

    The methodology applies different revenue thresholds by sub-category
    to reflect that producing tobacco is worse than merely retailing it:

    Production (manufacturing tobacco products):
        Any revenue at all → excluded  (> 0%)
        Rationale: direct manufacture is unambiguously a core activity.

    Related Products/Services (tobacco-specific inputs to manufacturers):
        Material revenue → excluded  (≥ 5%)
        Rationale: suppliers are meaningfully linked but not core producers.

    Retail (distributing/selling tobacco products to consumers):
        Material revenue → excluded  (≥ 5%)
        Rationale: a general retailer with a small tobacco section is not excluded,
        but one deriving 5%+ of revenue from tobacco sales is.

    Ref: §Exclusions Based on Business Activities, Tobacco Products row (pp. 6-7)
    """
    # Any tobacco manufacturing revenue → exclude immediately
    production = _exceeds_threshold(df["tobacco_production_revenue_pct"], 0.0)
    # Tobacco-related supply chain: ≥5% revenue threshold
    related = _meets_or_exceeds(df["tobacco_related_revenue_pct"], 5.0)
    # Tobacco retail: ≥5% revenue threshold
    retail = _meets_or_exceeds(df["tobacco_retail_revenue_pct"], 5.0)
    return production | related | retail


def exclude_thermal_coal(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in thermal coal (coal used to generate electricity
    or heat, as opposed to metallurgical/coking coal used in steelmaking).

    Both sub-categories use a > 0% threshold — any involvement excludes the company:

    Extraction: mining or quarrying thermal coal.
    Power Generation: burning thermal coal to produce electricity.

    The zero-tolerance threshold reflects the index's goal of minimising GHG
    emissions; thermal coal is one of the highest-emission energy sources.

    Ref: §Exclusions Based on Business Activities, Thermal Coal row (pp. 6-7)
    """
    # Any thermal coal mining revenue → exclude
    extraction = _exceeds_threshold(df["thermal_coal_extraction_revenue_pct"], 0.0)
    # Any revenue from coal-fired power generation → exclude
    power_gen = _exceeds_threshold(
        df["thermal_coal_power_generation_revenue_pct"], 0.0
    )
    return extraction | power_gen


def exclude_oil_sands(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies that extract oil sands (bituminous sands or tar sands).

    Oil sands extraction produces significantly higher lifecycle GHG emissions
    than conventional oil, so a zero-tolerance threshold is applied.
    Any revenue from extraction → excluded  (> 0%).

    Ref: §Exclusions Based on Business Activities, Oil Sands row (pp. 6-7)
    """
    return _exceeds_threshold(df["oil_sands_extraction_revenue_pct"], 0.0)


def exclude_shale_energy(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in shale energy (tight oil / shale gas) exploration
    or production (hydraulic fracturing / fracking).

    Any revenue from shale energy extraction → excluded  (> 0%).

    Ref: §Exclusions Based on Business Activities, Shale Energy row (pp. 6-7)
    """
    return _exceeds_threshold(df["shale_energy_extraction_revenue_pct"], 0.0)


def exclude_arctic_oil_gas(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in oil and gas exploration in Arctic regions.

    Arctic drilling carries heightened environmental risk (fragile ecosystem,
    spill remediation difficulty) in addition to GHG concerns. Any revenue
    from Arctic oil & gas activity → excluded  (> 0%).

    Ref: §Exclusions Based on Business Activities, Arctic Oil & Gas row (pp. 6-7)
    """
    return _exceeds_threshold(df["arctic_oil_gas_extraction_revenue_pct"], 0.0)


def exclude_oil_gas(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in oil and gas.

    This is the broadest fossil-fuel screen and covers the full value chain:

    Production (E&P — exploration, production, refining, transportation, storage):
        Any revenue → excluded  (> 0%)
        Rationale: upstream and midstream oil & gas are the primary emissions source.

    Generation (electricity from oil or gas):
        Any revenue → excluded  (> 0%)
        Rationale: gas-fired power is lower-emission than coal but still a
        significant GHG contributor.

    Supporting Products/Services (tailor-made services for oil & gas operations):
        ≥ 10% of revenue → excluded
        Rationale: a company earning a small proportion of revenue from generic
        support services is not considered materially linked to the industry,
        but one with ≥10% clearly derives significant value from it.

    Ref: §Exclusions Based on Business Activities, Oil & Gas row (pp. 6-7)
    """
    # Any upstream oil & gas revenue (exploration through storage)
    production = _exceeds_threshold(df["oil_gas_production_revenue_pct"], 0.0)
    # Any revenue from oil/gas-fired electricity generation
    generation = _exceeds_threshold(df["oil_gas_generation_revenue_pct"], 0.0)
    # Tailor-made O&G support services: only material exposure excluded (≥10%)
    supporting = _meets_or_exceeds(df["oil_gas_supporting_revenue_pct"], 10.0)
    return production | generation | supporting


def exclude_gambling(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in gambling with material revenue exposure.

    Unlike fossil fuels, gambling is not zero-tolerance — incidental involvement
    (e.g. a hotel with a small casino floor) is permitted. The thresholds reflect
    how directly a company participates in gambling:

    Operations (owns/operates casinos or gambling establishments):
        ≥ 5% of revenue → excluded
        Rationale: operating a gambling business is the most direct form of involvement.

    Specialized Equipment (manufactures equipment used exclusively for gambling):
        ≥ 10% of revenue → excluded
        Rationale: higher threshold because equipment manufacturers are one step
        removed from actual gambling activity.

    Supporting Products/Services (provides supporting services to gambling operators):
        ≥ 10% of revenue → excluded
        Rationale: same logic as equipment — indirect involvement warrants a
        higher materiality threshold.

    Ref: §Exclusions Based on Business Activities, Gambling row (pp. 6-7)
    """
    # Operating gambling establishments: ≥5% threshold
    operations = _meets_or_exceeds(df["gambling_operations_revenue_pct"], 5.0)
    # Manufacturing gambling-only equipment: ≥10% threshold
    equipment = _meets_or_exceeds(df["gambling_equipment_revenue_pct"], 10.0)
    # General gambling support services: ≥10% threshold
    supporting = _meets_or_exceeds(df["gambling_supporting_revenue_pct"], 10.0)
    return operations | equipment | supporting


def exclude_adult_entertainment(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in adult entertainment.

    Production (producing adult content or operating adult entertainment venues):
        Any revenue → excluded  (> 0%)
        Rationale: producing adult content is the core activity; no level of
        involvement is considered acceptable for the index.

    Distribution (distributing adult entertainment materials):
        ≥ 5% of revenue → excluded
        Rationale: distributors are one step removed from production, so a
        materiality threshold is applied — a general media company that happens
        to distribute some adult content is not excluded unless it is a
        meaningful portion of the business.

    Ref: §Exclusions Based on Business Activities, Adult Entertainment row (pp. 6-7)
    """
    # Any involvement in producing adult content → exclude immediately
    production = _exceeds_threshold(
        df["adult_entertainment_production_revenue_pct"], 0.0
    )
    # Distribution: only excluded if it accounts for ≥5% of revenue
    distribution = _meets_or_exceeds(
        df["adult_entertainment_distribution_revenue_pct"], 5.0
    )
    return production | distribution


def exclude_alcoholic_beverages(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material involvement in alcoholic beverages.

    Like gambling, alcoholic beverages are not zero-tolerance — grocery stores
    that sell alcohol are not excluded. Thresholds are set so that only companies
    where alcohol is a meaningful revenue driver are screened out:

    Production (manufacturing alcoholic beverages):
        ≥ 5% of revenue → excluded
        Rationale: producing alcohol is the most direct activity; the ≥5% bar
        excludes dedicated brewers, distillers, and wineries.

    Retail (distributing or retailing alcoholic beverages):
        ≥ 10% of revenue → excluded
        Rationale: a general retailer that stocks alcohol is permitted; a
        retailer where alcohol makes up ≥10% of sales is excluded.

    Related Products/Services (supplies to alcoholic beverage manufacturers):
        ≥ 10% of revenue → excluded
        Rationale: suppliers that are heavily dependent on the alcohol industry
        are materially linked to it.

    Ref: §Exclusions Based on Business Activities, Alcoholic Beverages row (pp. 6-7)
    """
    # Alcohol manufacturing: ≥5% revenue threshold
    production = _meets_or_exceeds(
        df["alcoholic_beverages_production_revenue_pct"], 5.0
    )
    # Alcohol retail/distribution: ≥10% revenue threshold
    retail = _meets_or_exceeds(df["alcoholic_beverages_retail_revenue_pct"], 10.0)
    # Alcohol-related supply chain: ≥10% revenue threshold
    related = _meets_or_exceeds(
        df["alcoholic_beverages_related_revenue_pct"], 10.0
    )
    return production | retail | related


# ---------------------------------------------------------------------------
# UNGC / Global Standards Screening Exclusions
# Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 7)
#
# Sustainalytics' Global Standards Screening (GSS) assesses each company's
# adherence to the ten principles of the UN Global Compact (UNGC), which cover
# human rights, labour standards, environmental responsibility, and anti-corruption.
#
# Three possible classifications:
#   Non-Compliant — confirmed violation of UNGC principles → EXCLUDED
#   Watchlist     — potential violation under investigation  → NOT excluded at rebalancing
#                   (the quarterly UNGC review may remove these between rebalancings)
#   Compliant     — no known violations → eligible
#   No Coverage   — Sustainalytics has not assessed the company → EXCLUDED
# ---------------------------------------------------------------------------


def exclude_ungc_non_compliant(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies classified as Non-Compliant with UNGC principles,
    and those for which Sustainalytics provides no GSS coverage.

    Watchlist companies are retained at the semi-annual rebalancing — they
    may be removed by the separate quarterly UNGC eligibility review instead.

    Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 7)
    """
    # Sustainalytics has not assessed this company for UNGC compliance
    no_coverage = df["ungc_status"] == "No Coverage"
    # Company confirmed to violate UNGC principles
    non_compliant = df["ungc_status"] == "Non-Compliant"
    return no_coverage | non_compliant


# ---------------------------------------------------------------------------
# MSA Controversy Exclusions
# Ref: §Controversies: Media and Stakeholder Analysis Overlay (p. 7)
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
# binary `msa_flagged` field on each stock — True means the committee has
# decided to exclude the company.
# ---------------------------------------------------------------------------


def exclude_msa_flagged(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies where the Index Committee has acted on an MSA alert.

    Ref: §Controversies: Media and Stakeholder Analysis Overlay (p. 7)
    """
    return df["msa_flagged"].astype(bool)


# ---------------------------------------------------------------------------
# Combined eligibility pipeline
# ---------------------------------------------------------------------------

# Ordered list of (human-readable reason, filter function) pairs.
# The ORDER IS SIGNIFICANT: each stock is recorded against the first filter it
# fails. If a stock would fail multiple screens, only the earliest one is logged.
# Callers can rely on reason strings being stable identifiers.
EXCLUSION_CHECKS: list[tuple[str, callable]] = [
    ("Missing ESG score", exclude_missing_esg),
    ("Bottom 25% ESG in industry group", exclude_bottom_quartile_esg),
    ("No Sustainalytics coverage", exclude_no_sustainalytics_coverage),
    ("Controversial weapons", exclude_controversial_weapons),
    ("Tobacco", exclude_tobacco),
    ("Thermal coal", exclude_thermal_coal),
    ("Oil sands", exclude_oil_sands),
    ("Shale energy", exclude_shale_energy),
    ("Arctic oil & gas", exclude_arctic_oil_gas),
    ("Oil & gas", exclude_oil_gas),
    ("Gambling", exclude_gambling),
    ("Adult entertainment", exclude_adult_entertainment),
    ("Alcoholic beverages", exclude_alcoholic_beverages),
    ("UNGC non-compliant / no coverage", exclude_ungc_non_compliant),
    ("MSA flagged", exclude_msa_flagged),
]


def apply_eligibility_filters(
    df: pd.DataFrame,
    universe_type: str = "developed",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Apply all eligibility filters in sequence, returning eligible stocks and
    a log of which stocks were excluded and why.

    Parameters
    ----------
    df:
        Full universe DataFrame from IndexUniverse.to_dataframe().
    universe_type:
        'developed' or 'emerging' — controls the ESG quartile screen behaviour.

    Returns
    -------
    eligible_df:
        Subset of df containing only stocks that passed every filter.
    excluded:
        Dict mapping ticker → first exclusion reason for every excluded stock.

    Design notes:
        - `remaining` shrinks after each filter so later filters only evaluate
          stocks that have not yet been excluded. This is intentional: it keeps
          the logic clean and ensures one reason per ticker.
        - The `exclude_bottom_quartile_esg` function requires the universe_type
          argument, so it receives special treatment in the loop. All other
          filters take only the DataFrame.

    Ref: §Eligibility Criteria (pp. 5-7)
    """
    excluded: dict[str, str] = {}
    # Work on a copy so we can safely drop rows without mutating the caller's data
    remaining = df.copy()

    for reason, check_fn in EXCLUSION_CHECKS:
        if remaining.empty:
            # No stocks left to screen — short-circuit the remaining checks
            break

        # Run the filter. The ESG quartile check needs an extra argument;
        # all other checks only need the DataFrame.
        if check_fn is exclude_bottom_quartile_esg:
            mask = check_fn(remaining, universe_type=universe_type)
        else:
            mask = check_fn(remaining)

        # Record every newly-excluded ticker against this reason
        newly_excluded = remaining.loc[mask, "ticker"].tolist()
        for ticker in newly_excluded:
            excluded[ticker] = reason

        # Remove excluded stocks so they don't appear in subsequent filter calls
        remaining = remaining.loc[~mask].copy()

    # `remaining` now contains only stocks that passed every filter
    return remaining, excluded
