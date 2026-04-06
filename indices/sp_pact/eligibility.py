"""
Eligibility filtering for the S&P PAB ESG & S&P CTB Indices.

Each public function takes the universe DataFrame (from IndexUniverse.to_dataframe())
and returns a boolean Series where True means the stock should be EXCLUDED.

The PACT methodology defines two tiers of exclusions:

  1. Core exclusions (shared by both CTB and PAB):
     - Missing carbon data (Scope 1+2+3 required)
     - Controversial weapons (> 0% revenue OR >= 25% ownership)
     - Tobacco (production > 0%, related >= 10%, retail varies by variant)
     - UNGC non-compliant or no Sustainalytics coverage
     - MSA flagged by Index Committee

  2. PAB-only exclusions (stricter, aligned with Paris Agreement 1.5 C target):
     - Small arms (all sub-categories > 0%)
     - Military contracting (integral > 0%, weapon-related >= 5%)
     - Thermal coal power generation (>= 5%)
     - Oil sands extraction (>= 5%)
     - Shale oil & gas extraction (>= 5%)
     - Gambling operations (>= 10%)
     - Alcohol (production >= 5%, related >= 10%, retail >= 10%)
     - Fossil fuel revenue (coal >= 1%, oil >= 10%, gas >= 50%, power >= 50%)

Filters are applied in a fixed sequence (see get_exclusion_checks). A stock is
recorded against the FIRST filter it fails -- subsequent filters are never
evaluated for that stock. This means the order matters both for performance
and for the reason label stored in the exclusion log.

Ref: "S&P PAB ESG and S&P CTB Indices Methodology" (S&P Dow Jones Indices)
     §Eligibility Criteria (pp. 8-9), §Index Exclusions (pp. 9-11)
"""

from __future__ import annotations

import pandas as pd

from .models import Variant


# ---------------------------------------------------------------------------
# Exclusion functions -- shared (CTB + PAB)
#
# These core screens apply to BOTH index variants. They represent the
# minimum exclusion standard that all PACT indices must meet.
# Ref: §Business Activity Exclusions — Core Screens (pp. 9-10)
# ---------------------------------------------------------------------------


def exclude_no_carbon_coverage(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies without Scope 1, 2, and 3 GHG emissions data.

    Unlike the Carbon Aware index (which only needs a single carbon intensity
    figure), the PACT methodology requires full Scope 1+2+3 data because
    both the WACI reduction target and the 7% annual decarbonization trajectory
    are computed on total (Scope 1+2+3) carbon intensity.

    Companies without coverage cannot contribute to the WACI calculation
    and would undermine the decarbonization target, so they are excluded
    upfront before any business-activity screening.

    Ref: §Eligibility Factors — Carbon Emissions Coverage (p. 8)
    """
    return ~df["has_carbon_coverage"]


def exclude_controversial_weapons(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any involvement in controversial weapons (CTB + PAB).

    Controversial weapons include cluster munitions, anti-personnel mines,
    biological/chemical/nuclear weapons, and depleted uranium weapons.

    Two independent triggers:
      - Revenue > 0%: the company derives ANY direct revenue from controversial
        weapons manufacturing, components, or services.
      - Ownership >= 25%: the company owns >= 25% of a subsidiary involved in
        controversial weapons. This threshold is higher than the Carbon Aware
        index (which uses >= 10%) because the PACT methodology defines
        "significant ownership" at the 25% level.

    Ref: §Business Activity Exclusions — Controversial Weapons (p. 9)
    """
    # Any direct revenue from controversial weapons triggers exclusion
    revenue = df["controversial_weapons_revenue_pct"] > 0.0
    # Significant ownership stake (>= 25%) in an involved subsidiary
    ownership = df["controversial_weapons_ownership_pct"] >= 25.0
    return revenue | ownership


def exclude_tobacco(df: pd.DataFrame, variant: Variant) -> pd.Series:
    """
    Exclude companies involved in tobacco products (CTB + PAB).

    The methodology applies different thresholds by sub-category and variant
    to reflect the severity of each type of involvement:

    Production (manufacturing tobacco products):
        Any revenue at all -> excluded (> 0%)
        Rationale: direct manufacture is the core activity; zero tolerance.

    Related Products/Services (tobacco-specific inputs to manufacturers):
        >= 10% of revenue -> excluded
        Rationale: suppliers with material dependency on the tobacco industry.

    Retail (distributing/selling tobacco products to consumers):
        PAB: >= 5% of revenue -> excluded  (stricter, Paris-aligned)
        CTB: >= 10% of revenue -> excluded (broader inclusion)
        Rationale: the PAB variant casts a wider net to exclude retailers
        with moderate tobacco exposure, while CTB only excludes those
        heavily dependent on tobacco sales.

    Ref: §Business Activity Exclusions — Tobacco (p. 9)
    """
    # Any tobacco manufacturing revenue -> exclude immediately
    production = df["tobacco_production_revenue_pct"] > 0.0
    # Tobacco supply chain: >= 10% revenue threshold
    related = df["tobacco_related_revenue_pct"] >= 10.0
    # Retail threshold varies by variant
    if variant == Variant.PAB:
        # PAB is stricter: 5% threshold catches more retailers
        retail = df["tobacco_retail_revenue_pct"] >= 5.0
    else:
        # CTB uses the standard 10% threshold
        retail = df["tobacco_retail_revenue_pct"] >= 10.0
    return production | related | retail


def exclude_ungc_non_compliant(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies classified as UNGC Non-Compliant, and those for which
    Sustainalytics provides no GSS coverage.

    Sustainalytics' Global Standards Screening assesses adherence to the ten
    principles of the UN Global Compact. The PACT methodology treats both
    Non-Compliant and No Coverage companies as ineligible:

      Non-Compliant: confirmed violation of UNGC principles (human rights,
                     labour, environment, or anti-corruption).
      No Coverage:   Sustainalytics has not assessed the company, so
                     compliance cannot be confirmed -- excluded as precaution.

    Watchlist companies (potential violations under investigation) are retained
    at rebalancing but may be removed in quarterly UNGC reviews.

    Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 10)
    """
    # Non-Compliant or No Coverage status -> exclude
    ungc_fail = df["ungc_status"].isin(["Non-Compliant", "No Coverage"])
    # Also exclude if Sustainalytics has no coverage at all (belt-and-suspenders)
    no_coverage = ~df["has_sustainalytics_coverage"]
    return ungc_fail | no_coverage


def exclude_msa_flagged(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies where the Index Committee has acted on an MSA alert.

    S&P Global Sustainable1 continuously monitors news and stakeholder sources
    for ESG incidents. When a Media and Stakeholder Analysis (MSA) report is
    issued, the Index Committee reviews it and may remove the company. Once
    removed, the company cannot re-enter for at least one full calendar year
    from the following rebalancing.

    In this implementation, the binary msa_flagged field represents the
    committee's decision -- True means the company has been flagged for removal.

    Ref: §Controversies Monitoring: Media and Stakeholder Analysis Overlay (p. 10)
    """
    return df["msa_flagged"].astype(bool)


# ---------------------------------------------------------------------------
# Exclusion functions -- PAB only
#
# These additional screens apply exclusively to the PAB variant.
# They reflect the stricter Paris Agreement alignment requirements,
# covering weapons, fossil fuels, and sin stocks that CTB permits.
# Ref: §Additional PAB Exclusions (pp. 10-11)
# ---------------------------------------------------------------------------


def exclude_small_arms(df: pd.DataFrame) -> pd.Series:
    """
    Exclude any involvement in small arms (PAB only).

    The PAB variant applies zero-tolerance (> 0%) across all sub-categories
    of small arms involvement:

    Civilian firearms:       manufacturing firearms for civilian use
    Non-civilian firearms:   manufacturing firearms for military/law enforcement
    Key components:          manufacturing essential components (e.g. triggers,
                            barrels, ammunition) for small arms
    Retail:                  distributing or selling small arms to end users

    Unlike the Carbon Aware index, which does not screen for small arms at all,
    the PAB methodology treats any firearms involvement as incompatible with
    Paris-aligned investing.

    Ref: §Business Activity Exclusions — Small Arms (PAB) (p. 10)
    """
    return (
        (df["small_arms_civilian_revenue_pct"] > 0.0) |
        (df["small_arms_noncivilian_revenue_pct"] > 0.0) |
        (df["small_arms_key_components_revenue_pct"] > 0.0) |
        (df["small_arms_retail_revenue_pct"] > 0.0)
    )


def exclude_military_contracting(df: pd.DataFrame) -> pd.Series:
    """
    Exclude military contracting involvement (PAB only).

    Two categories with different thresholds:

    Integral weapons systems (> 0%):
        Companies that manufacture complete weapon systems (tanks, fighter
        jets, warships, missile systems). Any revenue triggers exclusion
        because these are the core product of military industry.

    Weapon-related products/services (>= 5%):
        Companies providing tailor-made components, maintenance, or support
        services specifically for weapons platforms. A 5% materiality
        threshold is applied because some diversified industrials may have
        minor military contracts alongside their civilian business.

    Ref: §Business Activity Exclusions — Military Contracting (PAB) (p. 10)
    """
    # Manufacturing complete weapon systems: zero tolerance
    integral = df["military_integral_weapons_revenue_pct"] > 0.0
    # Weapon-related support services: materiality threshold of 5%
    weapon_related = df["military_weapon_related_revenue_pct"] >= 5.0
    return integral | weapon_related


def exclude_thermal_coal(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies deriving >= 5% of revenue from thermal coal power
    generation (PAB only).

    Unlike the Carbon Aware index (which uses > 0% zero-tolerance for both
    extraction and generation), the PACT PAB variant sets the threshold at
    >= 5%, allowing de minimis coal exposure for companies transitioning
    away from coal. This reflects the EU TEG's guidance that PAB indices
    should exclude companies with "significant" rather than "any" coal
    power generation.

    Note: this screen covers only power generation from thermal coal, not
    coal mining/extraction (which is covered by the fossil fuel revenue
    screen with a >= 1% threshold).

    Ref: §Business Activity Exclusions — Thermal Coal (PAB) (p. 10)
    """
    return df["thermal_coal_generation_revenue_pct"] >= 5.0


def exclude_oil_sands(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies deriving >= 5% of revenue from oil sands (tar sands)
    extraction (PAB only).

    Oil sands (bituminous sands) extraction produces significantly higher
    lifecycle GHG emissions than conventional oil production due to the
    energy-intensive extraction and upgrading processes. The 5% threshold
    excludes companies with material oil sands operations while permitting
    those with only incidental exposure.

    Ref: §Business Activity Exclusions — Oil Sands (PAB) (p. 10)
    """
    return df["oil_sands_extraction_revenue_pct"] >= 5.0


def exclude_shale_oil_gas(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies deriving >= 5% of revenue from shale oil & gas
    extraction via hydraulic fracturing (PAB only).

    Shale/tight oil and gas extraction (fracking) carries both direct GHG
    emissions and methane leakage concerns. The 5% materiality threshold
    mirrors the oil sands screen, allowing de minimis exposure for
    diversified energy companies in transition.

    Ref: §Business Activity Exclusions — Shale Oil & Gas (PAB) (p. 10)
    """
    return df["shale_oil_gas_extraction_revenue_pct"] >= 5.0


def exclude_gambling(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies deriving >= 10% of revenue from gambling operations
    (PAB only).

    Only direct operations (owning/operating casinos or gambling establishments)
    are screened. Unlike the Carbon Aware index (which also screens equipment
    and supporting services at different thresholds), the PACT PAB variant
    uses a single 10% threshold on operations only. This reflects a
    focus on companies where gambling is a core business activity.

    Ref: §Business Activity Exclusions — Gambling (PAB) (p. 10)
    """
    return df["gambling_operations_revenue_pct"] >= 10.0


def exclude_alcohol(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material involvement in alcoholic beverages
    (PAB only).

    Three sub-categories with tiered thresholds:

    Production (manufacturing alcoholic beverages):
        >= 5% of revenue -> excluded
        Rationale: producing alcohol is the most direct involvement. The
        5% bar excludes dedicated brewers, distillers, and wineries.

    Related Products/Services (supplies to alcohol manufacturers):
        >= 10% of revenue -> excluded
        Rationale: suppliers heavily dependent on the alcohol industry are
        materially linked but one step removed from direct production.

    Retail (distributing or retailing alcoholic beverages):
        >= 10% of revenue -> excluded
        Rationale: a general retailer stocking alcohol is permitted; one
        where alcohol makes up >= 10% of sales is excluded.

    Ref: §Business Activity Exclusions — Alcohol (PAB) (p. 10)
    """
    # Alcohol manufacturing: >= 5% revenue threshold
    production = df["alcohol_production_revenue_pct"] >= 5.0
    # Alcohol-related supply chain: >= 10% revenue threshold
    related = df["alcohol_related_revenue_pct"] >= 10.0
    # Alcohol retail/distribution: >= 10% revenue threshold
    retail = df["alcohol_retail_revenue_pct"] >= 10.0
    return production | related | retail


def exclude_fossil_fuel_revenue(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies exceeding fossil fuel revenue thresholds (PAB only).

    This is the broadest fossil-fuel screen in the PACT methodology, covering
    four fuel types with tiered thresholds that reflect each fuel's relative
    carbon intensity and role in the energy transition:

    Coal (>= 1%):
        Near-zero tolerance for the most carbon-intensive fossil fuel.
        Even 1% of revenue from thermal coal mining/sales triggers exclusion.

    Oil (>= 10%):
        Higher threshold recognises oil's broader role in the global economy
        while still excluding companies with material upstream exposure.

    Natural Gas (>= 50%):
        The most lenient fossil fuel threshold. Gas is the lowest-emission
        fossil fuel and is considered a "bridge fuel" in many transition
        scenarios, so only companies deriving a majority of revenue from
        gas are excluded.

    Fossil-fuel Power Generation (>= 50%):
        Excludes utilities where fossil-fuel-fired generation dominates the
        revenue mix. The 50% threshold allows diversified utilities with
        significant renewable capacity to remain eligible.

    Ref: §Exclusions Based on Revenue Thresholds in Fossil Fuel Operations (p. 11)
    """
    # Thermal coal: near-zero tolerance (>= 1%)
    coal = df["coal_revenue_pct"] >= 1.0
    # Oil: material exposure threshold (>= 10%)
    oil = df["oil_revenue_pct"] >= 10.0
    # Natural gas: majority-revenue threshold (>= 50%)
    gas = df["natural_gas_revenue_pct"] >= 50.0
    # Fossil-fuel power generation: majority-revenue threshold (>= 50%)
    power = df["power_generation_revenue_pct"] >= 50.0
    return coal | oil | gas | power


# ---------------------------------------------------------------------------
# Exclusion pipeline
# ---------------------------------------------------------------------------


def get_exclusion_checks(variant: Variant) -> list[tuple[str, callable]]:
    """
    Return the ordered exclusion pipeline for the given variant.

    The ORDER IS SIGNIFICANT: each stock is recorded against the first filter
    it fails. If a stock would fail multiple screens, only the earliest one
    is logged. This design:
      - Keeps the exclusion log unambiguous (one reason per ticker)
      - Avoids evaluating unnecessary filters on already-excluded stocks
      - Allows callers to rely on reason strings being stable identifiers

    Core checks (CTB + PAB) run first, followed by PAB-only checks. Within
    each group the order follows the methodology's presentation order.

    The tobacco check requires the variant argument (different retail thresholds),
    so it is wrapped in a lambda to match the common (df) -> Series signature.

    Ref: §Eligibility Criteria (pp. 8-9), §Index Exclusions (pp. 9-11)
    """
    # --- Core checks shared by both CTB and PAB ---
    checks: list[tuple[str, callable]] = [
        ("No carbon coverage", exclude_no_carbon_coverage),
        ("Controversial weapons", exclude_controversial_weapons),
        # Lambda wraps exclude_tobacco to inject the variant argument
        ("Tobacco", lambda df: exclude_tobacco(df, variant)),
        ("UNGC non-compliant / no coverage", exclude_ungc_non_compliant),
        ("MSA flagged", exclude_msa_flagged),
    ]

    # --- PAB-only checks (stricter Paris-aligned screens) ---
    if variant == Variant.PAB:
        checks.extend([
            ("Small arms", exclude_small_arms),
            ("Military contracting", exclude_military_contracting),
            ("Thermal coal", exclude_thermal_coal),
            ("Oil sands", exclude_oil_sands),
            ("Shale oil & gas", exclude_shale_oil_gas),
            ("Gambling", exclude_gambling),
            ("Alcohol", exclude_alcohol),
            ("Fossil fuel revenue", exclude_fossil_fuel_revenue),
        ])

    return checks


def apply_exclusions(
    df: pd.DataFrame,
    variant: Variant,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Apply all exclusion filters for the given variant in sequence, returning
    eligible stocks and a log of which stocks were excluded and why.

    Parameters
    ----------
    df:
        Full universe DataFrame from IndexUniverse.to_dataframe().
    variant:
        CTB or PAB — controls which exclusion screens are applied and
        what thresholds are used (e.g. tobacco retail 10% vs 5%).

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
        - The pipeline short-circuits (breaks) if remaining becomes empty,
          avoiding unnecessary evaluation of later filters.

    Ref: §Eligibility Criteria (pp. 8-11)
    """
    excluded: dict[str, str] = {}
    # Work on a copy so we can safely drop rows without mutating the caller's data
    remaining = df.copy()

    for reason, check_fn in get_exclusion_checks(variant):
        if remaining.empty:
            # No stocks left to screen -- short-circuit the remaining checks
            break

        # Run the filter function: returns a boolean Series (True = exclude)
        mask = check_fn(remaining)

        # Record every newly-excluded ticker against this reason
        for ticker in remaining.loc[mask, "ticker"].tolist():
            excluded[ticker] = reason

        # Remove excluded stocks so they don't appear in subsequent filter calls
        remaining = remaining.loc[~mask].copy()

    # `remaining` now contains only stocks that passed every filter
    return remaining, excluded
