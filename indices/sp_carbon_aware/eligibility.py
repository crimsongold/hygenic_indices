"""
Eligibility filtering for the S&P Carbon Aware Index Series.

Each public function takes the universe DataFrame (from IndexUniverse.to_dataframe())
and returns a boolean Series where True means the stock should be EXCLUDED.

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
    Applied to both Developed and Emerging variants.
    """
    return ~df["has_esg_coverage"] | df["esg_score"].isna()


def exclude_bottom_quartile_esg(
    df: pd.DataFrame,
    universe_type: str = "developed",
) -> pd.Series:
    """
    Exclude companies whose ESG score falls in the worst 25% of their
    global GICS Industry Group.

    For the Developed index: worst 25% within global GICS Industry Group
    (combined S&P Global LargeMidCap + S&P Global 1200).

    For the Emerging index: worst 25% within the underlying index's own
    GICS Industry Group. Companies without ESG scores are included in the
    worst 25% bucket.

    Ref: §Exclusions Based on S&P Global ESG Score (p. 5)
    """
    exclude = pd.Series(False, index=df.index)
    for group, group_df in df.groupby("gics_industry_group"):
        if universe_type == "emerging":
            # Companies with no ESG coverage are always treated as worst performers
            # and are unconditionally included in the worst 25%.
            # Ref: §Exclusions Based on S&P Global ESG Score (p. 5)
            no_score_mask = ~group_df["has_esg_coverage"] | group_df["esg_score"].isna()
            exclude.loc[group_df.index[no_score_mask]] = True

            scored = group_df.loc[~no_score_mask, "esg_score"]
            if scored.empty:
                continue
            threshold = scored.quantile(0.25)
            bottom_25 = scored.index[scored < threshold]
        else:
            scored = group_df.loc[group_df["has_esg_coverage"], "esg_score"].dropna()
            if scored.empty:
                continue
            threshold = scored.quantile(0.25)
            # Strictly below the 25th-percentile threshold
            bottom_25 = scored.index[scored < threshold]

        exclude.loc[bottom_25] = True

    return exclude


# ---------------------------------------------------------------------------
# Business Activity Exclusions
# Ref: §Exclusions Based on Business Activities (pp. 5-6)
# ---------------------------------------------------------------------------


def _exceeds_threshold(series: pd.Series, threshold: float) -> pd.Series:
    """Return True where a revenue/involvement % strictly exceeds threshold."""
    return series > threshold


def _meets_or_exceeds(series: pd.Series, threshold: float) -> pd.Series:
    """Return True where a revenue/involvement % is >= threshold."""
    return series >= threshold


def exclude_no_sustainalytics_coverage(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies without Sustainalytics coverage.
    Ref: §Exclusions Based on Business Activities (p. 5) — 'companies without coverage'
    """
    return ~df["has_sustainalytics_coverage"]


def exclude_controversial_weapons(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in controversial weapons.

    Tailor-Made and Essential involvement:
        Direct involvement (revenue proxy) > 0%  OR  ownership >= 10%
    Non Tailor-Made or Non-Essential involvement:
        Direct involvement (revenue proxy) > 0%  OR  ownership >= 10%

    Ref: §Exclusions Based on Business Activities, Controversial Weapons row (p. 5-6)
    """
    tailor_made_direct = _exceeds_threshold(
        df["controversial_weapons_tailor_made_essential_pct"], 0.0
    )
    tailor_made_ownership = _meets_or_exceeds(
        df["controversial_weapons_tailor_made_essential_ownership_pct"], 10.0
    )
    non_tailor_direct = _exceeds_threshold(
        df["controversial_weapons_non_tailor_made_pct"], 0.0
    )
    non_tailor_ownership = _meets_or_exceeds(
        df["controversial_weapons_non_tailor_made_ownership_pct"], 10.0
    )
    return (
        tailor_made_direct
        | tailor_made_ownership
        | non_tailor_direct
        | non_tailor_ownership
    )


def exclude_tobacco(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in tobacco.

    Production: revenue > 0%
    Related Products/Services: revenue >= 5%
    Retail: revenue >= 5%

    Ref: §Exclusions Based on Business Activities, Tobacco Products row (p. 6-7)
    """
    production = _exceeds_threshold(df["tobacco_production_revenue_pct"], 0.0)
    related = _meets_or_exceeds(df["tobacco_related_revenue_pct"], 5.0)
    retail = _meets_or_exceeds(df["tobacco_retail_revenue_pct"], 5.0)
    return production | related | retail


def exclude_thermal_coal(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in thermal coal.

    Extraction: revenue > 0%
    Power Generation: revenue > 0%

    Ref: §Exclusions Based on Business Activities, Thermal Coal row (p. 6-7)
    """
    extraction = _exceeds_threshold(df["thermal_coal_extraction_revenue_pct"], 0.0)
    power_gen = _exceeds_threshold(
        df["thermal_coal_power_generation_revenue_pct"], 0.0
    )
    return extraction | power_gen


def exclude_oil_sands(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in oil sands extraction (revenue > 0%).
    Ref: §Exclusions Based on Business Activities, Oil Sands row (p. 6-7)
    """
    return _exceeds_threshold(df["oil_sands_extraction_revenue_pct"], 0.0)


def exclude_shale_energy(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in shale energy extraction (revenue > 0%).
    Ref: §Exclusions Based on Business Activities, Shale Energy row (p. 6-7)
    """
    return _exceeds_threshold(df["shale_energy_extraction_revenue_pct"], 0.0)


def exclude_arctic_oil_gas(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in Arctic oil & gas exploration (revenue > 0%).
    Ref: §Exclusions Based on Business Activities, Arctic Oil & Gas row (p. 6-7)
    """
    return _exceeds_threshold(df["arctic_oil_gas_extraction_revenue_pct"], 0.0)


def exclude_oil_gas(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in oil & gas.

    Production: revenue > 0%
    Generation (electricity from oil/gas): revenue > 0%
    Supporting Products/Services (tailor-made): revenue >= 10%

    Ref: §Exclusions Based on Business Activities, Oil & Gas row (p. 6-7)
    """
    production = _exceeds_threshold(df["oil_gas_production_revenue_pct"], 0.0)
    generation = _exceeds_threshold(df["oil_gas_generation_revenue_pct"], 0.0)
    supporting = _meets_or_exceeds(df["oil_gas_supporting_revenue_pct"], 10.0)
    return production | generation | supporting


def exclude_gambling(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in gambling.

    Operations: revenue >= 5%
    Specialized Equipment: revenue >= 10%
    Supporting Products/Services: revenue >= 10%

    Ref: §Exclusions Based on Business Activities, Gambling row (p. 6-7)
    """
    operations = _meets_or_exceeds(df["gambling_operations_revenue_pct"], 5.0)
    equipment = _meets_or_exceeds(df["gambling_equipment_revenue_pct"], 10.0)
    supporting = _meets_or_exceeds(df["gambling_supporting_revenue_pct"], 10.0)
    return operations | equipment | supporting


def exclude_adult_entertainment(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in adult entertainment.

    Production: revenue > 0%
    Distribution: revenue >= 5%

    Ref: §Exclusions Based on Business Activities, Adult Entertainment row (p. 6-7)
    """
    production = _exceeds_threshold(
        df["adult_entertainment_production_revenue_pct"], 0.0
    )
    distribution = _meets_or_exceeds(
        df["adult_entertainment_distribution_revenue_pct"], 5.0
    )
    return production | distribution


def exclude_alcoholic_beverages(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies involved in alcoholic beverages.

    Production: revenue >= 5%
    Retail: revenue >= 10%
    Related Products/Services: revenue >= 10%

    Ref: §Exclusions Based on Business Activities, Alcoholic Beverages row (p. 6-7)
    """
    production = _meets_or_exceeds(
        df["alcoholic_beverages_production_revenue_pct"], 5.0
    )
    retail = _meets_or_exceeds(df["alcoholic_beverages_retail_revenue_pct"], 10.0)
    related = _meets_or_exceeds(
        df["alcoholic_beverages_related_revenue_pct"], 10.0
    )
    return production | retail | related


# ---------------------------------------------------------------------------
# UNGC / Global Standards Screening Exclusions
# Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 7)
# ---------------------------------------------------------------------------


def exclude_ungc_non_compliant(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies classified as Non-Compliant by Sustainalytics GSS,
    and those without Sustainalytics coverage.

    Ref: §Exclusions Based on Sustainalytics' Global Standards Screening (p. 7)
    """
    no_coverage = df["ungc_status"] == "No Coverage"
    non_compliant = df["ungc_status"] == "Non-Compliant"
    return no_coverage | non_compliant


# ---------------------------------------------------------------------------
# MSA Controversy Exclusions
# Ref: §Controversies: Media and Stakeholder Analysis Overlay (p. 7)
# ---------------------------------------------------------------------------


def exclude_msa_flagged(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies flagged by the Media and Stakeholder Analysis (MSA) review.
    In practice this is an Index Committee decision; here we use the msa_flagged field
    as a direct input representing that decision.

    Ref: §Controversies: Media and Stakeholder Analysis Overlay (p. 7)
    """
    return df["msa_flagged"].astype(bool)


# ---------------------------------------------------------------------------
# Combined eligibility filter
# ---------------------------------------------------------------------------

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
    Apply all eligibility filters in sequence and return:
      - eligible_df: rows that pass all filters
      - excluded: mapping of ticker -> first exclusion reason

    The ESG bottom-quartile check uses the universe_type parameter.
    Ref: §Eligibility Criteria (pp. 5-7)
    """
    excluded: dict[str, str] = {}
    remaining = df.copy()

    for reason, check_fn in EXCLUSION_CHECKS:
        if remaining.empty:
            break

        if check_fn is exclude_bottom_quartile_esg:
            mask = check_fn(remaining, universe_type=universe_type)
        else:
            mask = check_fn(remaining)

        newly_excluded = remaining.loc[mask, "ticker"].tolist()
        for ticker in newly_excluded:
            excluded[ticker] = reason

        remaining = remaining.loc[~mask].copy()

    return remaining, excluded
