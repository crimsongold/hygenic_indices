"""
Eligibility filtering for the Dow Jones Sustainability Diversified Indices.

The DJSI Diversified Select methodology applies a two-stage selection process:

  Stage 1 — Hard Exclusions:
    Remove companies involved in certain business activities (controversial
    weapons, tobacco, adult entertainment, alcohol, gambling, military
    contracting, small arms) as well as UNGC non-compliant companies and
    those flagged by the MSA controversy overlay. These are non-negotiable
    binary screens — a company either passes or it doesn't.

  Stage 2 — Best-in-Class Selection:
    Among the survivors, select the highest-ESG-scoring companies within
    each GICS Sector x Region group, targeting approximately 50% of each
    group's total float-adjusted market cap (FMC). The selection uses a
    40% initial threshold with a 40-60% buffer zone to balance precision
    against target coverage.

Each public exclusion function takes the universe DataFrame (from
IndexUniverse.to_dataframe()) and returns a boolean Series where True means
the stock should be EXCLUDED.

Filters are applied in a fixed sequence (see EXCLUSION_CHECKS). A stock is
recorded against the FIRST filter it fails — subsequent filters are never
evaluated for that stock. This means the order of EXCLUSION_CHECKS matters
both for performance and for the reason label stored in the exclusion log.

Ref: "Dow Jones Sustainability Diversified Indices Methodology" (S&P Dow Jones Indices)
     §Exclusions Based on Business Activities (pp. 5-6)
     §Constituent Selection (p. 6)
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# ESG Coverage Exclusion
# Ref: §Exclusions — "companies without coverage"
#
# Companies must have an S&P Global ESG Score (derived from the Corporate
# Sustainability Assessment) to participate in best-in-class ranking. Without
# a score, the company cannot be ranked against peers and is therefore
# excluded before the selection stage even runs.
# ---------------------------------------------------------------------------


def exclude_no_esg_coverage(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies without an S&P Global ESG score.

    A company is considered uncovered if either:
      - has_esg_coverage is False (S&P Global has not assessed it), OR
      - esg_score is NaN (assessed but score is unavailable/pending)

    This check runs first in the pipeline so that uncovered companies don't
    distort the ESG ranking in the subsequent best-in-class selection step.

    Ref: §Exclusions — "companies without coverage"
    """
    # Missing coverage flag OR null score value -> exclude
    return ~df["has_esg_coverage"] | df["esg_score"].isna()


# ---------------------------------------------------------------------------
# Business Activity Exclusions
# Ref: §Exclusions Based on Business Activities (pp. 5-6)
#
# The methodology defines hard exclusion screens for seven categories of
# business activity. Each category has one or more sub-categories with
# specific revenue or ownership thresholds:
#
#   "> 0%"  means ANY revenue from that activity triggers exclusion
#   ">= 5%" means the company must derive at least 5% of revenue
#   ">=10%" means at least 10% ownership of a subsidiary with involvement
#
# These thresholds reflect the methodology's graduated view of materiality:
# core activities (production, operations) use zero-tolerance or low
# thresholds, while peripheral activities (retail, components) use higher
# thresholds to avoid excluding companies with incidental exposure.
# ---------------------------------------------------------------------------


def exclude_controversial_weapons(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any involvement in controversial weapons
    (cluster munitions, anti-personnel mines, biological/chemical/nuclear
    weapons, depleted uranium weapons, and white phosphorus weapons).

    The methodology uses two independent triggers:

    Direct Revenue (> 0%):
        Any revenue at all from controversial weapons triggers exclusion.
        Zero-tolerance because these weapons are banned under international
        treaties and conventions.

    Significant Ownership (>= 10%):
        Owning >= 10% of a subsidiary involved in controversial weapons also
        triggers exclusion. This captures indirect participation through
        corporate ownership structures.

    Either trigger alone is sufficient for exclusion.

    Ref: §Business Activity Exclusions — Controversial Weapons (pp. 5-6)
    """
    # Any direct revenue from controversial weapons -> exclude
    revenue = df["controversial_weapons_revenue_pct"] > 0.0
    # Significant ownership stake (>= 10%) in an involved subsidiary -> exclude
    ownership = df["controversial_weapons_ownership_pct"] >= 10.0
    return revenue | ownership


def exclude_tobacco(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies that produce tobacco products.

    Only production (manufacturing) is screened — retail and distribution are
    not excluded under the DJSI Diversified methodology. This is a narrower
    screen than the S&P Carbon Aware methodology, which also screens tobacco
    retail and related services at >= 5%.

    Production (> 0%):
        Any revenue from manufacturing tobacco products triggers exclusion.
        Zero-tolerance because tobacco production is the core harmful activity.

    Ref: §Business Activity Exclusions — Tobacco (pp. 5-6), production > 0%
    """
    # Any tobacco manufacturing revenue -> exclude immediately
    return df["tobacco_production_revenue_pct"] > 0.0


def exclude_adult_entertainment(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material adult entertainment involvement.

    Unlike the S&P Carbon Aware methodology (which uses > 0% for production),
    the DJSI Diversified methodology applies a >= 5% materiality threshold to
    both production and retail. This means a media conglomerate with a very
    small adult content subsidiary (< 5%) would not be excluded.

    Production (>= 5% of revenue):
        Producing adult content or operating adult entertainment venues.

    Retail (>= 5% of revenue):
        Distributing or retailing adult entertainment materials.

    Ref: §Business Activity Exclusions — Adult Entertainment (pp. 5-6)
    """
    # Adult content production: material involvement threshold (>= 5%)
    production = df["adult_entertainment_production_revenue_pct"] >= 5.0
    # Adult content retail/distribution: same materiality threshold (>= 5%)
    retail = df["adult_entertainment_retail_revenue_pct"] >= 5.0
    return production | retail


def exclude_alcohol(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any alcohol production revenue.

    The DJSI Diversified methodology only screens alcohol production — retail
    and related services are not excluded. This reflects a focus on the core
    manufacturing activity rather than downstream distribution.

    Production (> 0%):
        Any revenue from manufacturing alcoholic beverages triggers exclusion.
        Zero-tolerance because producing alcohol is the primary activity.

    Note: this is stricter than the S&P Carbon Aware methodology for production
    (which uses >= 5%) but narrower in scope (no retail or related screens).

    Ref: §Business Activity Exclusions — Alcohol (pp. 5-6), production > 0%
    """
    # Any alcohol manufacturing revenue -> exclude immediately
    return df["alcohol_production_revenue_pct"] > 0.0


def exclude_gambling(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with any involvement in gambling.

    Both operations and equipment manufacturing use zero-tolerance thresholds,
    meaning any revenue at all from either sub-category triggers exclusion.
    This is stricter than the S&P Carbon Aware methodology (which uses >= 5%
    for operations and >= 10% for equipment).

    Operations (> 0%):
        Owning or operating casinos, betting establishments, or online
        gambling platforms. Any revenue -> excluded.

    Equipment (> 0%):
        Manufacturing equipment used exclusively for gambling (slot machines,
        gaming tables, etc.). Any revenue -> excluded.

    Ref: §Business Activity Exclusions — Gambling (pp. 5-6)
    """
    # Operating gambling establishments: zero-tolerance
    operations = df["gambling_operations_revenue_pct"] > 0.0
    # Manufacturing gambling equipment: zero-tolerance
    equipment = df["gambling_equipment_revenue_pct"] > 0.0
    return operations | equipment


def exclude_military_contracting(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material involvement in military contracting.

    This screen targets companies that derive significant revenue from
    weapons systems, but uses a >= 5% materiality threshold to avoid
    excluding large diversified defence/aerospace companies where military
    weapons are a small fraction of the business.

    Integral Weapons (>= 5% of revenue):
        Companies that manufacture complete weapons systems or integral
        components without which the weapon cannot function.

    Weapon Related (>= 5% of revenue):
        Companies that provide weapon-related products or services (e.g.
        guidance systems, ammunition, weapon-specific electronics).

    Either sub-category alone triggers exclusion at the 5% threshold.

    Ref: §Business Activity Exclusions — Military Contracting (pp. 5-6)
    """
    # Integral weapons manufacturing: material involvement (>= 5%)
    integral = df["military_integral_weapons_revenue_pct"] >= 5.0
    # Weapon-related products/services: material involvement (>= 5%)
    weapon_related = df["military_weapon_related_revenue_pct"] >= 5.0
    return integral | weapon_related


def exclude_small_arms(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies with material involvement in small arms (firearms).

    The methodology distinguishes four sub-categories to capture the full
    small arms value chain — from manufacturing through retail — but applies
    the same >= 5% materiality threshold to each. A company is excluded if
    ANY single sub-category reaches >= 5%.

    Civilian Production (>= 5%):
        Manufacturing firearms intended for civilian/commercial sale
        (hunting rifles, sporting shotguns, personal defence weapons).

    Key Components (>= 5%):
        Manufacturing essential components (barrels, receivers, triggers)
        without which the firearm cannot function.

    Non-Civilian Production (>= 5%):
        Manufacturing firearms for military or law enforcement use.

    Retail (>= 5%):
        Distributing or retailing small arms to end consumers.

    Ref: §Business Activity Exclusions — Small Arms (pp. 5-6)
    """
    # Civilian firearms manufacturing: material involvement (>= 5%)
    civilian = df["small_arms_civilian_production_revenue_pct"] >= 5.0
    # Key firearm components: material involvement (>= 5%)
    components = df["small_arms_key_components_revenue_pct"] >= 5.0
    # Military/law-enforcement firearms: material involvement (>= 5%)
    noncivilian = df["small_arms_noncivilian_production_revenue_pct"] >= 5.0
    # Firearms retail/distribution: material involvement (>= 5%)
    retail = df["small_arms_retail_revenue_pct"] >= 5.0
    return civilian | components | noncivilian | retail


# ---------------------------------------------------------------------------
# UNGC / Global Standards Screening Exclusions
# Ref: §UN Global Compact Compliance
#
# Sustainalytics' Global Standards Screening (GSS) assesses each company's
# adherence to the ten principles of the UN Global Compact (UNGC), which
# cover human rights, labour standards, environmental responsibility, and
# anti-corruption.
#
# Three possible classifications:
#   Non-Compliant — confirmed violation of UNGC principles -> EXCLUDED
#   Watchlist     — potential violation under investigation -> NOT excluded
#   Compliant     — no known violations -> eligible
#   No Coverage   — Sustainalytics has not assessed the company -> EXCLUDED
#
# Watchlist companies are retained at rebalancing — they may be removed by
# a separate quarterly UNGC eligibility review between rebalancings.
# ---------------------------------------------------------------------------


def exclude_ungc_non_compliant(df: pd.DataFrame) -> pd.Series:
    """
    Exclude companies classified as Non-Compliant with UNGC principles,
    and those for which Sustainalytics provides no GSS coverage.

    Non-Compliant companies have a confirmed violation of one or more UNGC
    principles and are excluded as a hard rule. Companies with no coverage
    are treated conservatively — without an assessment, compliance cannot be
    verified, so they are excluded as a precaution.

    Watchlist companies (potential violations under investigation) are
    retained at the semi-annual rebalancing per methodology rules.

    Ref: §UN Global Compact Compliance
    """
    # Both "Non-Compliant" and "No Coverage" are grounds for exclusion
    return df["ungc_status"].isin(["Non-Compliant", "No Coverage"])


# ---------------------------------------------------------------------------
# MSA Controversy Exclusions
# Ref: §Controversies — MSA Overlay
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

    The MSA overlay is a discretionary screen — the Index Committee reviews
    each case individually and decides whether the controversy warrants
    removal. Once flagged, the company is excluded for at least one full
    calendar year from the next rebalancing before it can be reconsidered.

    Ref: §Controversies — MSA Overlay
    """
    return df["msa_flagged"].astype(bool)


# ---------------------------------------------------------------------------
# Hard exclusion pipeline (run before best-in-class selection)
# ---------------------------------------------------------------------------

# Ordered list of (human-readable reason, filter function) pairs.
# The ORDER IS SIGNIFICANT: each stock is recorded against the first filter
# it fails. If a stock would fail multiple screens, only the earliest one is
# logged. Callers can rely on reason strings being stable identifiers.
#
# Design note: ESG coverage is checked first so that companies without scores
# are removed before the business activity screens run. Business activity
# screens are ordered by severity (controversial weapons first, then other
# activities). UNGC and MSA are last because they are governance/controversy
# screens that apply to companies that might otherwise pass activity screens.
EXCLUSION_CHECKS: list[tuple[str, callable]] = [
    ("No ESG coverage", exclude_no_esg_coverage),
    ("Controversial weapons", exclude_controversial_weapons),
    ("Tobacco", exclude_tobacco),
    ("Adult entertainment", exclude_adult_entertainment),
    ("Alcohol", exclude_alcohol),
    ("Gambling", exclude_gambling),
    ("Military contracting", exclude_military_contracting),
    ("Small arms", exclude_small_arms),
    ("UNGC non-compliant / no coverage", exclude_ungc_non_compliant),
    ("MSA flagged", exclude_msa_flagged),
]


def apply_hard_exclusions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Apply all hard exclusion filters in sequence, returning eligible stocks
    and a log of which stocks were excluded and why.

    Parameters
    ----------
    df:
        Full universe DataFrame from IndexUniverse.to_dataframe().

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
        - The loop short-circuits if `remaining` is empty, avoiding unnecessary
          filter evaluations on an empty DataFrame.

    Ref: §Exclusions Based on Business Activities (pp. 5-6)
    """
    excluded: dict[str, str] = {}
    # Work on a copy so we can safely drop rows without mutating the caller's data
    remaining = df.copy()

    for reason, check_fn in EXCLUSION_CHECKS:
        if remaining.empty:
            # No stocks left to screen — short-circuit the remaining checks
            break
        mask = check_fn(remaining)
        # Record every newly-excluded ticker against this reason
        for ticker in remaining.loc[mask, "ticker"].tolist():
            excluded[ticker] = reason
        # Remove excluded stocks so they don't appear in subsequent filter calls
        remaining = remaining.loc[~mask].copy()

    return remaining, excluded


# ---------------------------------------------------------------------------
# Best-in-class selection (run on hard-exclusion survivors)
# Ref: §Constituent Selection (p. 6)
#
# The selection algorithm targets 50% of each GICS Sector x Region group's
# total float-adjusted market cap (FMC). Rather than selecting exactly to 50%
# in one pass (which would be brittle at group boundaries), the algorithm
# uses a three-phase approach:
#
#   Phase 1 (0% -> 40%): Greedily add top-ESG companies until cumulative FMC
#       reaches 40% of the group total. All companies in this range are
#       unconditionally selected — they are clearly needed to reach the target.
#
#   Phase 2 (40% -> 60%): "Buffer zone" — continue adding companies, but for
#       each candidate, check whether including it brings cumulative FMC
#       closer to the 50% target. If adding a company would overshoot and
#       we're already above the initial 40% threshold, we can skip it to stay
#       closer to target. This prevents groups from significantly exceeding
#       50% due to one large-cap company at the boundary.
#
#   Phase 3 (if still < 50%): Continue adding in ESG order until the target
#       is met. This phase only activates if Phase 2 decisions left the group
#       under 50%.
#
# The 40%/60% buffer zone around the 50% target gives current constituents a
# stability advantage at rebalancing — a company already in the index that
# falls within the buffer zone is less likely to be removed, reducing turnover.
# ---------------------------------------------------------------------------

# Target fraction of each sector-region group's total FMC to select
FLOAT_CAP_TARGET: float = 0.50  # 50%

# Initial selection threshold before buffer zone kicks in
INITIAL_THRESHOLD: float = 0.40  # 40%

# Upper bound of the buffer zone for current constituent preference
BUFFER_UPPER: float = 0.60  # 60%


def select_best_in_class(eligible_df: pd.DataFrame) -> list[str]:
    """
    Select the highest-ESG-scoring companies within each GICS Sector x Region
    group, targeting 50% of each group's float-adjusted market cap.

    The algorithm processes each GICS Sector x Region group independently:

      1. Sort companies by ESG score descending (ties broken by FMC, largest
         first — this ensures deterministic selection when scores are equal).
      2. Greedily add companies until cumulative FMC reaches 40% of the group
         total. All companies in this range are unconditionally selected.
      3. At the boundary: if adding the next company would exceed the 50%
         target, check whether including or excluding it brings us closer to
         50%. Only skip the company if we're already above 40% (the initial
         threshold) AND excluding it gets us closer to the 50% target.
      4. Stop once cumulative FMC reaches or exceeds the target.

    The 50% target (not 25%) is specific to the DJSI Diversified Select
    variant. Other DJSI variants may use different targets. The three
    constants FLOAT_CAP_TARGET, INITIAL_THRESHOLD, and BUFFER_UPPER control
    the selection behaviour and can be adjusted for different index variants.

    Parameters
    ----------
    eligible_df:
        DataFrame of companies that passed all hard exclusion screens.
        Must contain columns: ticker, gics_sector, region,
        float_adjusted_market_cap, esg_score.

    Returns
    -------
    List of selected tickers across all GICS Sector x Region groups.

    Ref: §Constituent Selection (p. 6)
    """
    selected: list[str] = []

    for (_sector, _region), group_df in eligible_df.groupby(["gics_sector", "region"]):
        if group_df.empty:
            continue

        # Sort by ESG score descending (ties broken by float-cap, largest first)
        # This ensures the highest-quality companies are selected first, and
        # among equals, the largest by market cap gets priority (more index-
        # representative and reduces turnover from tie-breaking randomness).
        sorted_df = group_df.sort_values(
            ["esg_score", "float_adjusted_market_cap"],
            ascending=[False, False],
        ).reset_index(drop=True)

        group_total_fmc = group_df["float_adjusted_market_cap"].sum()
        if group_total_fmc == 0:
            # Degenerate case: all companies have zero FMC (e.g. test data).
            # Skip the group entirely — no meaningful weighting is possible.
            continue

        target_fmc = FLOAT_CAP_TARGET * group_total_fmc

        cumulative = 0.0
        for i, row in sorted_df.iterrows():
            prev_cumulative = cumulative
            cumulative += row["float_adjusted_market_cap"]
            selected.append(row["ticker"])

            # Once we pass the target, check if this stock brought us closer
            if cumulative >= target_fmc:
                # Compare distance to target with vs without this company
                diff_with = abs(cumulative - target_fmc)
                diff_without = abs(prev_cumulative - target_fmc)
                if diff_with > diff_without and prev_cumulative >= INITIAL_THRESHOLD * group_total_fmc:
                    # Removing this stock gets us closer to the 50% target,
                    # and we've already passed the 40% initial threshold, so
                    # it's safe to skip — undo the selection
                    selected.pop()
                    cumulative = prev_cumulative
                break

    return selected
