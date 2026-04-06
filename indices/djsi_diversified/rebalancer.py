"""
Rebalancer for the Dow Jones Sustainability Diversified Indices.

Orchestrates the full rebalancing pipeline:
  1. Load the universe from a CSV file into typed Pydantic models.
  2. Apply hard exclusion screens (business activities, UNGC, MSA, ESG coverage).
  3. Run best-in-class selection within each GICS Sector x Region group,
     targeting 50% of each group's float-adjusted market cap.
  4. Compute float-adjusted market-cap weights for selected companies.
  5. Apply the 10% single-stock cap with iterative redistribution to prevent
     any one company from dominating the index.
  6. Return a RebalanceResult with final weights, selected tickers, and
     a full exclusion log.

The index rebalances semi-annually in March and September. Between
rebalancings, the quarterly UNGC eligibility review may remove additional
companies that become non-compliant.

Ref: "Dow Jones Sustainability Diversified Indices Methodology" (S&P Dow Jones Indices)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .eligibility import apply_hard_exclusions, select_best_in_class
from .models import (
    BusinessActivityExposures,
    IndexUniverse,
    RebalanceResult,
    Stock,
    UNGCStatus,
)


# ---------------------------------------------------------------------------
# Weight capping constants
# Ref: §Index Calculations — Single-Stock Capping
#
# The methodology imposes a 10% cap on any single constituent's weight to
# ensure diversification. When a stock is capped, its excess weight is
# redistributed proportionally to all uncapped stocks. This redistribution
# may push previously-uncapped stocks above 10%, so the process iterates
# until convergence (or a safety limit is reached).
# ---------------------------------------------------------------------------

# Maximum single-stock weight after capping
MAX_STOCK_WEIGHT: float = 0.10  # 10%


# ---------------------------------------------------------------------------
# Rebalancing pipeline
# ---------------------------------------------------------------------------


def rebalance(
    universe: IndexUniverse,
    *,
    max_stock_weight: float = MAX_STOCK_WEIGHT,
) -> RebalanceResult:
    """
    Execute the full DJSI Diversified rebalancing pipeline and return results.

    Pipeline steps:

      1. Flatten the universe to a DataFrame for vectorised eligibility checks.
      2. Apply all hard exclusion filters (ESG coverage, business activities,
         UNGC, MSA). Stocks that fail any filter are removed; the first failure
         reason is recorded in the exclusion log.
      3. Run best-in-class selection on the survivors. The selection algorithm
         picks the highest-ESG-scoring companies within each GICS Sector x
         Region group, targeting 50% of each group's float-adjusted market cap.
      4. Compute initial weights as each selected company's float-adjusted
         market cap divided by the total FMC of all selected companies.
      5. Apply the single-stock cap: iteratively cap any stock above 10% and
         redistribute its excess weight proportionally to uncapped stocks.

    Parameters
    ----------
    universe:
        Full candidate universe including stocks that may be ineligible.
    max_stock_weight:
        Maximum weight for any single stock (default 10%). Exposed as a
        parameter for testing; production always uses 10%.

    Returns
    -------
    RebalanceResult with:
      - rebalanced_weights: final ticker -> weight mapping (sums to ~1.0)
      - selected_tickers: tickers chosen by best-in-class selection
      - excluded_tickers: ticker -> first exclusion reason for removed stocks
      - capped_tickers: tickers whose weights were reduced by the cap

    Ref: §Index Calculations, §Eligibility Criteria (pp. 5-6)
    """
    # Flatten the Pydantic universe into a DataFrame that the eligibility and
    # selection modules can work with using standard pandas operations.
    df = universe.to_dataframe()

    # --- Step 1: Hard exclusions ---
    # eligible_df contains only stocks that passed every filter.
    # excluded maps every removed ticker to the reason it was first rejected.
    eligible_df, excluded = apply_hard_exclusions(df)

    # --- Step 2: Best-in-class selection ---
    # Select the highest-ESG-scoring companies within each GICS Sector x Region
    # group, targeting 50% of each group's float-adjusted market cap.
    selected_tickers = select_best_in_class(eligible_df)

    # --- Step 3: Float-cap weighting ---
    # Weight each selected company by its float-adjusted market cap relative to
    # the total FMC of all selected companies. This produces a pure FMC-weighted
    # portfolio before any capping is applied.
    selected_df = eligible_df[eligible_df["ticker"].isin(selected_tickers)]
    total_fmc = selected_df["float_adjusted_market_cap"].sum()

    if total_fmc == 0:
        # Edge case: no selected stocks have any FMC (can happen with degenerate
        # test data). Return an empty result rather than dividing by zero.
        return RebalanceResult(
            rebalanced_weights={},
            selected_tickers=selected_tickers,
            excluded_tickers=excluded,
            capped_tickers=[],
        )

    raw_weights = {
        row["ticker"]: row["float_adjusted_market_cap"] / total_fmc
        for _, row in selected_df.iterrows()
    }

    # --- Step 4: Single-stock cap with iterative redistribution ---
    # Cap any stock above 10% and redistribute its excess weight proportionally
    # to uncapped stocks. The process iterates because redistribution may push
    # previously-uncapped stocks above the cap.
    capped_weights, capped_tickers = _apply_weight_cap(raw_weights, max_stock_weight)

    return RebalanceResult(
        rebalanced_weights=capped_weights,
        selected_tickers=selected_tickers,
        excluded_tickers=excluded,
        capped_tickers=capped_tickers,
    )


# ---------------------------------------------------------------------------
# Weight capping algorithm
# Ref: §Index Calculations — Single-Stock Capping
#
# The iterative redistribution algorithm works as follows:
#   1. Find all stocks whose current weight exceeds the cap.
#   2. Set those stocks to exactly the cap weight and "lock" them.
#   3. Compute the remaining weight budget (1.0 minus all locked weight).
#   4. Redistribute the remaining budget to unlocked stocks, proportionally
#      to their current (pre-redistribution) weights.
#   5. Repeat until no unlocked stock exceeds the cap or max_iterations.
#
# The algorithm converges quickly because each iteration locks at least one
# stock and shrinks the unlocked set. In practice, 2-3 iterations suffice
# for typical index sizes.
# ---------------------------------------------------------------------------


def _apply_weight_cap(
    weights: dict[str, float],
    cap: float,
    max_iterations: int = 50,
) -> tuple[dict[str, float], list[str]]:
    """
    Iteratively cap stock weights and redistribute excess to uncapped stocks.

    Parameters
    ----------
    weights:
        Initial (uncapped) ticker -> weight mapping. Should sum to ~1.0.
    cap:
        Maximum allowed weight for any single stock (e.g. 0.10 for 10%).
    max_iterations:
        Safety valve to prevent infinite loops in pathological cases.
        50 iterations is far more than needed for any realistic index size.

    Returns
    -------
    capped_weights:
        Final ticker -> weight mapping after capping and redistribution.
    capped_tickers:
        List of tickers that were capped (i.e. whose original weight exceeded
        the cap and was reduced to exactly `cap`).

    Design notes:
        - Stocks are "locked" once capped — they will not be adjusted again.
        - Redistribution is proportional: if stock A had 2x the weight of
          stock B before redistribution, it gets 2x the redistributed weight.
        - The algorithm terminates when no unlocked stock exceeds the cap,
          when the remaining budget is exhausted, or at max_iterations.
    """
    locked: set[str] = set()
    w = dict(weights)

    for _ in range(max_iterations):
        # Find all unlocked stocks that exceed the cap
        over = {t for t, wt in w.items() if wt > cap and t not in locked}
        if not over:
            # Converged: no unlocked stock exceeds the cap
            break

        # Cap the over-weight stocks and lock them
        for t in over:
            w[t] = cap
            locked.add(t)

        # Redistribute: unlocked stocks fill the remaining budget proportionally
        locked_total = len(locked) * cap
        remaining_budget = 1.0 - locked_total
        unlocked = {t: wt for t, wt in w.items() if t not in locked}
        unlocked_sum = sum(unlocked.values())
        if unlocked_sum <= 0 or remaining_budget <= 0:
            # Edge case: all stocks are locked or no budget left
            break

        # Scale each unlocked stock proportionally to consume the full
        # remaining budget while preserving relative weight ordering
        for t in unlocked:
            w[t] = w[t] / unlocked_sum * remaining_budget

    return w, list(locked)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def result_to_dataframe(result: RebalanceResult, universe: IndexUniverse) -> pd.DataFrame:
    """
    Convert a RebalanceResult into a tidy DataFrame for display or CSV export.

    Each row represents one constituent of the rebalanced index and includes
    all stock metadata (ticker, company, country, region, sector, ESG score,
    etc.) plus the final rebalanced weight.

    Sorted by rebalanced_weight descending (largest position first).
    """
    df = universe.to_dataframe()
    selected_df = df[df["ticker"].isin(result.selected_tickers)].copy()
    selected_df["rebalanced_weight"] = selected_df["ticker"].map(result.rebalanced_weights)
    return selected_df.sort_values("rebalanced_weight", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_universe_from_csv(path: Path) -> IndexUniverse:
    """
    Parse a universe CSV into an IndexUniverse of typed Stock models.

    The CSV must contain one row per security. Float-adjusted market-cap
    weights are computed automatically by IndexUniverse from the market_cap_usd
    and float_ratio columns — pre-computed weights are not needed.

    Type coercion:
        - esg_score may be blank (NaN in pandas) to represent missing data;
          it becomes None in the Stock model.
        - ungc_status is parsed as a UNGCStatus enum value; defaults to
          "Compliant" if the column is missing.
        - has_esg_coverage and msa_flagged are stored as 0/1 integers in the
          CSV and cast to bool here.
        - All business activity percentage columns are cast to float, with
          missing values defaulting to 0.0 (no involvement).

    Ref: §Eligibility Criteria, §Index Construction
    """
    df = pd.read_csv(path)
    stocks = []
    for _, row in df.iterrows():
        # Build the BusinessActivityExposures model.
        # All percentage columns default to 0.0 if missing from the CSV,
        # which means "no involvement" — the safest assumption for optional data.
        ba = BusinessActivityExposures(
            controversial_weapons_revenue_pct=float(row.get("controversial_weapons_revenue_pct", 0)),
            controversial_weapons_ownership_pct=float(row.get("controversial_weapons_ownership_pct", 0)),
            tobacco_production_revenue_pct=float(row.get("tobacco_production_revenue_pct", 0)),
            adult_entertainment_production_revenue_pct=float(row.get("adult_entertainment_production_revenue_pct", 0)),
            adult_entertainment_retail_revenue_pct=float(row.get("adult_entertainment_retail_revenue_pct", 0)),
            alcohol_production_revenue_pct=float(row.get("alcohol_production_revenue_pct", 0)),
            gambling_operations_revenue_pct=float(row.get("gambling_operations_revenue_pct", 0)),
            gambling_equipment_revenue_pct=float(row.get("gambling_equipment_revenue_pct", 0)),
            military_integral_weapons_revenue_pct=float(row.get("military_integral_weapons_revenue_pct", 0)),
            military_weapon_related_revenue_pct=float(row.get("military_weapon_related_revenue_pct", 0)),
            small_arms_civilian_production_revenue_pct=float(row.get("small_arms_civilian_production_revenue_pct", 0)),
            small_arms_key_components_revenue_pct=float(row.get("small_arms_key_components_revenue_pct", 0)),
            small_arms_noncivilian_production_revenue_pct=float(row.get("small_arms_noncivilian_production_revenue_pct", 0)),
            small_arms_retail_revenue_pct=float(row.get("small_arms_retail_revenue_pct", 0)),
        )

        # Optional fields: blank CSV cells become None (not 0.0 or NaN),
        # so that downstream code can distinguish "no data" from "zero".
        stocks.append(Stock(
            ticker=str(row["ticker"]),
            company_name=str(row["company_name"]),
            country=str(row["country"]),
            region=str(row["region"]),
            gics_sector=str(row["gics_sector"]),
            gics_industry_group=str(row["gics_industry_group"]),
            market_cap_usd=float(row["market_cap_usd"]),
            float_ratio=float(row["float_ratio"]),
            esg_score=float(row["esg_score"]) if pd.notna(row.get("esg_score")) else None,
            has_esg_coverage=bool(row.get("has_esg_coverage", True)),
            ungc_status=UNGCStatus(row.get("ungc_status", "Compliant")),
            msa_flagged=bool(row.get("msa_flagged", False)),
            business_activities=ba,
        ))

    # IndexUniverse automatically computes float-adjusted market-cap weights
    # for each stock if no explicit underlying_weights dict is provided.
    return IndexUniverse(stocks=stocks)
