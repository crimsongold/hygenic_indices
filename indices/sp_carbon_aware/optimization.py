"""
Portfolio optimization for the S&P Carbon Aware Index Series.

The optimizer minimizes the weighted-average carbon intensity (WACI) of the
eligible constituent universe subject to a set of linear and quadratic constraints
that keep the resulting portfolio close to its underlying benchmark.

All constraints are formulated as convex expressions and solved by CLARABEL via
the cvxpy modelling layer. If the problem is infeasible at the default parameters,
a relaxation hierarchy progressively loosens soft constraints until a solution is
found. Hard constraints (minimum stock weight) are never relaxed.

Ref: "S&P Carbon Aware Index Series Methodology" (March 2026),
     §Index Construction (p. 8), §Optimization Constraints (pp. 8-9),
     §Constraint Relaxation Hierarchy (p. 9)
"""

from __future__ import annotations

import warnings
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Methodology constants
# Ref: §Optimization Constraints (pp. 8-9)
# ---------------------------------------------------------------------------

# Stock weight multiplier — how many times a stock's underlying weight it can
# be scaled up to in the optimized index (base: 10×, relaxed up to: 20×).
# Upper bound per stock = min(W_i × multiplier, W_i + excess_stock_weight).
STOCK_WEIGHT_MULTIPLIER_BASE = 10
STOCK_WEIGHT_MULTIPLIER_MAX = 20

# Excess stock weight — maximum absolute deviation from each stock's underlying
# weight (base: ±2%, relaxed up to: ±4%).
# Upper bound: W_i + excess   Lower bound: max(0, W_i - excess)
EXCESS_STOCK_WEIGHT_BASE = 0.02   # 2%
EXCESS_STOCK_WEIGHT_MAX = 0.04    # 4% (max relaxation)

# Country / Industry Group deviation band and scale factors.
# For each group g, the optimized allocation must satisfy:
#   max(W_g - 5%, W_g × 0.75)  ≤  Σw_i(in g)  ≤  min(W_g + 5%, W_g × 1.25)
COUNTRY_IG_BAND = 0.05       # ±5 percentage-point absolute band
COUNTRY_IG_SCALE_LB = 0.75   # lower bound = max(W_g - band, W_g × 0.75)
COUNTRY_IG_SCALE_UB = 1.25   # upper bound = min(W_g + band, W_g × 1.25)

# Minimum stock weight — a hard constraint achieved via post-processing rather
# than inside the optimizer. Any stock whose optimized weight falls below 1 bps
# is zeroed and its weight is redistributed to larger stocks.
MIN_STOCK_WEIGHT = 0.0001    # 1 basis point = 0.01%

# Active-share parameter T controls how much the optimized portfolio can diverge
# from the underlying index as a whole.
#   T = max(T_MIN, T_BASE − ineligible_weight)
# When many stocks are ineligible, more of the total weight must shift, so T
# is reduced to give the optimizer more room — but never below T_MIN (5%).
T_BASE = 0.25   # 25% — default maximum active share allowed
T_MIN = 0.05    # 5%  — floor applied when ineligible weight is large


class OptimizationResult(BaseModel):
    """Raw output from a single cvxpy solve attempt."""

    # numpy arrays are not natively validated by Pydantic, so we opt in to
    # arbitrary type support for this model only.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Optional[np.ndarray]  # optimized weights over eligible stocks; None if infeasible
    status: str                    # cvxpy solver status string (e.g. 'optimal', 'infeasible')
    relaxation_level: int          # 0 = no relaxation applied; higher = more constraints relaxed


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _impute_carbon_intensities(
    carbon_intensities: pd.Series, industry_groups: pd.Series
) -> pd.Series:
    """
    Fill missing carbon intensity values using the median of each stock's
    GICS Industry Group, so the optimizer has a complete CI vector.

    Why the group median?
        The methodology assigns missing-coverage companies the median carbon
        intensity of their GICS Industry Group. This is a neutral assumption
        that neither rewards nor penalises companies for lacking Trucost data —
        they're treated as a typical representative of their sector.

    If an entire industry group has no coverage (all NaN), the global median
    across all stocks in the eligible universe is used as a final fallback to
    ensure the CI vector is always fully populated.

    Ref: §Carbon Intensity footnote 2 (p. 8)
    """
    filled = carbon_intensities.copy()

    for group in industry_groups.unique():
        # Boolean mask: True for stocks in this industry group
        group_mask = industry_groups == group
        group_ci = carbon_intensities[group_mask]

        # Compute the median of all non-NaN values in the group
        group_median = group_ci.median()

        if pd.isna(group_median):
            # No coverage anywhere in this group → fall back to the global median
            group_median = carbon_intensities.median()

        # Only fill the NaN entries; leave existing values untouched
        needs_imputation = group_mask & filled.isna()
        filled[needs_imputation] = group_median

    return filled


def _compute_T(ineligible_weight: float) -> float:
    """
    Compute the active-share parameter T.

    T caps how much total weight can shift between the underlying index and the
    optimized index (used in both the Diversification Constraint and the Absolute
    Weight Deviation Constraint).

    Formula:  T = max(5%, 25% − ineligible_weight)

    Intuition: when ineligible stocks represent, say, 15% of the underlying
    index, the optimizer must move at least 15% of weight somewhere else.
    Keeping T at 25% would allow too little extra room, so we shrink T to
    (25% − 15%) = 10% — meaning the eligible-stock deviation budget is 10%
    on top of the forced ineligible-weight redistribution.

    Ref: §Diversification Constraint (p. 9)
    """
    return max(T_MIN, T_BASE - ineligible_weight)


# ---------------------------------------------------------------------------
# Core single-attempt solver
# ---------------------------------------------------------------------------


def _solve_once(
    carbon_intensities: np.ndarray,
    underlying_weights_eligible: np.ndarray,
    country_labels: np.ndarray,
    underlying_country_weights: dict[str, float],
    ig_labels: np.ndarray,
    underlying_ig_weights: dict[str, float],
    ineligible_weight: float,
    n_eligible: int,
    excess_stock_weight: float,
    stock_weight_multiplier: float,
    T: float,
) -> OptimizationResult:
    """
    Formulate and solve the carbon-intensity minimization problem once with
    the given constraint parameters.

    The problem is a Quadratically Constrained Quadratic Program (QCQP):
      - Linear objective (weighted sum of carbon intensities)
      - Linear constraints (weight bounds, country/IG bounds, sum-to-one)
      - Quadratic constraints (diversification, active share uses abs values
        which are linearized by CLARABEL internally)

    Parameters
    ----------
    carbon_intensities:
        Array CI of length n_eligible. CI[i] = tCO2e per $M revenue for stock i.
    underlying_weights_eligible:
        Array W of length n_eligible. W[i] = underlying index weight for stock i.
    country_labels / ig_labels:
        String arrays of length n_eligible identifying each stock's country / IG.
    underlying_country_weights / underlying_ig_weights:
        Pre-aggregated underlying weight totals per country / industry group,
        computed over ALL underlying stocks (including ineligible ones).
    ineligible_weight:
        Sum of underlying weights for stocks excluded from the eligible universe.
        This weight must be redistributed among eligible stocks.
    n_eligible:
        Number of eligible stocks (length of the optimization variable).
    excess_stock_weight / stock_weight_multiplier:
        Control the per-stock weight bounds (may be relaxed between attempts).
    T:
        Active-share budget parameter (may be relaxed between attempts).

    Returns
    -------
    OptimizationResult with weights=None if the solver could not find a solution.
    """
    # w[i] is the optimized weight for eligible stock i (decision variable)
    # nonneg=True enforces w[i] ≥ 0 implicitly without an explicit constraint
    w = cp.Variable(n_eligible, nonneg=True)

    # Shorthand alias for the underlying weight vector
    W = underlying_weights_eligible

    # -----------------------------------------------------------------------
    # Objective: minimize weighted-average carbon intensity
    # Ref: §The Optimization Objective Function (p. 8)
    #
    # minimize  Σᵢ (w[i] × CI[i])
    #
    # This is a dot product, expressed as carbon_intensities @ w in cvxpy.
    # The optimizer will overweight low-CI stocks and underweight high-CI ones,
    # subject to the constraints below preventing excessive deviation from the
    # benchmark.
    # -----------------------------------------------------------------------
    objective = cp.Minimize(carbon_intensities @ w)

    constraints = []

    # -----------------------------------------------------------------------
    # Constraint 1: Weights sum to 1.0
    #
    # The eligible stocks must account for 100% of the index. Ineligible stocks
    # have been removed and their underlying weights reallocated here, so the
    # total optimized weight must equal 1 (not 1 − ineligible_weight).
    # -----------------------------------------------------------------------
    constraints.append(cp.sum(w) == 1.0)

    # -----------------------------------------------------------------------
    # Constraint 2: Per-stock weight bounds
    # Ref: §Optimization Constraints, Stock weight Upper/Lower Bound (p. 8-9)
    #
    # Upper bound per stock:  min(W[i] × multiplier, W[i] + excess)
    #   The multiplier cap prevents any stock from growing to an unreasonable
    #   multiple of its benchmark weight. The additive cap limits absolute
    #   deviation regardless of how small W[i] is.
    #
    # Lower bound per stock:  max(0, W[i] − excess)
    #   A stock can be underweighted by at most `excess` percentage points,
    #   but never below zero (no shorting).
    # -----------------------------------------------------------------------
    stock_ub = np.minimum(W * stock_weight_multiplier, W + excess_stock_weight)
    stock_lb = np.maximum(0.0, W - excess_stock_weight)
    constraints.append(w <= stock_ub)
    constraints.append(w >= stock_lb)

    # -----------------------------------------------------------------------
    # Constraint 3: Country weight bounds
    # Ref: §Optimization Constraints, Country Weight Lower/Upper Bound (p. 8)
    #
    # For each country c:
    #   max(W_c − 5%, W_c × 0.75) ≤ Σ w[i] (where stock i is in country c)
    #                              ≤ min(W_c + 5%, W_c × 1.25)
    #
    # The two-part formula means:
    #   - For large countries (W_c > 20%), the ±5pp absolute band is binding.
    #   - For small countries (W_c < 20%), the ×0.75/×1.25 scale is binding,
    #     giving those countries a proportionally wider absolute band.
    #
    # Note: W_c here is the country's weight in the FULL underlying index
    # (including ineligible stocks), so the bounds are relative to the
    # benchmark allocation, not just the eligible subset.
    # -----------------------------------------------------------------------
    for country, uw in underlying_country_weights.items():
        # Only apply if at least one eligible stock belongs to this country
        mask = country_labels == country
        if not mask.any():
            continue
        lb = max(uw - COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_LB)
        ub = min(uw + COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_UB)
        country_w = cp.sum(w[mask])
        constraints.append(country_w >= lb)
        constraints.append(country_w <= ub)

    # -----------------------------------------------------------------------
    # Constraint 4: Industry Group weight bounds
    # Ref: §Optimization Constraints, Industry Group Lower/Upper Bound (p. 8)
    #
    # Upper bound:  min(W_ig + 5%, W_ig × 1.25)  — same formula as country.
    #
    # Lower bound is more complex:
    #   min(
    #     Σ W[i] (eligible in ig) × multiplier,   ← max the group CAN absorb
    #     max(W_ig − 5%, W_ig × 0.75)              ← normal band lower bound
    #   )
    #
    # The first term caps the lower bound at what eligible stocks in the group
    # could physically hold if each reached its individual upper bound. This
    # prevents an infeasible lower bound when few eligible stocks remain in a
    # large industry group after exclusions.
    # -----------------------------------------------------------------------
    for ig, uw in underlying_ig_weights.items():
        mask = ig_labels == ig
        if not mask.any():
            continue
        # Maximum weight the eligible stocks in this IG could collectively absorb
        eligible_sum_in_ig = float(np.sum(W[mask])) * stock_weight_multiplier
        # Standard lower bound formula from the methodology
        standard_lb = max(uw - COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_LB)
        # Take the lesser: the IG lower bound cannot exceed capacity of eligible stocks
        ig_lb = min(eligible_sum_in_ig, standard_lb)
        ig_ub = min(uw + COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_UB)
        ig_w = cp.sum(w[mask])
        constraints.append(ig_w >= ig_lb)
        constraints.append(ig_w <= ig_ub)

    # -----------------------------------------------------------------------
    # Constraint 5: Diversification Constraint (quadratic)
    # Ref: §Optimization Constraints (p. 9)
    #
    # Limits how concentrated the active bets are. Formally:
    #
    #   Σᵢ (W'[i] − w[i])² / W'[i]  ≤  Σᵢ (2T + ineligible/N)² / W'[i]
    #
    # where W'[i] = W[i] + ineligible_weight / N_eligible
    #
    # W'[i] is the "adjusted underlying weight": each eligible stock receives a
    # proportional share of the ineligible weight added to its benchmark weight.
    # This reflects the fact that the optimizer must redistribute ineligible weight
    # and the constraint should be evaluated against the adjusted starting point.
    #
    # The left-hand side is a weighted sum of squared active weights — it measures
    # how unevenly the active bets are distributed across stocks. Small values
    # mean bets are spread widely; large values mean they are concentrated.
    #
    # The right-hand side is the value the LHS would take if every eligible stock
    # had an equal active weight of exactly (2T + ineligible/N). This is the
    # "uniform distribution" benchmark — a theoretical upper bound on how unequal
    # the distribution of active weights can be.
    #
    # The constraint is quadratic (convex), so it is valid for CLARABEL/SCS.
    # -----------------------------------------------------------------------
    # Adjusted underlying weight per eligible stock
    W_prime = W + ineligible_weight / n_eligible

    # Right-hand side: compute per-stock terms and sum to a scalar constant
    rhs_per_stock = (2 * T + ineligible_weight / n_eligible) ** 2 / W_prime
    rhs_div = float(np.sum(rhs_per_stock))

    # Left-hand side: quadratic cvxpy expression over variable w
    lhs_div = cp.sum(cp.square(w - W_prime) / W_prime)

    constraints.append(lhs_div <= rhs_div)

    # -----------------------------------------------------------------------
    # Constraint 6: Absolute Weight Deviation (Active Share) Constraint
    # Ref: §Optimization Constraints (p. 9)
    #
    #   (1/2) × Σ |W[i] − w[i]|  ≤  T + ineligible_weight
    #          (all underlying i)
    #
    # "Active share" is a standard portfolio metric defined as half the sum of
    # absolute weight differences between the portfolio and its benchmark.
    # Active share = 0 means the portfolio perfectly replicates the benchmark.
    # Active share = 1 means the portfolio shares no holdings with the benchmark.
    #
    # The budget on the right-hand side is T + ineligible_weight:
    #   - T is the base active-share budget (default 25%)
    #   - ineligible_weight is the forced deviation caused by mandatory exclusions
    #
    # The sum is over ALL underlying stocks, not just eligible ones. For ineligible
    # stocks w[i] = 0, so |W[i] − 0| = W[i]. Their total contribution to the
    # left-hand side equals ineligible_weight, which then cancels with the
    # ineligible_weight on the right-hand side, leaving the budget T for the
    # eligible-stock deviations.
    #
    # In code we separate the two contributions:
    #   eligible part:   cp.sum(cp.abs(W - w))          (cvxpy variable expression)
    #   ineligible part: ineligible_weight               (scalar constant)
    # -----------------------------------------------------------------------
    active_share_rhs = T + ineligible_weight
    constraints.append(
        (cp.sum(cp.abs(W - w)) + ineligible_weight) / 2 <= active_share_rhs
    )

    # -----------------------------------------------------------------------
    # Solve
    # -----------------------------------------------------------------------
    problem = cp.Problem(objective, constraints)
    try:
        with warnings.catch_warnings():
            # Suppress solver verbosity warnings that occasionally surface
            # through cvxpy's internal logging
            warnings.simplefilter("ignore")
            problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        # Solver raised an unrecoverable exception (e.g. numerical failure)
        return OptimizationResult(weights=None, status="solver_error", relaxation_level=0)

    if w.value is None:
        # Solver ran but could not find a feasible/optimal solution
        # (status will be 'infeasible', 'unbounded', etc.)
        return OptimizationResult(weights=None, status=problem.status, relaxation_level=0)

    return OptimizationResult(
        weights=np.array(w.value),
        status=problem.status,
        relaxation_level=0,  # caller sets the actual level after success
    )


# ---------------------------------------------------------------------------
# Public optimizer entry point with relaxation hierarchy
# ---------------------------------------------------------------------------


def optimize(
    eligible_df: pd.DataFrame,
    all_underlying_weights: dict[str, float],
) -> OptimizationResult:
    """
    Run the carbon-intensity minimization with the constraint-relaxation hierarchy.

    If the optimizer cannot find a feasible solution at the base parameters,
    constraints are progressively relaxed in the order specified by the methodology
    until a solution is found. Each attempt uses the same solver but with looser
    bounds, giving the optimizer more room to manoeuvre.

    Relaxation hierarchy (Ref: §Constraint Relaxation Hierarchy, p. 9):
      Level 0 — base parameters (no relaxation)
      Level 1 — stock weight bounds loosened: excess 2%→4%, multiplier 10×→20×
      Level 2 — T reduced to its floor (5%), loosening the diversification and
                 active-share constraints
      Levels 3-4 — same parameters as level 2; in a full implementation these
                    would progressively relax IG then country bounds separately.
                    Here both are already at max-relaxation from level 1 onwards.

    Hard constraints never relaxed:
      - Minimum stock weight threshold (1 bps) — enforced by post-processing.

    Fallback:
      If all relaxation levels fail, the underlying index weights (rescaled to
      the eligible universe) are returned. This preserves relative benchmark
      exposures and avoids returning an empty result.

    Parameters
    ----------
    eligible_df:
        DataFrame of stocks that survived the eligibility filters.
    all_underlying_weights:
        Benchmark weights for ALL stocks including ineligible ones.
        The ineligible weight total is derived by subtracting eligible tickers.

    Returns
    -------
    OptimizationResult — always has non-None weights (falls back to benchmark
    proportional weights if all solve attempts fail).
    """
    eligible_tickers = eligible_df["ticker"].tolist()
    all_tickers = list(all_underlying_weights.keys())

    # Identify ineligible tickers (present in the underlying but excluded from eligible)
    ineligible_tickers = [t for t in all_tickers if t not in eligible_tickers]
    # Sum the benchmark weights of all excluded stocks — this is the "forced
    # deviation" that must be redistributed to eligible stocks.
    ineligible_weight = sum(all_underlying_weights.get(t, 0.0) for t in ineligible_tickers)

    n_eligible = len(eligible_tickers)
    if n_eligible == 0:
        # No eligible stocks — cannot form a portfolio
        return OptimizationResult(
            weights=np.array([]),
            status="no_eligible_stocks",
            relaxation_level=0,
        )

    # Underlying benchmark weights for the eligible stocks only,
    # in the same order as eligible_tickers
    W_eligible = np.array(
        [all_underlying_weights.get(t, 0.0) for t in eligible_tickers]
    )

    # Build the carbon intensity vector, imputing missing values with group medians
    ci_series = _impute_carbon_intensities(
        eligible_df.set_index("ticker")["carbon_intensity"],
        eligible_df.set_index("ticker")["gics_industry_group"],
    )
    # Reorder to match eligible_tickers order
    CI = np.array([ci_series[t] for t in eligible_tickers])

    # Country and industry group label arrays (same order as eligible_tickers)
    country_labels = np.array(
        eligible_df.set_index("ticker").loc[eligible_tickers, "country"].tolist()
    )
    ig_labels = np.array(
        eligible_df.set_index("ticker")
        .loc[eligible_tickers, "gics_industry_group"]
        .tolist()
    )

    # Aggregate underlying weights by country and industry group.
    # IMPORTANT: these are computed over ELIGIBLE stocks only, because we only
    # have full metadata (country, IG) for eligible stocks. Ineligible stocks
    # are not in eligible_df, so their country/IG contributions are omitted.
    # This approximation is acceptable — constraints are still centred on the
    # eligible benchmark allocation.
    underlying_country_weights: dict[str, float] = {}
    underlying_ig_weights: dict[str, float] = {}

    for t in eligible_tickers:
        row = eligible_df[eligible_df["ticker"] == t]
        if row.empty:
            continue
        country = row.iloc[0]["country"]
        ig = row.iloc[0]["gics_industry_group"]
        underlying_country_weights[country] = (
            underlying_country_weights.get(country, 0.0)
            + all_underlying_weights[t]
        )
        underlying_ig_weights[ig] = (
            underlying_ig_weights.get(ig, 0.0) + all_underlying_weights[t]
        )

    # Compute T — tightens when many stocks are ineligible (large ineligible_weight)
    T_base = _compute_T(ineligible_weight)

    # Each entry is (excess_stock_weight, stock_weight_multiplier, T).
    # Later levels have looser parameters to help the solver find a feasible point.
    relaxation_schedule = [
        # Level 0: base parameters — tightest constraints, smallest feasible region
        (EXCESS_STOCK_WEIGHT_BASE, STOCK_WEIGHT_MULTIPLIER_BASE, T_base),
        # Level 1: widen stock weight bounds (excess 2%→4%, multiplier 10×→20×)
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_base),
        # Level 2: also reduce T to 5%, widening diversification and active-share budgets
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),
        # Level 3: same params — IG bounds already at max relaxation from level 1
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),
        # Level 4: same params — country bounds already at max relaxation from level 1
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),
    ]

    for level, (excess_sw, sw_mult, T) in enumerate(relaxation_schedule):
        result = _solve_once(
            carbon_intensities=CI,
            underlying_weights_eligible=W_eligible,
            country_labels=country_labels,
            underlying_country_weights=underlying_country_weights,
            ig_labels=ig_labels,
            underlying_ig_weights=underlying_ig_weights,
            ineligible_weight=ineligible_weight,
            n_eligible=n_eligible,
            excess_stock_weight=excess_sw,
            stock_weight_multiplier=sw_mult,
            T=T,
        )
        if result.weights is not None and result.status in ("optimal", "optimal_inaccurate"):
            # Record which relaxation level was required to find a solution
            result.relaxation_level = level
            return result

    # All relaxation levels exhausted — fall back to rescaled benchmark weights.
    # This preserves the relative benchmark allocation and ensures the caller
    # always receives a usable weight vector, even if it is not carbon-minimized.
    return OptimizationResult(
        weights=W_eligible / W_eligible.sum(),
        status="fallback_equal_weight",
        relaxation_level=len(relaxation_schedule),
    )


# ---------------------------------------------------------------------------
# Post-processing: minimum weight threshold
# ---------------------------------------------------------------------------


def apply_minimum_weight_threshold(
    weights: np.ndarray,
    tickers: list[str],
    min_weight: float = MIN_STOCK_WEIGHT,
) -> dict[str, float]:
    """
    Remove stocks whose optimized weight falls below the minimum threshold (1 bps)
    and redistribute their weight proportionally to the remaining stocks.

    Why post-processing instead of a hard constraint in the optimizer?
        The methodology specifies this as a clean-up step performed after the
        optimizer has run, not as a constraint within the optimization itself.
        Adding it as a constraint would complicate the problem with integer-like
        logic (a stock is either in or out), whereas post-processing keeps the
        optimization convex.

    Redistribution logic:
        Sub-threshold weight is pooled and then distributed across stocks still
        above the threshold in proportion to their current weights. This is
        iterated because redistributing weight can push previously-borderline
        stocks back above the threshold, but may also cause edge cases in the
        opposite direction — the loop repeats until no stocks remain below the
        threshold or the weight pool is exhausted.

    Final normalization:
        After redistribution the weights are re-normalized to sum exactly to 1.0,
        correcting for any floating-point drift accumulated during iteration.

    Ref: §Optimization Constraints, Minimum Stock Weight Lower Threshold
         footnote 3 (p. 9)

    Parameters
    ----------
    weights:
        Raw optimizer output weights, same length as tickers.
    tickers:
        Ticker symbols corresponding to each weight.
    min_weight:
        Threshold below which a weight is zeroed (default: 0.0001 = 1 bps).

    Returns
    -------
    Dict mapping ticker → final weight, containing only stocks with weight > 0.
    """
    w = weights.copy()
    tickers = list(tickers)

    # Iterate to handle cascading effects of redistribution
    for _ in range(len(tickers)):
        below = w < min_weight
        if not below.any():
            # All remaining stocks are above the threshold — done
            break

        # Pool the weight from all sub-threshold stocks
        redistributed = w[below].sum()
        w[below] = 0.0  # zero out sub-threshold stocks

        # Redistribute proportionally among stocks still above zero
        above = w > 0
        if above.any():
            # Each surviving stock receives a proportional share of the
            # redistributed weight, preserving their relative allocations
            w[above] += redistributed * (w[above] / w[above].sum())

    # Re-normalize to correct floating-point drift so weights sum exactly to 1
    total = w.sum()
    if total > 0:
        w /= total

    # Return only stocks with a non-zero final weight
    return {t: float(wi) for t, wi in zip(tickers, w) if wi > 0}
