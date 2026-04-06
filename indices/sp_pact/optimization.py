"""
Portfolio optimization for the S&P PAB ESG & S&P CTB Indices.

The optimizer minimizes tracking error from the parent index (sum of squared
relative weight deviations) while satisfying a set of climate, ESG, and
diversification constraints that keep the resulting portfolio aligned with
its decarbonization target.

Unlike the S&P Carbon Aware index (which minimizes WACI directly), the PACT
indices treat WACI as a constraint rather than the objective. The objective
is to stay as close to the parent index as possible while meeting:
  - A WACI reduction target (30% CTB / 50% PAB, with 5% buffer)
  - A 7% annual decarbonization trajectory from the anchor year
  - SBTI-committed company weight >= 120% of the parent index
  - Weighted-average ESG score >= the parent index
  - High climate impact sector weight >= the parent index
  - Non-disclosing company weight <= 110% of the parent index
  - Per-stock weight bounds (+-2% from parent, capped at max(5%, parent weight))

All constraints are formulated as convex expressions and solved by CLARABEL
via the cvxpy modelling layer. If the problem is infeasible at the default
parameters, a relaxation hierarchy progressively loosens stock weight bounds
until a solution is found.

Ref: "S&P PAB ESG and S&P CTB Indices Methodology" (S&P Dow Jones Indices)
     §Index Construction (pp. 13-16), §Constraint Relaxation Hierarchy (pp. 16-17)
"""

from __future__ import annotations

import warnings
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .models import Variant

# ---------------------------------------------------------------------------
# Methodology constants
# Ref: §Constituent Weighting (p. 13), §Optimization Constraints (pp. 14-16)
# ---------------------------------------------------------------------------

# WACI reduction targets — the minimum % reduction in weighted-average carbon
# intensity the optimized index must achieve relative to the parent index.
# CTB: 30% reduction (Climate Transition Benchmark — moderate target)
# PAB: 50% reduction (Paris-Aligned Benchmark — aggressive target)
WACI_REDUCTION = {
    Variant.CTB: 0.30,   # 30% reduction
    Variant.PAB: 0.50,   # 50% reduction
}

# 5% buffer (multiply the WACI target by 0.95) to prevent marginal breaches
# due to market movements between rebalancing dates. The buffer means the
# optimized WACI must be at most 95% of the theoretical target, providing
# a margin of safety against drift.
# Ref: §WACI Constraint — Buffer (p. 14)
WACI_BUFFER = 0.95       # 5% margin for drift

# Annual decarbonization rate — the index's WACI target decreases by 7% per
# year (compounded) from the anchor date, ensuring continuous improvement.
# After Q quarters from anchor:
#   trajectory_target = anchor_waci * (1 - 0.07)^(Q/4) * buffer
# Ref: §Decarbonization Trajectory (p. 14)
DECARBONIZATION_RATE = 0.07  # 7% per year

# Per-stock weight deviation band — each stock's optimized weight can deviate
# at most +-2% from its parent index weight.
# Upper bound = max(W[i] + 2%, 5%)   (floor of 5% prevents tiny stocks from
#                                      being locked to near-zero weight)
# Lower bound = max(0, W[i] - 2%)    (no shorting allowed)
# Ref: §Diversification Constraints — Stock Weight Bounds (p. 15)
RELATIVE_WEIGHT_BAND = 0.02  # +-2% from parent weight
ABSOLUTE_MAX_WEIGHT = 0.05   # max(5%, parent weight) as upper bound floor

# SBTI (Science Based Targets initiative) weight multiplier.
# The total weight of SBTI-committed companies in the optimized index must be
# >= 120% of their total weight in the parent index. This incentivises the
# optimizer to overweight companies with validated climate targets.
# Ref: §SBTI Weight Constraint (p. 15)
SBTI_WEIGHT_MULTIPLIER = 1.20  # >= 120% of underlying SBTI weight

# ESG score floor percentile — used in PAB variant to remove bottom 20%
# of companies by ESG score before optimization (not used as a constraint
# in this implementation; the ESG constraint is applied as a weighted-average
# floor instead).
ESG_FLOOR_PERCENTILE = 0.20  # Remove bottom 20% by count (PAB)

# Non-disclosing companies: those without carbon emissions data. Their total
# weight in the optimized index cannot exceed 110% of their weight in the
# parent index, preventing the optimizer from overweighting companies that
# lack transparency on emissions.
# Ref: §Non-Disclosing Company Constraint (p. 16)
NON_DISCLOSING_MAX_RATIO = 1.10  # <= 110% of underlying weight

# Minimum stock weight — enforced by post-processing rather than inside the
# optimizer. Any stock whose optimized weight falls below 1 bps (0.01%) is
# zeroed and its weight is redistributed to larger stocks.
# Ref: §Minimum Stock Weight Lower Threshold (p. 16)
MIN_STOCK_WEIGHT = 0.0001  # 1 basis point

# High climate impact GICS sectors — the methodology requires that the total
# weight of stocks in these sectors in the optimized index >= their total
# weight in the parent index. This prevents the optimizer from simply
# dumping high-emission sectors to meet the WACI target, which would
# reduce the index's real-economy climate impact.
# Ref: §High Climate Impact Sector Constraint (p. 15)
HIGH_CLIMATE_IMPACT_SECTORS = {
    "Energy", "Materials", "Utilities", "Transportation",
    "Capital Goods", "Automobiles & Components",
}


# ---------------------------------------------------------------------------
# Optimization result model
# ---------------------------------------------------------------------------


class OptimizationResult(BaseModel):
    """
    Raw output from a single cvxpy solve attempt.

    weights: optimized weights over eligible stocks (None if infeasible).
    status:  cvxpy solver status string (e.g. 'optimal', 'infeasible').
    relaxation_level: 0 = no relaxation applied; higher = more constraints relaxed.
    """

    # numpy arrays are not natively validated by Pydantic, so we opt in to
    # arbitrary type support for this model only.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Optional[np.ndarray]
    status: str
    relaxation_level: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_waci(weights: np.ndarray, carbon_intensities: np.ndarray) -> float:
    """
    Compute weighted-average carbon intensity, ignoring NaN entries.

    WACI = sum(w[i] * CI[i]) / sum(w[i])  for all i where CI[i] is valid.

    Stocks with missing CI (NaN) are excluded from both numerator and
    denominator so the result is not biased by missing coverage.

    Returns 0.0 if no valid data exists (defensive fallback).
    """
    valid = ~np.isnan(carbon_intensities)
    if not valid.any():
        return 0.0
    w_valid = weights[valid]
    ci_valid = carbon_intensities[valid]
    total_w = w_valid.sum()
    if total_w == 0:
        return 0.0
    return float(np.dot(w_valid, ci_valid) / total_w)


def _impute_carbon_intensities(
    ci_series: pd.Series, ig_series: pd.Series
) -> pd.Series:
    """
    Fill missing carbon intensities with the GICS Industry Group median.

    Why the group median?
        The methodology assigns missing-coverage companies the median carbon
        intensity of their GICS Industry Group. This is a neutral assumption
        that neither rewards nor penalises companies for lacking Trucost data --
        they're treated as a typical representative of their sector.

    If an entire industry group has no coverage (all NaN), the global median
    across all stocks in the eligible universe is used as a final fallback to
    ensure the CI vector is always fully populated.

    Ref: §Carbon Intensity Imputation (p. 13)
    """
    filled = ci_series.copy()
    for group in ig_series.unique():
        mask = ig_series == group
        group_ci = ci_series[mask]
        # Compute the median of all non-NaN values in the group
        group_median = group_ci.median()
        if pd.isna(group_median):
            # No coverage anywhere in this group -> fall back to global median
            group_median = ci_series.median()
        # Only fill the NaN entries; leave existing values untouched
        needs_fill = mask & filled.isna()
        filled[needs_fill] = group_median
    return filled


# ---------------------------------------------------------------------------
# Core single-attempt solver
# ---------------------------------------------------------------------------


def _solve_once(
    CI: np.ndarray,
    W: np.ndarray,
    underlying_waci: float,
    waci_target: float,
    sector_labels: np.ndarray,
    underlying_sector_weights: dict[str, float],
    country_labels: np.ndarray,
    underlying_country_weights: dict[str, float],
    sbti_mask: np.ndarray,
    underlying_sbti_weight: float,
    esg_scores: np.ndarray,
    underlying_wa_esg: float,
    non_disclosing_mask: np.ndarray,
    underlying_non_disclosing_weight: float,
    n: int,
    relative_band: float,
    absolute_max: float,
) -> OptimizationResult:
    """
    Formulate and solve the PACT optimization problem once with the given
    constraint parameters.

    The problem is a Quadratic Program (QP):
      - Quadratic objective (sum of squared relative weight deviations)
      - Linear constraints (weight bounds, WACI, sector, SBTI, ESG, etc.)
      - The ESG constraint is bilinear in form but is linearised because
        the underlying weighted-average ESG score is a known constant.

    Parameters
    ----------
    CI:
        Array of total carbon intensities (Scope 1+2+3), length n.
    W:
        Array of parent index weights (rescaled to eligible universe), length n.
    underlying_waci:
        WACI of the parent index (used to compute the reduction target).
    waci_target:
        Maximum allowable WACI for the optimized index (after reduction and buffer).
    sector_labels / country_labels:
        String arrays of length n identifying each stock's GICS sector / country.
    underlying_sector_weights / underlying_country_weights:
        Pre-aggregated parent weight totals per sector / country.
    sbti_mask:
        Boolean array — True for stocks with validated SBTI targets.
    underlying_sbti_weight:
        Total parent weight of SBTI-committed companies.
    esg_scores:
        Array of S&P Global ESG scores (may contain NaN), length n.
    underlying_wa_esg:
        Weighted-average ESG score of the parent index.
    non_disclosing_mask:
        Boolean array — True for stocks without carbon emissions disclosure.
    underlying_non_disclosing_weight:
        Total parent weight of non-disclosing companies.
    n:
        Number of eligible stocks.
    relative_band / absolute_max:
        Per-stock weight bound parameters (may be relaxed between attempts).

    Returns
    -------
    OptimizationResult with weights=None if the solver could not find a solution.
    """
    # w[i] is the optimized weight for eligible stock i (decision variable).
    # nonneg=True enforces w[i] >= 0 implicitly (no short selling).
    w = cp.Variable(n, nonneg=True)

    # -------------------------------------------------------------------
    # Objective: minimize tracking error (sum of squared relative deviations)
    # Ref: §Constituent Weighting — Objective Function (p. 13)
    #
    #   minimize  sum_i  (w[i] - W[i])^2 / W[i]
    #
    # This is a chi-squared-like measure that penalises relative deviations
    # from the parent weight. Dividing by W[i] means that a 1% deviation
    # in a large-cap stock costs the same as a 1% deviation in a small-cap
    # stock, keeping the optimizer from concentrating all its active bets
    # in small stocks where absolute deviations are cheap.
    #
    # safe_W prevents division by zero for stocks with near-zero parent
    # weight (can occur after rescaling the eligible universe).
    # -------------------------------------------------------------------
    safe_W = np.maximum(W, 1e-10)
    objective = cp.Minimize(cp.sum(cp.square(w - W) / safe_W))

    constraints = []

    # -------------------------------------------------------------------
    # Constraint 1: Weights sum to 1.0
    #
    # The eligible stocks must account for 100% of the index. Ineligible
    # stocks have been removed and the parent weights rescaled, so the
    # total optimized weight must equal exactly 1.
    # -------------------------------------------------------------------
    constraints.append(cp.sum(w) == 1.0)

    # -------------------------------------------------------------------
    # Constraint 2: Per-stock weight bounds
    # Ref: §Diversification Constraints — Stock Weight Bounds (p. 15)
    #
    # Upper bound:  max(W[i] + relative_band, absolute_max)
    #   The relative_band (default +-2%) limits deviation from the parent
    #   weight. The absolute_max floor (default 5%) ensures that even
    #   very small stocks can be meaningfully overweighted if needed to
    #   satisfy other constraints.
    #
    # Lower bound:  max(0, W[i] - relative_band)
    #   A stock can be underweighted by at most relative_band percentage
    #   points, but never below zero (no shorting).
    # -------------------------------------------------------------------
    stock_ub = np.maximum(W + relative_band, absolute_max)
    stock_lb = np.maximum(0.0, W - relative_band)
    constraints.append(w <= stock_ub)
    constraints.append(w >= stock_lb)

    # -------------------------------------------------------------------
    # Constraint 3: WACI constraint
    # Ref: §WACI Reduction Constraint (p. 14)
    #
    #   sum_i (w[i] * CI[i])  <=  waci_target
    #
    # The waci_target incorporates both the static reduction (30% or 50%)
    # and the 7% annual decarbonization trajectory, whichever is stricter.
    # The 5% buffer has already been applied to the target value.
    #
    # This is a linear constraint (dot product of variable w with constant
    # CI vector), so it is straightforward for the QP solver.
    # -------------------------------------------------------------------
    constraints.append(CI @ w <= waci_target)

    # -------------------------------------------------------------------
    # Constraint 4: High climate impact sector weight
    # Ref: §High Climate Impact Sector Constraint (p. 15)
    #
    #   For each high-climate-impact sector s:
    #     sum(w[i] for i in sector s)  >=  underlying_sector_weight[s]
    #
    # This prevents the optimizer from simply dumping high-emission sectors
    # (Energy, Materials, Utilities, Transportation, Capital Goods,
    # Automobiles & Components) to meet the WACI target. Without this
    # constraint, the optimizer would underweight these sectors, reducing
    # the index's ability to influence corporate behaviour through
    # capital allocation.
    # -------------------------------------------------------------------
    for sector, uw in underlying_sector_weights.items():
        if sector in HIGH_CLIMATE_IMPACT_SECTORS:
            mask = sector_labels == sector
            if mask.any():
                constraints.append(cp.sum(w[mask]) >= uw)

    # -------------------------------------------------------------------
    # Constraint 5: Country weight bounds
    # Ref: §Diversification Constraints — Country Bounds (p. 15)
    #
    #   For each country c:
    #     max(0, W_c - 5%)  <=  sum(w[i] for i in country c)  <=  W_c + 5%
    #
    # A +-5% absolute band around each country's parent weight prevents
    # the optimizer from creating extreme country tilts while meeting
    # the carbon target. This is wider than the per-stock +-2% band
    # because country-level deviations aggregate many individual bets.
    # -------------------------------------------------------------------
    for country, uw in underlying_country_weights.items():
        mask = country_labels == country
        if not mask.any():
            continue
        constraints.append(cp.sum(w[mask]) >= max(0, uw - 0.05))
        constraints.append(cp.sum(w[mask]) <= uw + 0.05)

    # -------------------------------------------------------------------
    # Constraint 6: SBTI weight constraint
    # Ref: §SBTI Weight Constraint (p. 15)
    #
    #   sum(w[i] for SBTI-committed i)  >=  1.20 * underlying_sbti_weight
    #
    # Companies with validated Science Based Targets must collectively
    # receive at least 120% of their parent index weight. This rewards
    # companies that have made credible climate commitments and nudges
    # capital toward transition leaders.
    #
    # Only applied when there are SBTI companies AND they have positive
    # parent weight (otherwise the constraint is vacuous or trivially met).
    # -------------------------------------------------------------------
    if sbti_mask.any() and underlying_sbti_weight > 0:
        constraints.append(
            cp.sum(w[sbti_mask]) >= SBTI_WEIGHT_MULTIPLIER * underlying_sbti_weight
        )

    # -------------------------------------------------------------------
    # Constraint 7: ESG score constraint
    # Ref: §ESG Score Constraint (p. 15)
    #
    #   sum(esg[i] * w[i])  >=  underlying_wa_esg * sum(w[i])
    #       (summed over stocks with valid ESG scores)
    #
    # The optimized portfolio's weighted-average ESG score must be at least
    # as high as the parent index's. This prevents the optimizer from
    # sacrificing ESG quality to meet carbon targets.
    #
    # Formally this is:  E[esg_scores] @ w[valid] >= underlying_wa_esg * sum(w[valid])
    # which is a linear constraint in w (underlying_wa_esg is a constant).
    #
    # Stocks without ESG scores are excluded from both sides of the
    # inequality so they neither help nor hurt the ESG average.
    # -------------------------------------------------------------------
    valid_esg = ~np.isnan(esg_scores)
    if valid_esg.any() and underlying_wa_esg > 0:
        constraints.append(esg_scores[valid_esg] @ w[valid_esg] >= underlying_wa_esg * cp.sum(w[valid_esg]))

    # -------------------------------------------------------------------
    # Constraint 8: Non-disclosing company weight cap
    # Ref: §Non-Disclosing Company Constraint (p. 16)
    #
    #   sum(w[i] for non-disclosing i)  <=  1.10 * underlying_non_disclosing_weight
    #
    # Companies without carbon emissions disclosure cannot collectively
    # receive more than 110% of their parent index weight. This prevents
    # the optimizer from "hiding" carbon exposure by overweighting stocks
    # whose emissions are unknown and therefore imputed at the (potentially
    # favourable) group median.
    #
    # Only applied when there are non-disclosing companies with positive
    # parent weight.
    # -------------------------------------------------------------------
    if non_disclosing_mask.any() and underlying_non_disclosing_weight > 0:
        constraints.append(
            cp.sum(w[non_disclosing_mask]) <= NON_DISCLOSING_MAX_RATIO * underlying_non_disclosing_weight
        )

    # -------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------
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
    variant: Variant,
    rebalance_quarter: int = 0,
    anchor_waci: Optional[float] = None,
) -> OptimizationResult:
    """
    Run the PACT optimization with the constraint-relaxation hierarchy.

    If the optimizer cannot find a feasible solution at the base parameters,
    constraints are progressively relaxed until a solution is found. Each
    attempt uses the same solver but with looser stock weight bounds, giving
    the optimizer more room to manoeuvre.

    Relaxation hierarchy (Ref: §Constraint Relaxation Hierarchy, pp. 16-17):
      Level 0 — base parameters: +-2% relative band, 5% absolute max
      Level 1 — widen stock bounds: +-4% relative band, 10% absolute max
      Level 2 — further widen:     +-6% relative band, 15% absolute max

    Climate constraints (WACI target, SBTI weight, ESG score, sector exposure,
    non-disclosing cap) are NEVER relaxed — they are hard requirements of the
    methodology. Only the diversification (stock weight) bounds are loosened.

    Fallback:
        If all relaxation levels fail, the parent index weights (rescaled to
        the eligible universe) are returned. This preserves relative benchmark
        exposures and avoids returning an empty result, but the portfolio may
        not meet the WACI target.

    Parameters
    ----------
    eligible_df:
        DataFrame of stocks that survived the eligibility filters.
    all_underlying_weights:
        Benchmark weights for ALL stocks including ineligible ones.
    variant:
        CTB or PAB — determines the WACI reduction target (30% vs 50%).
    rebalance_quarter:
        Number of quarters since the anchor date. Used to compute the 7%
        annual decarbonization trajectory: target tightens by
        (1 - 0.07)^(quarters/4) each year.
    anchor_waci:
        WACI at the anchor date for the decarbonization trajectory. If None,
        uses the current underlying WACI as anchor (first rebalancing).

    Returns
    -------
    OptimizationResult — always has non-None weights (falls back to benchmark
    proportional weights if all solve attempts fail).

    Ref: §Index Construction (pp. 13-16)
    """
    eligible_tickers = eligible_df["ticker"].tolist()
    n = len(eligible_tickers)

    if n == 0:
        # No eligible stocks — cannot form a portfolio
        return OptimizationResult(weights=np.array([]), status="no_eligible_stocks", relaxation_level=0)

    # --- Prepare parent weight vector for eligible stocks ---
    # Extract weights for eligible tickers and rescale to sum to 1.0.
    # This redistribution accounts for the weight of excluded stocks.
    W = np.array([all_underlying_weights.get(t, 0.0) for t in eligible_tickers])
    W_sum = W.sum()
    if W_sum > 0:
        W = W / W_sum  # Rescale eligible weights to sum to 1

    # --- Carbon intensity vector (imputed) ---
    # Fill missing CI values with GICS Industry Group medians so the
    # optimizer has a complete vector for the WACI constraint.
    ci_series = _impute_carbon_intensities(
        eligible_df.set_index("ticker")["total_carbon_intensity"],
        eligible_df.set_index("ticker")["gics_industry_group"],
    )
    CI = np.array([ci_series.get(t, 0.0) for t in eligible_tickers])

    # --- Compute the underlying WACI ---
    # The WACI of the parent index (eligible universe only, at rescaled weights)
    # serves as the baseline for the reduction target calculation.
    underlying_waci = _compute_waci(
        np.array([all_underlying_weights.get(t, 0.0) for t in all_underlying_weights]),
        np.full(len(all_underlying_weights), np.nan),  # placeholder
    )
    # Use eligible-universe WACI as proxy (ineligible stocks are excluded)
    underlying_waci = float(np.dot(W, CI)) if W.sum() > 0 else 0.0

    # --- WACI target calculation ---
    # Two targets are computed; the binding (stricter) one is used:
    #
    # 1. Static target: underlying_waci * (1 - reduction) * buffer
    #    e.g. for PAB: WACI * 0.50 * 0.95 = 47.5% of parent WACI
    #
    # 2. Trajectory target (if anchor_waci provided):
    #    anchor_waci * (1 - 0.07)^(Q/4) * buffer
    #    This tightens by 7% per year from the anchor date.
    #
    # The min() ensures the index meets whichever target is stricter.
    # Ref: §WACI Reduction Target & Decarbonization Trajectory (p. 14)
    reduction = WACI_REDUCTION[variant]
    static_target = underlying_waci * (1 - reduction) * WACI_BUFFER

    if anchor_waci is not None:
        # Apply the 7% annual decarbonization trajectory
        trajectory_target = anchor_waci * (1 - DECARBONIZATION_RATE) ** (rebalance_quarter / 4) * WACI_BUFFER
        waci_target = min(static_target, trajectory_target)
    else:
        # First rebalancing: no anchor yet, use static target only
        waci_target = static_target

    # --- Sector and country label arrays ---
    # Same order as eligible_tickers for correct constraint indexing
    sector_labels = np.array(eligible_df.set_index("ticker").loc[eligible_tickers, "gics_sector"].tolist())
    country_labels = np.array(eligible_df.set_index("ticker").loc[eligible_tickers, "country"].tolist())

    # --- Aggregate parent weights by sector and country ---
    # These are computed over ELIGIBLE stocks only (using rescaled weights)
    # because ineligible stocks are not in eligible_df.
    underlying_sector_weights: dict[str, float] = {}
    underlying_country_weights: dict[str, float] = {}
    for i, t in enumerate(eligible_tickers):
        underlying_sector_weights[sector_labels[i]] = underlying_sector_weights.get(sector_labels[i], 0.0) + W[i]
        underlying_country_weights[country_labels[i]] = underlying_country_weights.get(country_labels[i], 0.0) + W[i]

    # --- SBTI mask ---
    # Boolean array identifying stocks with validated Science Based Targets.
    # The optimizer will ensure their collective weight >= 120% of parent.
    sbti_mask = np.array(eligible_df.set_index("ticker").loc[eligible_tickers, "has_sbti_target"].tolist(), dtype=bool)
    underlying_sbti_weight = float(W[sbti_mask].sum()) if sbti_mask.any() else 0.0

    # --- ESG scores ---
    # Weighted-average ESG of the parent index (eligible stocks only).
    # The optimizer must match or exceed this level.
    esg_scores = np.array(eligible_df.set_index("ticker").loc[eligible_tickers, "esg_score"].tolist(), dtype=float)
    valid_esg = ~np.isnan(esg_scores)
    underlying_wa_esg = float(np.dot(W[valid_esg], esg_scores[valid_esg]) / W[valid_esg].sum()) if valid_esg.any() and W[valid_esg].sum() > 0 else 0.0

    # --- Non-disclosing mask ---
    # Companies without carbon emissions data. Their collective weight is
    # capped at 110% of the parent to prevent overweighting opaque companies.
    non_disclosing_mask = ~np.array(eligible_df.set_index("ticker").loc[eligible_tickers, "has_carbon_coverage"].tolist(), dtype=bool)
    underlying_non_disclosing_weight = float(W[non_disclosing_mask].sum()) if non_disclosing_mask.any() else 0.0

    # --- Relaxation schedule ---
    # Progressively loosen stock weight bounds if the problem is infeasible.
    # Each level widens the per-stock deviation band and absolute max cap.
    # Climate and ESG constraints are NEVER relaxed.
    #
    # Level 0: base parameters (tightest feasible region)
    # Level 1: double the relative band and absolute max
    # Level 2: triple the relative band and absolute max
    #
    # Ref: §Constraint Relaxation Hierarchy (pp. 16-17)
    relaxation_schedule = [
        # Level 0: base — +-2% band, 5% absolute cap
        (RELATIVE_WEIGHT_BAND, ABSOLUTE_MAX_WEIGHT),
        # Level 1: widen — +-4% band, 10% absolute cap
        (RELATIVE_WEIGHT_BAND * 2, ABSOLUTE_MAX_WEIGHT * 2),
        # Level 2: further widen — +-6% band, 15% absolute cap
        (RELATIVE_WEIGHT_BAND * 3, ABSOLUTE_MAX_WEIGHT * 3),
    ]

    for level, (rel_band, abs_max) in enumerate(relaxation_schedule):
        result = _solve_once(
            CI=CI, W=W, underlying_waci=underlying_waci, waci_target=waci_target,
            sector_labels=sector_labels, underlying_sector_weights=underlying_sector_weights,
            country_labels=country_labels, underlying_country_weights=underlying_country_weights,
            sbti_mask=sbti_mask, underlying_sbti_weight=underlying_sbti_weight,
            esg_scores=esg_scores, underlying_wa_esg=underlying_wa_esg,
            non_disclosing_mask=non_disclosing_mask,
            underlying_non_disclosing_weight=underlying_non_disclosing_weight,
            n=n, relative_band=rel_band, absolute_max=abs_max,
        )
        if result.weights is not None and result.status in ("optimal", "optimal_inaccurate"):
            # Record which relaxation level was required to find a solution
            result.relaxation_level = level
            return result

    # --- Fallback: rescaled benchmark weights ---
    # All relaxation levels exhausted. Return the parent index weights
    # (rescaled to eligible universe) as a fallback. This preserves
    # relative benchmark exposures but may not meet the WACI target.
    return OptimizationResult(
        weights=W / W.sum() if W.sum() > 0 else W,
        status="fallback_benchmark",
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
    Remove stocks whose optimized weight falls below the minimum threshold
    (1 bps) and redistribute their weight proportionally to remaining stocks.

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
        stocks above the threshold, but may also cause edge cases in the
        opposite direction -- the loop repeats until no stocks remain below
        the threshold or the weight pool is exhausted.

    Final normalization:
        After redistribution, weights are re-normalized to sum exactly to 1.0,
        correcting for any floating-point drift accumulated during iteration.

    Ref: §Minimum Stock Weight Lower Threshold (p. 16)

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
    Dict mapping ticker -> final weight, containing only stocks with weight > 0.
    """
    w = weights.copy()

    # Iterate to handle cascading effects of redistribution
    for _ in range(len(tickers)):
        below = w < min_weight
        if not below.any():
            # All remaining stocks are above the threshold -- done
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
