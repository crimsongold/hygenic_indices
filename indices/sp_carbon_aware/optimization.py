"""
Portfolio optimization for the S&P Carbon Aware Index Series.

Minimizes the weighted-average carbon intensity of eligible constituents subject
to country, industry group, stock weight, diversification, and active-share
constraints. Implements the constraint-relaxation hierarchy from the methodology.

Ref: "S&P Carbon Aware Index Series Methodology" (March 2026),
     §Index Construction, §Optimization Constraints, §Constraint Relaxation Hierarchy
"""

from __future__ import annotations

import warnings
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


# ---------------------------------------------------------------------------
# Parameter defaults (hard-coded per the methodology)
# ---------------------------------------------------------------------------

# Stock weight multiplier (max = 10x underlying, relaxed up to 20x)
STOCK_WEIGHT_MULTIPLIER_BASE = 10
STOCK_WEIGHT_MULTIPLIER_MAX = 20

# Excess stock weight (±2%, relaxed up to ±4%)
EXCESS_STOCK_WEIGHT_BASE = 0.02
EXCESS_STOCK_WEIGHT_MAX = 0.04

# Country / Industry Group band
COUNTRY_IG_BAND = 0.05       # ±5%
COUNTRY_IG_SCALE_LB = 0.75   # lower: max(underlying - 5%, underlying × 0.75)
COUNTRY_IG_SCALE_UB = 1.25   # upper: min(underlying + 5%, underlying × 1.25)

# Minimum stock weight (hard constraint; not relaxed)
MIN_STOCK_WEIGHT = 0.0001    # 1 basis point = 0.01%

# Base active-share target
T_BASE = 0.25                # 25%
T_MIN = 0.05                 # 5% (lower floor for T)


class OptimizationResult(BaseModel):
    """Raw output from a single cvxpy solve attempt."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    weights: Optional[np.ndarray]  # optimized weights over eligible stocks
    status: str
    relaxation_level: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _impute_carbon_intensities(
    carbon_intensities: pd.Series, industry_groups: pd.Series
) -> pd.Series:
    """
    Assign median carbon intensity for each GICS Industry Group to stocks
    that are missing Trucost coverage.

    Ref: §Carbon Intensity footnote 2 (p. 8)
    """
    filled = carbon_intensities.copy()
    for group in industry_groups.unique():
        mask = industry_groups == group
        group_vals = carbon_intensities[mask]
        median = group_vals.median()
        if pd.isna(median):
            median = carbon_intensities.median()
        filled[mask & filled.isna()] = median
    return filled


def _country_ig_bounds(
    weights: np.ndarray, band: float, scale_lb: float, scale_ub: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return lower/upper bounds for country or industry group weights."""
    lb = np.maximum(weights - band, weights * scale_lb)
    ub = np.minimum(weights + band, weights * scale_ub)
    return lb, ub


def _compute_T(ineligible_weight: float) -> float:
    """
    T = max(5%, 25% − ineligible_weight).
    Ref: §Diversification Constraint (p. 9)
    """
    return max(T_MIN, T_BASE - ineligible_weight)


# ---------------------------------------------------------------------------
# Core optimization
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
    Attempt a single cvxpy solve with the given parameters.
    Returns an OptimizationResult (may have status != 'optimal').
    """
    w = cp.Variable(n_eligible, nonneg=True)
    W = underlying_weights_eligible  # shorthand

    # --- Objective: minimize weighted-average carbon intensity ---
    objective = cp.Minimize(carbon_intensities @ w)

    constraints = []

    # --- Weights sum to 1 (eligible stocks absorb ineligible weight) ---
    constraints.append(cp.sum(w) == 1.0)

    # --- Stock weight bounds ---
    # Upper: min(W_i × multiplier, W_i + excess)
    stock_ub = np.minimum(W * stock_weight_multiplier, W + excess_stock_weight)
    # Lower: max(0, W_i - excess)
    stock_lb = np.maximum(0.0, W - excess_stock_weight)
    constraints.append(w <= stock_ub)
    constraints.append(w >= stock_lb)

    # --- Country weight bounds ---
    for country, uw in underlying_country_weights.items():
        mask = country_labels == country
        if not mask.any():
            continue
        lb = max(uw - COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_LB)
        ub = min(uw + COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_UB)
        country_w = cp.sum(w[mask])
        constraints.append(country_w >= lb)
        constraints.append(country_w <= ub)

    # --- Industry group weight bounds ---
    for ig, uw in underlying_ig_weights.items():
        mask = ig_labels == ig
        if not mask.any():
            continue
        # Lower bound: min(sum of eligible weights × multiplier, max(uw - 5%, uw × 0.75))
        eligible_sum_in_ig = float(np.sum(W[mask])) * stock_weight_multiplier
        floor = max(uw - COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_LB)
        ig_lb = min(eligible_sum_in_ig, floor)
        ig_ub = min(uw + COUNTRY_IG_BAND, uw * COUNTRY_IG_SCALE_UB)
        ig_w = cp.sum(w[mask])
        constraints.append(ig_w >= ig_lb)
        constraints.append(ig_w <= ig_ub)

    # --- Diversification constraint (quadratic) ---
    # W'_i = W_i + ineligible_weight / N_eligible
    W_prime = W + ineligible_weight / n_eligible
    rhs_per_stock = (2 * T + ineligible_weight / n_eligible) ** 2 / W_prime
    rhs_div = float(np.sum(rhs_per_stock))
    lhs_div = cp.sum(cp.square(w - W_prime) / W_prime)
    constraints.append(lhs_div <= rhs_div)

    # --- Absolute weight deviation constraint (active share ≤ T + ineligible) ---
    # (1/2) × sum(|W_i - w_i| for all underlying) ≤ T + ineligible_weight
    # For eligible stocks: |W_i - w_i| is variable
    # For ineligible stocks: W_i - 0 = W_i (constant, already in ineligible_weight sum)
    active_share_rhs = T + ineligible_weight
    # sum(|W_i - w_i|) / 2 ≤ T + ineligible
    # left side for eligible only; ineligible contributes ineligible_weight:
    constraints.append(
        (cp.sum(cp.abs(W - w)) + ineligible_weight) / 2 <= active_share_rhs
    )

    problem = cp.Problem(objective, constraints)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return OptimizationResult(weights=None, status="solver_error", relaxation_level=0)

    if w.value is None:
        return OptimizationResult(weights=None, status=problem.status, relaxation_level=0)

    return OptimizationResult(
        weights=np.array(w.value),
        status=problem.status,
        relaxation_level=0,
    )


def optimize(
    eligible_df: pd.DataFrame,
    all_underlying_weights: dict[str, float],
) -> OptimizationResult:
    """
    Run the optimization with the constraint-relaxation hierarchy.

    eligible_df: DataFrame of eligible stocks (from eligibility.apply_eligibility_filters).
    all_underlying_weights: weights for ALL stocks in the underlying index (including ineligible).

    Relaxation hierarchy (Ref: §Constraint Relaxation Hierarchy, p. 9):
      Level 0: no relaxation
      Level 1: stock weight bounds relaxed (excess up to 4%, multiplier up to 20×)
      Level 2: T relaxed toward 5%
      Level 3: industry group bounds relaxed (further)
      Level 4: country bounds relaxed (further)

    Hard constraints (never relaxed): minimum stock weight threshold.
    """
    eligible_tickers = eligible_df["ticker"].tolist()
    all_tickers = list(all_underlying_weights.keys())

    ineligible_tickers = [t for t in all_tickers if t not in eligible_tickers]
    ineligible_weight = sum(all_underlying_weights.get(t, 0.0) for t in ineligible_tickers)

    n_eligible = len(eligible_tickers)
    if n_eligible == 0:
        return OptimizationResult(
            weights=np.array([]),
            status="no_eligible_stocks",
            relaxation_level=0,
        )

    W_eligible = np.array(
        [all_underlying_weights.get(t, 0.0) for t in eligible_tickers]
    )

    # Impute missing carbon intensities with GICS Industry Group median
    ci_series = _impute_carbon_intensities(
        eligible_df.set_index("ticker")["carbon_intensity"],
        eligible_df.set_index("ticker")["gics_industry_group"],
    )
    CI = np.array([ci_series[t] for t in eligible_tickers])

    country_labels = np.array(
        eligible_df.set_index("ticker").loc[eligible_tickers, "country"].tolist()
    )
    ig_labels = np.array(
        eligible_df.set_index("ticker")
        .loc[eligible_tickers, "gics_industry_group"]
        .tolist()
    )

    # Underlying country/IG weights computed over ALL underlying stocks
    all_df = pd.DataFrame(
        {"ticker": all_tickers, "weight": list(all_underlying_weights.values())}
    )
    underlying_country_weights: dict[str, float] = {}
    underlying_ig_weights: dict[str, float] = {}

    for t in all_tickers:
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

    T_base = _compute_T(ineligible_weight)

    # Relaxation schedule
    relaxation_schedule = [
        # (excess_stock_weight, stock_weight_multiplier, T_override)
        (EXCESS_STOCK_WEIGHT_BASE, STOCK_WEIGHT_MULTIPLIER_BASE, T_base),           # Level 0
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_base),             # Level 1
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),              # Level 2
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),              # Level 3 (IG relaxed by caller — same params here)
        (EXCESS_STOCK_WEIGHT_MAX, STOCK_WEIGHT_MULTIPLIER_MAX, T_MIN),              # Level 4 (country relaxed by caller)
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
            result.relaxation_level = level
            return result

    # Final fallback: return equal-weight over eligible if all attempts fail
    return OptimizationResult(
        weights=W_eligible / W_eligible.sum(),
        status="fallback_equal_weight",
        relaxation_level=len(relaxation_schedule),
    )


def apply_minimum_weight_threshold(
    weights: np.ndarray,
    tickers: list[str],
    min_weight: float = MIN_STOCK_WEIGHT,
) -> dict[str, float]:
    """
    Post-processing step: exclude stocks with optimized weight < 1bps and
    redistribute their weight proportionally to stocks already above the threshold.

    Ref: §Optimization Constraints, Minimum Stock Weight Lower Threshold footnote 3 (p. 9)
    """
    w = weights.copy()
    tickers = list(tickers)

    for _ in range(len(tickers)):
        below = w < min_weight
        if not below.any():
            break
        redistributed = w[below].sum()
        w[below] = 0.0
        above = w > 0
        if above.any():
            w[above] += redistributed * (w[above] / w[above].sum())

    total = w.sum()
    if total > 0:
        w /= total  # re-normalize to account for floating point

    return {t: float(wi) for t, wi in zip(tickers, w) if wi > 0}
