"""
Carbon efficiency tilt weighting for the S&P Global Carbon Efficient Index Series.

This module implements the core weight-adjustment logic of the Carbon Efficient
index. Unlike the S&P Carbon Aware Index (which uses hard exclusions + convex
optimization), this index keeps ALL benchmark constituents and applies a
multiplicative carbon efficiency factor (CEF) to each stock's underlying weight.

The actual S&P methodology uses a decile-based system:
  - Within each GICS Industry Group, stocks are ranked by carbon-to-revenue
    footprint (tCO2e / $M revenue from Trucost) and assigned to deciles 1-10.
  - Each decile receives a fixed percentage weight adjustment (e.g. decile 1
    gets +20%, decile 10 gets -20%), scaled by the group's Impact Factor.
  - Impact Factors classify groups as Low (x0.5), Mid (x1.0), or High (x3.0)
    based on the range of carbon intensities within the group.

Our implementation uses a simplified exponential tilt approximation:

    z_i = (CI_i - mean(CI_group)) / std(CI_group)   [within-group z-score]
    CEF_i = exp(-lambda * z_i)

This produces a smooth, continuous version of the decile-based step function.
The exponential form is convex in the carbon intensity, so low-CI stocks are
always rewarded and high-CI stocks are always penalised — the same directional
behaviour as the decile system, without requiring discrete bucket boundaries.

After computing CEFs, each stock's final weight is:

    w_i = W_i * CEF_i / sum_j(W_j * CEF_j)    [within each GICS Industry Group]

This normalization ensures the index is industry-group-neutral: the total weight
allocated to each GICS Industry Group matches the benchmark allocation. Only
the WITHIN-group distribution of weight changes.

Companies without carbon data (CI = None) are assigned CEF = 1.0, matching the
methodology's treatment of "Not-disclosed" companies in the 4th-7th decile
range with a 0% weight adjustment — they pass through at their underlying weight.

Ref: "S&P Global Carbon Efficient Index Series Methodology",
     §Index Construction, §Carbon-to-Revenue Footprint,
     §Decile-Based Carbon Weight Adjustment, §Industry Group Impact Factors
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# Ref: §Index Construction — Tilt Parameters
# ---------------------------------------------------------------------------

# Default tilt strength (lambda). Increasing lambda amplifies the carbon tilt:
# - lambda = 0 → no tilt at all (CEF = 1 for every stock, index = benchmark)
# - lambda = 0.5 → moderate tilt (standard variant)
# - lambda > 1 → aggressive tilt (large overweight of clean companies)
#
# The published methodology uses values around 0.5 for the standard variant.
# This parameter controls the trade-off between carbon intensity reduction and
# tracking error relative to the benchmark.
DEFAULT_LAMBDA: float = 0.5


# ---------------------------------------------------------------------------
# Carbon Efficiency Factor computation
# Ref: §Index Construction, §Carbon-to-Revenue Footprint
# ---------------------------------------------------------------------------


def compute_carbon_efficiency_factors(
    df: pd.DataFrame,
    tilt_lambda: float = DEFAULT_LAMBDA,
) -> pd.Series:
    """
    Compute a Carbon Efficiency Factor (CEF) for each stock in the universe.

    The CEF determines how much a stock's underlying weight is scaled up or down:
      CEF > 1.0 → stock receives MORE weight (low emitter relative to group)
      CEF < 1.0 → stock receives LESS weight (high emitter relative to group)
      CEF = 1.0 → neutral (no carbon data, or single-stock group, or zero variance)

    Algorithm (per GICS Industry Group):
      1. Collect carbon intensities for all stocks with Trucost coverage.
      2. Compute the within-group mean and standard deviation.
      3. For each stock with CI data, compute z_i = (CI_i - mean) / std.
      4. Set CEF_i = exp(-lambda * z_i).
      5. Stocks without CI data get CEF = 1.0 (neutral — no penalty or reward).

    Why z-scores instead of raw CI values?
        Standardising by group ensures that a stock in a high-emission industry
        (e.g. Utilities, CI ~ 500) is compared against its industry peers, not
        against a low-emission industry (e.g. Software, CI ~ 5). Without z-scores
        the exponential would wildly over-penalise high-emission industries and
        under-penalise low-emission ones.

    Why exponential?
        exp(-lambda * z) is a smooth, strictly monotone function: lower z (lower
        CI relative to peers) always gives a higher CEF. This avoids the
        discontinuities of a step-function decile system and is differentiable,
        which would matter if this were embedded in a larger optimization.

    Edge cases:
      - Group with 0 or 1 stocks with CI data → all members get CEF = 1.0.
        Rationale: cannot compute a meaningful z-score without at least 2 data
        points (std would be 0 or undefined).
      - Group where all CI values are identical → std = 0 → all CEFs = 1.0.
        Rationale: z-scores would require division by zero; all stocks are
        equally carbon-efficient so no tilt is warranted.

    Parameters
    ----------
    df:
        Universe DataFrame from IndexUniverse.to_dataframe(). Must have columns
        `gics_industry_group` and `carbon_intensity`.
    tilt_lambda:
        Exponential tilt strength. Higher values give a more aggressive tilt.

    Returns
    -------
    pd.Series indexed like df, one CEF value per row.

    Ref: §Index Construction, §Decile-Based Carbon Weight Adjustment
    """
    # Default every stock to neutral tilt; only stocks with sufficient group
    # data will be overwritten below
    cef = pd.Series(1.0, index=df.index)

    for _group, group_df in df.groupby("gics_industry_group"):
        # Collect only stocks that have Trucost carbon coverage
        ci_vals = group_df["carbon_intensity"].dropna()

        if ci_vals.empty or len(ci_vals) < 2:
            # Single stock or no data in this group — neutral tilt for all members.
            # Cannot compute a meaningful z-score without at least 2 observations.
            continue

        # Within-group statistics for z-score calculation
        group_mean = ci_vals.mean()
        group_std = ci_vals.std(ddof=1)  # sample std (Bessel's correction)

        if group_std == 0:
            # All companies in the group have identical CI → z-scores would all
            # be 0 → CEF would all be exp(0) = 1.0. Skip to avoid division by zero.
            continue

        # Compute z-score and CEF for each stock in this group
        for idx in group_df.index:
            ci = group_df.at[idx, "carbon_intensity"]
            if ci is None or (isinstance(ci, float) and math.isnan(ci)):
                # No Trucost coverage → neutral tilt (CEF = 1.0).
                # Matches the methodology's "Not-disclosed" treatment: companies
                # without carbon data get a 0% weight adjustment (4th-7th decile).
                cef.at[idx] = 1.0
            else:
                # z > 0 means above-average CI (dirtier) → CEF < 1 (penalised)
                # z < 0 means below-average CI (cleaner) → CEF > 1 (rewarded)
                z = (ci - group_mean) / group_std
                cef.at[idx] = math.exp(-tilt_lambda * z)

    return cef


# ---------------------------------------------------------------------------
# Weight tilt application
# Ref: §Index Construction — Weight Calculation
# ---------------------------------------------------------------------------


def apply_tilt(
    df: pd.DataFrame,
    underlying_weights: dict[str, float],
    tilt_lambda: float = DEFAULT_LAMBDA,
) -> dict[str, float]:
    """
    Apply the carbon efficiency tilt and return normalised final weights.

    The tilt formula for each stock is:

        raw_i = W_i * CEF_i
        w_i   = raw_i / sum(raw_j)    for all j in universe

    This produces industry-group-neutral weights: because the CEF z-scores are
    computed within each group, the relative tilts within a group average out
    to roughly preserve the group's total weight. The final normalisation step
    ensures the portfolio sums exactly to 1.0.

    Why multiply rather than add?
        A multiplicative tilt (W * CEF) preserves the relative ordering of
        stocks by size within each group. A large-cap stock with a neutral CEF
        of 1.0 still receives a large weight. An additive adjustment would
        distort the size exposure.

    Parameters
    ----------
    df:
        Universe DataFrame from IndexUniverse.to_dataframe().
    underlying_weights:
        Dict mapping ticker -> underlying float-cap weight (from IndexUniverse).
    tilt_lambda:
        Exponential tilt strength (default 0.5).

    Returns
    -------
    Dict mapping ticker -> final tilted weight (sums to 1.0).

    Ref: §Index Construction — Weight Calculation
    """
    cef = compute_carbon_efficiency_factors(df, tilt_lambda=tilt_lambda)

    # Multiply each underlying weight by the stock's Carbon Efficiency Factor.
    # This is the core tilt operation: low-CI stocks get CEF > 1 (weight goes up),
    # high-CI stocks get CEF < 1 (weight goes down).
    raw_tilted = {}
    for i, row in df.iterrows():
        ticker = row["ticker"]
        raw_tilted[ticker] = underlying_weights[ticker] * cef.at[i]

    # Normalise so weights sum to 1.0.
    # Without normalisation, the sum of raw_tilted weights would deviate from 1.0
    # because the CEFs are not guaranteed to preserve the total weight.
    total = sum(raw_tilted.values())
    if total == 0:
        # Degenerate case: all weights are zero (empty universe or all zero market cap).
        # Return zeros to avoid division by zero.
        return {t: 0.0 for t in raw_tilted}
    return {t: w / total for t, w in raw_tilted.items()}


# ---------------------------------------------------------------------------
# WACI diagnostic
# Ref: §Weighted Average Carbon Intensity
# ---------------------------------------------------------------------------


def weighted_avg_carbon_intensity(
    df: pd.DataFrame,
    weights: dict[str, float],
) -> float:
    """
    Compute Weighted Average Carbon Intensity (WACI) for a given weight vector.

    WACI = sum(w_i * CI_i) / sum(w_i)    over stocks i with known CI

    Stocks without carbon data (CI = None or NaN) are excluded from BOTH the
    numerator and denominator. This avoids biasing the WACI downward by
    treating non-covered companies as zero-emission.

    The denominator is the sum of weights of stocks WITH CI data, not the full
    portfolio weight. This means the WACI reflects the carbon intensity per
    unit of "covered" weight. If coverage is low, the WACI may not be
    representative of the full portfolio.

    Returns NaN if no stocks have carbon data (total covered weight is zero).

    Ref: §Weighted Average Carbon Intensity
    """
    total_w = 0.0
    waci = 0.0
    for _, row in df.iterrows():
        ci = row["carbon_intensity"]
        w = weights.get(row["ticker"], 0.0)
        if ci is None or (isinstance(ci, float) and np.isnan(ci)):
            # No Trucost coverage for this stock — skip it
            continue
        waci += w * ci
        total_w += w
    if total_w == 0:
        return float("nan")
    return waci / total_w
