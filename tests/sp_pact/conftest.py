"""Shared test fixtures for sp_pact tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.sp_pact.models import (
    BusinessActivityExposures,
    IndexUniverse,
    Stock,
    UNGCStatus,
    Variant,
)


SAMPLE_CSV = Path(__file__).resolve().parent.parent.parent / "indices" / "sp_pact" / "sample_data" / "universe.csv"


def make_stock(
    ticker: str = "TEST",
    company_name: str = "Test Co",
    country: str = "US",
    gics_sector: str = "Information Technology",
    gics_industry_group: str = "Software & Services",
    market_cap_usd: float = 1e9,
    float_ratio: float = 0.8,
    esg_score: float | None = 75.0,
    has_esg_coverage: bool = True,
    scope_1_2_carbon_intensity: float | None = 50.0,
    scope_3_carbon_intensity: float | None = 30.0,
    has_carbon_coverage: bool = True,
    has_sbti_target: bool = False,
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT,
    msa_flagged: bool = False,
    **ba_overrides,
) -> Stock:
    ba_kwargs = {
        "controversial_weapons_revenue_pct": 0.0,
        "controversial_weapons_ownership_pct": 0.0,
        "tobacco_production_revenue_pct": 0.0,
        "tobacco_related_revenue_pct": 0.0,
        "tobacco_retail_revenue_pct": 0.0,
        "small_arms_civilian_revenue_pct": 0.0,
        "small_arms_noncivilian_revenue_pct": 0.0,
        "small_arms_key_components_revenue_pct": 0.0,
        "small_arms_retail_revenue_pct": 0.0,
        "military_integral_weapons_revenue_pct": 0.0,
        "military_weapon_related_revenue_pct": 0.0,
        "thermal_coal_generation_revenue_pct": 0.0,
        "oil_sands_extraction_revenue_pct": 0.0,
        "shale_oil_gas_extraction_revenue_pct": 0.0,
        "gambling_operations_revenue_pct": 0.0,
        "alcohol_production_revenue_pct": 0.0,
        "alcohol_related_revenue_pct": 0.0,
        "alcohol_retail_revenue_pct": 0.0,
        "coal_revenue_pct": 0.0,
        "oil_revenue_pct": 0.0,
        "natural_gas_revenue_pct": 0.0,
        "power_generation_revenue_pct": 0.0,
    }
    ba_kwargs.update(ba_overrides)
    return Stock(
        ticker=ticker,
        company_name=company_name,
        country=country,
        gics_sector=gics_sector,
        gics_industry_group=gics_industry_group,
        market_cap_usd=market_cap_usd,
        float_ratio=float_ratio,
        esg_score=esg_score,
        has_esg_coverage=has_esg_coverage,
        scope_1_2_carbon_intensity=scope_1_2_carbon_intensity,
        scope_3_carbon_intensity=scope_3_carbon_intensity,
        has_carbon_coverage=has_carbon_coverage,
        has_sbti_target=has_sbti_target,
        ungc_status=ungc_status,
        msa_flagged=msa_flagged,
        business_activities=BusinessActivityExposures(**ba_kwargs),
    )


def make_universe(*stocks: Stock) -> IndexUniverse:
    return IndexUniverse(stocks=list(stocks))


@pytest.fixture
def sample_universe_csv() -> Path:
    return SAMPLE_CSV
