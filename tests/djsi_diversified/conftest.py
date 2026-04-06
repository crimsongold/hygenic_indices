"""Shared test fixtures for djsi_diversified tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.djsi_diversified.models import (
    BusinessActivityExposures,
    IndexUniverse,
    Stock,
    UNGCStatus,
)


SAMPLE_CSV = Path(__file__).resolve().parent.parent.parent / "indices" / "djsi_diversified" / "sample_data" / "universe.csv"


def make_stock(
    ticker: str = "TEST",
    company_name: str = "Test Co",
    country: str = "US",
    region: str = "North America",
    gics_sector: str = "Information Technology",
    gics_industry_group: str = "Software",
    market_cap_usd: float = 1e9,
    float_ratio: float = 0.8,
    esg_score: float | None = 75.0,
    has_esg_coverage: bool = True,
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT,
    msa_flagged: bool = False,
    **ba_overrides,
) -> Stock:
    ba_kwargs = {
        "controversial_weapons_revenue_pct": 0.0,
        "controversial_weapons_ownership_pct": 0.0,
        "tobacco_production_revenue_pct": 0.0,
        "adult_entertainment_production_revenue_pct": 0.0,
        "adult_entertainment_retail_revenue_pct": 0.0,
        "alcohol_production_revenue_pct": 0.0,
        "gambling_operations_revenue_pct": 0.0,
        "gambling_equipment_revenue_pct": 0.0,
        "military_integral_weapons_revenue_pct": 0.0,
        "military_weapon_related_revenue_pct": 0.0,
        "small_arms_civilian_production_revenue_pct": 0.0,
        "small_arms_key_components_revenue_pct": 0.0,
        "small_arms_noncivilian_production_revenue_pct": 0.0,
        "small_arms_retail_revenue_pct": 0.0,
    }
    ba_kwargs.update(ba_overrides)
    return Stock(
        ticker=ticker,
        company_name=company_name,
        country=country,
        region=region,
        gics_sector=gics_sector,
        gics_industry_group=gics_industry_group,
        market_cap_usd=market_cap_usd,
        float_ratio=float_ratio,
        esg_score=esg_score,
        has_esg_coverage=has_esg_coverage,
        ungc_status=ungc_status,
        msa_flagged=msa_flagged,
        business_activities=BusinessActivityExposures(**ba_kwargs),
    )


def make_universe(*stocks: Stock) -> IndexUniverse:
    return IndexUniverse(stocks=list(stocks))


@pytest.fixture
def sample_universe_csv() -> Path:
    return SAMPLE_CSV
