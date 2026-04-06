"""
Shared fixtures for all index tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.sp_carbon_aware.models import (
    BusinessActivityExposures,
    IndexUniverse,
    Stock,
    UNGCStatus,
)


SAMPLE_CSV = (
    Path(__file__).parent.parent
    / "indices" / "sp_carbon_aware" / "sample_data" / "universe.csv"
)


def make_stock(
    ticker: str = "TEST",
    company_name: str = "Test Co",
    country: str = "US",
    gics_sector: str = "Information Technology",
    gics_industry_group: str = "Software & Services",
    market_cap_usd: float = 100_000_000,
    float_ratio: float = 0.9,
    esg_score: float | None = 65.0,
    has_esg_coverage: bool = True,
    carbon_intensity: float | None = 50.0,
    ungc_status: UNGCStatus = UNGCStatus.COMPLIANT,
    business_activities: BusinessActivityExposures | None = None,
    msa_flagged: bool = False,
) -> Stock:
    """Factory for creating test Stock instances with sensible defaults."""
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
        carbon_intensity=carbon_intensity,
        ungc_status=ungc_status,
        business_activities=business_activities or BusinessActivityExposures(),
        msa_flagged=msa_flagged,
    )


def make_universe(stocks: list[Stock]) -> IndexUniverse:
    """Build an IndexUniverse from a list of stocks (weights computed automatically)."""
    return IndexUniverse(stocks=stocks)


@pytest.fixture
def clean_stock() -> Stock:
    """A single stock that passes every eligibility filter."""
    return make_stock()


@pytest.fixture
def two_stock_universe() -> IndexUniverse:
    """Minimal two-stock universe for optimization smoke tests."""
    s1 = make_stock("A", "Alpha Inc", market_cap_usd=600_000_000, carbon_intensity=50.0)
    s2 = make_stock("B", "Beta Inc", market_cap_usd=400_000_000, carbon_intensity=200.0)
    return make_universe([s1, s2])


@pytest.fixture
def sample_universe_csv() -> Path:
    return SAMPLE_CSV
