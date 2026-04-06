"""Shared pytest fixtures for sp_esg tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.sp_esg.models import (
    BusinessActivityExposures,
    IndexUniverse,
    Stock,
    UNGCStatus,
)


def make_stock(**overrides) -> Stock:
    """
    Build a Stock that passes every eligibility filter by default.
    Override any field with kwargs.
    """
    defaults = dict(
        ticker="TEST",
        company_name="Test Corp",
        country="US",
        gics_sector="Information Technology",
        gics_industry_group="Software & Services",
        market_cap_usd=1_000_000_000,
        float_ratio=1.0,
        esg_score=75.0,
        has_esg_coverage=True,
        ungc_status=UNGCStatus.COMPLIANT,
        msa_flagged=False,
        business_activities=BusinessActivityExposures(),
    )
    defaults.update(overrides)
    return Stock(**defaults)


def make_universe(*stocks: Stock) -> IndexUniverse:
    return IndexUniverse(stocks=list(stocks))


@pytest.fixture
def clean_stock() -> Stock:
    return make_stock()


@pytest.fixture
def sample_universe_csv() -> Path:
    return Path(__file__).parent.parent.parent / "indices" / "sp_esg" / "sample_data" / "universe.csv"
