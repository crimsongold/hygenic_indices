"""Shared fixtures for sp_carbon_efficient tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from indices.sp_carbon_efficient.models import IndexUniverse, Stock


def make_stock(**overrides) -> Stock:
    defaults = dict(
        ticker="TEST",
        company_name="Test Corp",
        country="US",
        gics_sector="Information Technology",
        gics_industry_group="Software & Services",
        market_cap_usd=1_000_000_000,
        float_ratio=1.0,
        carbon_intensity=50.0,
    )
    defaults.update(overrides)
    return Stock(**defaults)


def make_universe(*stocks: Stock) -> IndexUniverse:
    return IndexUniverse(stocks=list(stocks))


@pytest.fixture
def sample_universe_csv() -> Path:
    return (
        Path(__file__).parent.parent.parent
        / "indices" / "sp_carbon_efficient" / "sample_data" / "universe.csv"
    )
