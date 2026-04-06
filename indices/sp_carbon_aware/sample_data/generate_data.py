"""
Generates a realistic sample universe CSV for the S&P Carbon Aware Index Series.

Run this script to regenerate universe.csv:
    python indices/sp_carbon_aware/sample_data/generate_data.py

The output has 60 stocks across 10 GICS Industry Groups and 8 countries,
with a mix of clean and excluded companies to exercise all eligibility filters.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

OUTPUT_PATH = Path(__file__).parent / "universe.csv"

COUNTRIES = ["US", "GB", "DE", "JP", "FR", "KR", "AU", "CA"]

GICS_INDUSTRY_GROUPS = [
    ("Energy", "Energy"),
    ("Materials", "Materials"),
    ("Capital Goods", "Industrials"),
    ("Transportation", "Industrials"),
    ("Consumer Durables & Apparel", "Consumer Discretionary"),
    ("Food, Beverage & Tobacco", "Consumer Staples"),
    ("Health Care Equipment & Services", "Health Care"),
    ("Software & Services", "Information Technology"),
    ("Banks", "Financials"),
    ("Utilities", "Utilities"),
]

UNGC_STATUSES = ["Compliant", "Compliant", "Compliant", "Watchlist", "Non-Compliant"]


def _row(
    ticker: str,
    company: str,
    country: str,
    ig: str,
    sector: str,
    market_cap: float,
    float_ratio: float,
    esg_score: float | None,
    has_esg_coverage: bool,
    carbon_intensity: float | None,
    ungc_status: str,
    msa_flagged: bool,
    has_sustainalytics_coverage: bool,
    # Business activities (all default to 0)
    cw_tailor_pct: float = 0.0,
    cw_tailor_own: float = 0.0,
    cw_non_tailor_pct: float = 0.0,
    cw_non_tailor_own: float = 0.0,
    tobacco_prod: float = 0.0,
    tobacco_rel: float = 0.0,
    tobacco_ret: float = 0.0,
    coal_extr: float = 0.0,
    coal_power: float = 0.0,
    oil_sands: float = 0.0,
    shale: float = 0.0,
    arctic: float = 0.0,
    og_prod: float = 0.0,
    og_gen: float = 0.0,
    og_support: float = 0.0,
    gamble_ops: float = 0.0,
    gamble_eq: float = 0.0,
    gamble_supp: float = 0.0,
    adult_prod: float = 0.0,
    adult_dist: float = 0.0,
    alc_prod: float = 0.0,
    alc_ret: float = 0.0,
    alc_rel: float = 0.0,
) -> dict:
    return {
        "ticker": ticker,
        "company_name": company,
        "country": country,
        "gics_sector": sector,
        "gics_industry_group": ig,
        "market_cap_usd": market_cap,
        "float_ratio": float_ratio,
        "esg_score": "" if esg_score is None else esg_score,
        "has_esg_coverage": int(has_esg_coverage),
        "carbon_intensity": "" if carbon_intensity is None else carbon_intensity,
        "ungc_status": ungc_status,
        "msa_flagged": int(msa_flagged),
        "has_sustainalytics_coverage": int(has_sustainalytics_coverage),
        "controversial_weapons_tailor_made_essential_pct": cw_tailor_pct,
        "controversial_weapons_tailor_made_essential_ownership_pct": cw_tailor_own,
        "controversial_weapons_non_tailor_made_pct": cw_non_tailor_pct,
        "controversial_weapons_non_tailor_made_ownership_pct": cw_non_tailor_own,
        "tobacco_production_revenue_pct": tobacco_prod,
        "tobacco_related_revenue_pct": tobacco_rel,
        "tobacco_retail_revenue_pct": tobacco_ret,
        "thermal_coal_extraction_revenue_pct": coal_extr,
        "thermal_coal_power_generation_revenue_pct": coal_power,
        "oil_sands_extraction_revenue_pct": oil_sands,
        "shale_energy_extraction_revenue_pct": shale,
        "arctic_oil_gas_extraction_revenue_pct": arctic,
        "oil_gas_production_revenue_pct": og_prod,
        "oil_gas_generation_revenue_pct": og_gen,
        "oil_gas_supporting_revenue_pct": og_support,
        "gambling_operations_revenue_pct": gamble_ops,
        "gambling_equipment_revenue_pct": gamble_eq,
        "gambling_supporting_revenue_pct": gamble_supp,
        "adult_entertainment_production_revenue_pct": adult_prod,
        "adult_entertainment_distribution_revenue_pct": adult_dist,
        "alcoholic_beverages_production_revenue_pct": alc_prod,
        "alcoholic_beverages_retail_revenue_pct": alc_ret,
        "alcoholic_beverages_related_revenue_pct": alc_rel,
    }


def build_universe() -> list[dict]:
    rows: list[dict] = []

    # --- Clean, eligible companies (pass all filters) ---
    clean_companies = [
        # Software & Services — low carbon intensity, high ESG
        ("MSFT", "MegaSoft Corp", "US", "Software & Services", "Information Technology", 2_800_000, 0.98, 82.0, 52.0),
        ("GOOG", "Alphabet Inc", "US", "Software & Services", "Information Technology", 1_900_000, 0.97, 79.0, 48.0),
        ("SAP", "SAP SE", "DE", "Software & Services", "Information Technology", 850_000, 0.85, 76.0, 45.0),
        ("SONY", "Sony Group", "JP", "Software & Services", "Information Technology", 720_000, 0.82, 68.0, 55.0),
        ("ASML", "ASML Holding", "DE", "Software & Services", "Information Technology", 680_000, 0.88, 71.0, 50.0),

        # Health Care Equipment & Services — low carbon
        ("JNJ", "Johnson & Johnson", "US", "Health Care Equipment & Services", "Health Care", 950_000, 0.91, 74.0, 38.0),
        ("ROG", "Roche Holding", "CH", "Health Care Equipment & Services", "Health Care", 780_000, 0.86, 72.0, 35.0),
        ("SAN", "Sanofi SA", "FR", "Health Care Equipment & Services", "Health Care", 650_000, 0.84, 69.0, 40.0),
        ("MED", "MedTech Global", "GB", "Health Care Equipment & Services", "Health Care", 420_000, 0.80, 66.0, 33.0),
        ("AZN", "AstraZeneca", "GB", "Health Care Equipment & Services", "Health Care", 580_000, 0.89, 73.0, 37.0),

        # Banks — moderate carbon
        ("JPM", "JPMorgan Chase", "US", "Banks", "Financials", 700_000, 0.92, 65.0, 60.0),
        ("HSBC", "HSBC Holdings", "GB", "Banks", "Financials", 520_000, 0.88, 63.0, 58.0),
        ("BNP", "BNP Paribas", "FR", "Banks", "Financials", 490_000, 0.85, 61.0, 62.0),
        ("DBK", "Deutsche Bank", "DE", "Banks", "Financials", 380_000, 0.82, 59.0, 64.0),
        ("MUFG", "Mitsubishi UFJ", "JP", "Banks", "Financials", 610_000, 0.79, 57.0, 66.0),

        # Capital Goods — moderate carbon
        ("SIE", "Siemens AG", "DE", "Capital Goods", "Industrials", 620_000, 0.87, 70.0, 120.0),
        ("ABB", "ABB Ltd", "DE", "Capital Goods", "Industrials", 480_000, 0.84, 68.0, 115.0),
        ("HWM", "Honeywell Mfg", "US", "Capital Goods", "Industrials", 450_000, 0.90, 66.0, 130.0),
        ("EMR", "Emerson Electric", "US", "Capital Goods", "Industrials", 390_000, 0.88, 64.0, 125.0),
        ("SFK", "SKF Group", "DE", "Capital Goods", "Industrials", 280_000, 0.82, 62.0, 135.0),

        # Consumer Durables & Apparel — moderate carbon
        ("TYT", "Toyota Motor", "JP", "Consumer Durables & Apparel", "Consumer Discretionary", 880_000, 0.76, 65.0, 180.0),
        ("BMW", "BMW AG", "DE", "Consumer Durables & Apparel", "Consumer Discretionary", 650_000, 0.74, 67.0, 175.0),
        ("LVMH", "LVMH Group", "FR", "Consumer Durables & Apparel", "Consumer Discretionary", 720_000, 0.78, 69.0, 80.0),
        ("NKE", "Nike Inc", "US", "Consumer Durables & Apparel", "Consumer Discretionary", 550_000, 0.93, 64.0, 70.0),
        ("HEN", "Henkel AG", "DE", "Consumer Durables & Apparel", "Consumer Discretionary", 310_000, 0.81, 63.0, 90.0),

        # Transportation — higher carbon
        ("UPS", "UPS Inc", "US", "Transportation", "Industrials", 560_000, 0.91, 60.0, 250.0),
        ("DHL", "Deutsche Post DHL", "DE", "Transportation", "Industrials", 420_000, 0.86, 61.0, 260.0),
        ("FDX", "FedEx Corp", "US", "Transportation", "Industrials", 490_000, 0.92, 59.0, 270.0),
        ("JR", "Japan Rail Group", "JP", "Transportation", "Industrials", 380_000, 0.77, 63.0, 150.0),
        ("AIR", "Air Liquide", "FR", "Transportation", "Industrials", 350_000, 0.83, 62.0, 200.0),

        # Materials — higher carbon
        ("BHP", "BHP Group", "AU", "Materials", "Materials", 920_000, 0.85, 58.0, 450.0),
        ("RIO", "Rio Tinto", "AU", "Materials", "Materials", 780_000, 0.83, 56.0, 430.0),
        ("LIN", "Linde PLC", "US", "Materials", "Materials", 680_000, 0.91, 60.0, 380.0),
        ("AIR2", "Air Products", "US", "Materials", "Materials", 520_000, 0.88, 57.0, 400.0),
        ("NUE", "Nucor Corp", "US", "Materials", "Materials", 460_000, 0.89, 55.0, 500.0),

        # Utilities — missing some carbon intensity data (will be imputed)
        ("NEE", "NextEra Energy", "US", "Utilities", "Utilities", 680_000, 0.90, 68.0, None),
        ("IBE", "Iberdrola SA", "FR", "Utilities", "Utilities", 590_000, 0.87, 70.0, 120.0),
        ("EDP", "EDP Energias", "FR", "Utilities", "Utilities", 420_000, 0.84, 66.0, 130.0),
        ("RWE", "RWE AG", "DE", "Utilities", "Utilities", 380_000, 0.82, 64.0, None),
        ("CNA", "Centrica PLC", "GB", "Utilities", "Utilities", 290_000, 0.80, 62.0, 140.0),
    ]

    for t, co, ctry, ig, sec, mcap, fr, esg, ci in clean_companies:
        rows.append(_row(
            ticker=t, company=co, country=ctry, ig=ig, sector=sec,
            market_cap=mcap * 1_000_000, float_ratio=fr,
            esg_score=esg, has_esg_coverage=True, carbon_intensity=ci,
            ungc_status="Compliant", msa_flagged=False,
            has_sustainalytics_coverage=True,
        ))

    # --- Companies excluded for various reasons ---

    # Missing ESG coverage
    rows.append(_row(
        "NOESC", "NoESG Corp", "US", "Materials", "Materials",
        200_000_000, 0.80, esg_score=None, has_esg_coverage=False,
        carbon_intensity=300.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))

    # Bottom 25% ESG in Energy industry group — these get excluded by ESG screen
    rows.append(_row(
        "LOESG1", "LowESG Energy A", "US", "Energy", "Energy",
        150_000_000, 0.75, esg_score=15.0, has_esg_coverage=True,
        carbon_intensity=800.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))
    rows.append(_row(
        "LOESG2", "LowESG Energy B", "GB", "Energy", "Energy",
        130_000_000, 0.72, esg_score=12.0, has_esg_coverage=True,
        carbon_intensity=750.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))
    # Mid-range Energy ESG (eligible)
    rows.append(_row(
        "ENEOK", "GreenEnergy Ltd", "DE", "Energy", "Energy",
        400_000_000, 0.82, esg_score=55.0, has_esg_coverage=True,
        carbon_intensity=350.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))
    rows.append(_row(
        "ENEOK2", "CleanPower AG", "FR", "Energy", "Energy",
        350_000_000, 0.80, esg_score=50.0, has_esg_coverage=True,
        carbon_intensity=300.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))

    # Controversial weapons (tailor-made >0%)
    rows.append(_row(
        "WEAP1", "Arms Tech Inc", "US", "Capital Goods", "Industrials",
        180_000_000, 0.78, esg_score=40.0, has_esg_coverage=True,
        carbon_intensity=200.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        cw_tailor_pct=5.0,
    ))

    # Tobacco producer (>0%)
    rows.append(_row(
        "SMOK1", "TobaccoCo Ltd", "GB", "Food, Beverage & Tobacco", "Consumer Staples",
        250_000_000, 0.83, esg_score=45.0, has_esg_coverage=True,
        carbon_intensity=150.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        tobacco_prod=45.0,
    ))

    # Thermal coal power generator (>0%)
    rows.append(_row(
        "COAL1", "CoalPower Corp", "AU", "Utilities", "Utilities",
        180_000_000, 0.76, esg_score=38.0, has_esg_coverage=True,
        carbon_intensity=1200.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        coal_power=30.0,
    ))

    # Oil & gas producer (>0%)
    rows.append(_row(
        "OIL1", "Petro Giant", "US", "Energy", "Energy",
        900_000_000, 0.88, esg_score=52.0, has_esg_coverage=True,
        carbon_intensity=950.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        og_prod=70.0,
    ))

    # Gambling operations (>=5%)
    rows.append(_row(
        "GAMB1", "CasinoWorld PLC", "GB", "Consumer Durables & Apparel", "Consumer Discretionary",
        120_000_000, 0.77, esg_score=42.0, has_esg_coverage=True,
        carbon_intensity=90.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        gamble_ops=8.0,
    ))

    # Alcoholic beverages producer (>=5%)
    rows.append(_row(
        "BEER1", "BrewMaster Inc", "DE", "Food, Beverage & Tobacco", "Consumer Staples",
        300_000_000, 0.84, esg_score=50.0, has_esg_coverage=True,
        carbon_intensity=110.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
        alc_prod=60.0,
    ))

    # UNGC Non-Compliant
    rows.append(_row(
        "UNGC1", "NormViolator SA", "FR", "Banks", "Financials",
        220_000_000, 0.81, esg_score=48.0, has_esg_coverage=True,
        carbon_intensity=70.0, ungc_status="Non-Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))

    # No Sustainalytics coverage
    rows.append(_row(
        "NOSC1", "NoSust Corp", "KR", "Materials", "Materials",
        160_000_000, 0.79, esg_score=55.0, has_esg_coverage=True,
        carbon_intensity=200.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=False,
    ))

    # MSA flagged
    rows.append(_row(
        "MSA1", "Flagged Corp", "US", "Software & Services", "Information Technology",
        300_000_000, 0.91, esg_score=60.0, has_esg_coverage=True,
        carbon_intensity=45.0, ungc_status="Compliant",
        msa_flagged=True, has_sustainalytics_coverage=True,
    ))

    # Food, Beverage & Tobacco (clean — food only)
    rows.append(_row(
        "FOOD1", "GlobalFood SA", "FR", "Food, Beverage & Tobacco", "Consumer Staples",
        480_000_000, 0.86, esg_score=62.0, has_esg_coverage=True,
        carbon_intensity=140.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))
    rows.append(_row(
        "FOOD2", "NutriCorp", "US", "Food, Beverage & Tobacco", "Consumer Staples",
        420_000_000, 0.88, esg_score=60.0, has_esg_coverage=True,
        carbon_intensity=130.0, ungc_status="Compliant",
        msa_flagged=False, has_sustainalytics_coverage=True,
    ))

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    rows = build_universe()
    write_csv(rows, OUTPUT_PATH)
