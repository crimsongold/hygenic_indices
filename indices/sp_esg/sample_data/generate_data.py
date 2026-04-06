"""
Generates a synthetic sample universe CSV for the S&P ESG Index Series.
Run directly: python generate_data.py
"""

import csv
import random
from pathlib import Path

random.seed(42)

COUNTRIES = ["US", "GB", "DE", "JP", "FR", "CH", "CA", "AU"]
SECTORS = {
    "Information Technology": ["Software & Services", "Technology Hardware & Equipment"],
    "Health Care": ["Pharmaceuticals Biotechnology & Life Sciences", "Health Care Equipment & Services"],
    "Financials": ["Banks", "Diversified Financials"],
    "Consumer Discretionary": ["Consumer Durables & Apparel", "Retailing"],
    "Industrials": ["Capital Goods", "Commercial & Professional Services"],
    "Energy": ["Energy", "Energy"],
    "Materials": ["Materials", "Materials"],
}

STOCKS = [
    ("MSFT", "Microsoft Corp", "US", "Information Technology", "Software & Services"),
    ("AAPL", "Apple Inc", "US", "Information Technology", "Technology Hardware & Equipment"),
    ("GOOGL", "Alphabet Inc", "US", "Information Technology", "Software & Services"),
    ("AMZN", "Amazon.com Inc", "US", "Consumer Discretionary", "Retailing"),
    ("META", "Meta Platforms", "US", "Information Technology", "Software & Services"),
    ("NVDA", "NVIDIA Corp", "US", "Information Technology", "Technology Hardware & Equipment"),
    ("JNJ", "Johnson & Johnson", "US", "Health Care", "Pharmaceuticals Biotechnology & Life Sciences"),
    ("UNH", "UnitedHealth Group", "US", "Health Care", "Health Care Equipment & Services"),
    ("JPM", "JPMorgan Chase", "US", "Financials", "Banks"),
    ("BAC", "Bank of America", "US", "Financials", "Banks"),
    ("WFC", "Wells Fargo", "US", "Financials", "Banks"),
    ("GS", "Goldman Sachs", "US", "Financials", "Diversified Financials"),
    ("TSLA", "Tesla Inc", "US", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("HD", "Home Depot", "US", "Consumer Discretionary", "Retailing"),
    ("BA", "Boeing Co", "US", "Industrials", "Capital Goods"),
    ("CAT", "Caterpillar Inc", "US", "Industrials", "Capital Goods"),
    ("GE", "General Electric", "US", "Industrials", "Capital Goods"),
    ("MMM", "3M Company", "US", "Industrials", "Capital Goods"),
    ("SAP", "SAP SE", "DE", "Information Technology", "Software & Services"),
    ("ASML", "ASML Holding", "NL", "Information Technology", "Technology Hardware & Equipment"),
    ("NESN", "Nestle SA", "CH", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("ROG", "Roche Holding", "CH", "Health Care", "Pharmaceuticals Biotechnology & Life Sciences"),
    ("NOVN", "Novartis AG", "CH", "Health Care", "Pharmaceuticals Biotechnology & Life Sciences"),
    ("SIE", "Siemens AG", "DE", "Industrials", "Capital Goods"),
    ("MC", "LVMH", "FR", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("TM", "Toyota Motor", "JP", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("6758", "Sony Group", "JP", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("RIO", "Rio Tinto", "GB", "Materials", "Materials"),
    ("BHP", "BHP Group", "AU", "Materials", "Materials"),
    ("XOM", "ExxonMobil", "US", "Energy", "Energy"),
    ("CVX", "Chevron Corp", "US", "Energy", "Energy"),
    ("SHEL", "Shell PLC", "GB", "Energy", "Energy"),
    ("BTI", "British American Tobacco", "GB", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("PM", "Philip Morris Intl", "US", "Consumer Discretionary", "Consumer Durables & Apparel"),
    ("RDS", "Rio Tinto Spin", "GB", "Materials", "Materials"),
    ("ABB", "ABB Ltd", "CH", "Industrials", "Capital Goods"),
    ("AZN", "AstraZeneca", "GB", "Health Care", "Pharmaceuticals Biotechnology & Life Sciences"),
    ("HSBA", "HSBC Holdings", "GB", "Financials", "Banks"),
    ("PRU", "Prudential PLC", "GB", "Financials", "Diversified Financials"),
    ("ADBE", "Adobe Inc", "US", "Information Technology", "Software & Services"),
]

UNGC_STATUSES = ["Compliant"] * 7 + ["Watchlist"] + ["Non-Compliant"] + ["No Coverage"]

HEADERS = [
    "ticker", "company_name", "country", "gics_sector", "gics_industry_group",
    "market_cap_usd", "float_ratio",
    "esg_score", "has_esg_coverage",
    "ungc_status", "msa_flagged",
    "controversial_weapons_revenue_pct", "controversial_weapons_ownership_pct",
    "tobacco_production_revenue_pct", "tobacco_retail_revenue_pct",
    "thermal_coal_extraction_revenue_pct", "thermal_coal_power_revenue_pct",
    "small_arms_manufacture_revenue_pct", "small_arms_retail_revenue_pct",
]

out = Path(__file__).parent / "universe.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()
    for i, (ticker, name, country, sector, ig) in enumerate(STOCKS):
        mktcap = random.uniform(5e9, 800e9)
        esg = round(random.uniform(20, 90), 1) if random.random() > 0.08 else None
        has_cov = 1 if esg is not None else 0
        ungc = random.choice(UNGC_STATUSES)
        msa = 1 if random.random() < 0.03 else 0

        # A few stocks get non-zero activity flags to test exclusions
        tobacco_prod = 80.0 if name in ("British American Tobacco", "Philip Morris Intl") else 0.0
        coal_ext = random.uniform(8, 40) if sector == "Energy" and random.random() < 0.15 else 0.0
        cw_rev = random.uniform(1, 5) if ticker in ("BA", "GE") else 0.0

        writer.writerow({
            "ticker": ticker,
            "company_name": name,
            "country": country,
            "gics_sector": sector,
            "gics_industry_group": ig,
            "market_cap_usd": round(mktcap, 2),
            "float_ratio": round(random.uniform(0.7, 1.0), 3),
            "esg_score": esg if esg is not None else "",
            "has_esg_coverage": has_cov,
            "ungc_status": ungc,
            "msa_flagged": msa,
            "controversial_weapons_revenue_pct": round(cw_rev, 2),
            "controversial_weapons_ownership_pct": 0.0,
            "tobacco_production_revenue_pct": tobacco_prod,
            "tobacco_retail_revenue_pct": 0.0,
            "thermal_coal_extraction_revenue_pct": round(coal_ext, 2),
            "thermal_coal_power_revenue_pct": 0.0,
            "small_arms_manufacture_revenue_pct": 0.0,
            "small_arms_retail_revenue_pct": 0.0,
        })

print(f"Written {len(STOCKS)} rows to {out}")
