"""
Generates a synthetic sample universe CSV for the S&P Global Carbon Efficient Index Series.
Run directly: python generate_data.py
"""

import csv
import random
from pathlib import Path

random.seed(7)

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
    ("TTE", "TotalEnergies", "FR", "Energy", "Energy"),
    ("BP", "BP PLC", "GB", "Energy", "Energy"),
    ("ABB", "ABB Ltd", "CH", "Industrials", "Capital Goods"),
    ("AZN", "AstraZeneca", "GB", "Health Care", "Pharmaceuticals Biotechnology & Life Sciences"),
    ("HSBA", "HSBC Holdings", "GB", "Financials", "Banks"),
    ("PRU", "Prudential PLC", "GB", "Financials", "Diversified Financials"),
    ("ADBE", "Adobe Inc", "US", "Information Technology", "Software & Services"),
    ("CRM", "Salesforce Inc", "US", "Information Technology", "Software & Services"),
    ("LIN", "Linde PLC", "GB", "Materials", "Materials"),
    ("PKX", "POSCO Holdings", "KR", "Materials", "Materials"),
    ("NEE", "NextEra Energy", "US", "Utilities", "Utilities"),
    ("DUK", "Duke Energy", "US", "Utilities", "Utilities"),
    ("SO", "Southern Company", "US", "Utilities", "Utilities"),
]

# Carbon intensity ranges by sector (tCO2e / $M revenue)
CI_RANGES = {
    "Energy": (200, 900),
    "Utilities": (150, 800),
    "Materials": (100, 500),
    "Industrials": (30, 200),
    "Consumer Discretionary": (10, 80),
    "Health Care": (5, 50),
    "Financials": (2, 20),
    "Information Technology": (3, 30),
}

HEADERS = [
    "ticker", "company_name", "country", "gics_sector", "gics_industry_group",
    "market_cap_usd", "float_ratio", "carbon_intensity",
]

out = Path(__file__).parent / "universe.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()
    for ticker, name, country, sector, ig in STOCKS:
        lo, hi = CI_RANGES.get(sector, (10, 100))
        ci = round(random.uniform(lo, hi), 1) if random.random() > 0.10 else None
        writer.writerow({
            "ticker": ticker,
            "company_name": name,
            "country": country,
            "gics_sector": sector,
            "gics_industry_group": ig,
            "market_cap_usd": round(random.uniform(5e9, 600e9), 2),
            "float_ratio": round(random.uniform(0.7, 1.0), 3),
            "carbon_intensity": ci if ci is not None else "",
        })

print(f"Written {len(STOCKS)} rows to {out}")
