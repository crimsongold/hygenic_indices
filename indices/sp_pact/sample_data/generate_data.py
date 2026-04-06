"""Generate synthetic sample data for the S&P PACT index."""

from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    random.seed(42)
    out = Path(__file__).parent / "universe.csv"

    sectors = ["Information Technology", "Health Care", "Financials", "Energy", "Utilities", "Materials", "Industrials"]
    industry_groups = {
        "Information Technology": "Software & Services",
        "Health Care": "Pharmaceuticals",
        "Financials": "Banks",
        "Energy": "Energy",
        "Utilities": "Utilities",
        "Materials": "Materials",
        "Industrials": "Capital Goods",
    }
    countries = ["US", "GB", "DE", "JP", "CA", "FR", "AU"]

    fieldnames = [
        "ticker", "company_name", "country", "gics_sector", "gics_industry_group",
        "market_cap_usd", "float_ratio",
        "esg_score", "has_esg_coverage",
        "scope_1_2_carbon_intensity", "scope_3_carbon_intensity",
        "has_carbon_coverage", "has_sbti_target",
        "ungc_status", "msa_flagged",
        "controversial_weapons_revenue_pct", "controversial_weapons_ownership_pct",
        "tobacco_production_revenue_pct", "tobacco_related_revenue_pct", "tobacco_retail_revenue_pct",
        "small_arms_civilian_revenue_pct", "small_arms_noncivilian_revenue_pct",
        "small_arms_key_components_revenue_pct", "small_arms_retail_revenue_pct",
        "military_integral_weapons_revenue_pct", "military_weapon_related_revenue_pct",
        "thermal_coal_generation_revenue_pct",
        "oil_sands_extraction_revenue_pct", "shale_oil_gas_extraction_revenue_pct",
        "gambling_operations_revenue_pct",
        "alcohol_production_revenue_pct", "alcohol_related_revenue_pct", "alcohol_retail_revenue_pct",
        "coal_revenue_pct", "oil_revenue_pct", "natural_gas_revenue_pct", "power_generation_revenue_pct",
    ]

    rows = []
    for i in range(1, 61):
        sector = random.choice(sectors)
        row = {
            "ticker": f"P{i:03d}",
            "company_name": f"PACTCo {i}",
            "country": random.choice(countries),
            "gics_sector": sector,
            "gics_industry_group": industry_groups[sector],
            "market_cap_usd": round(random.uniform(1e9, 5e10), 0),
            "float_ratio": round(random.uniform(0.5, 1.0), 2),
            "esg_score": round(random.uniform(30, 95), 1),
            "has_esg_coverage": True,
            "scope_1_2_carbon_intensity": round(random.uniform(5, 500), 1),
            "scope_3_carbon_intensity": round(random.uniform(10, 300), 1),
            "has_carbon_coverage": True,
            "has_sbti_target": random.random() < 0.3,
            "ungc_status": "Compliant",
            "msa_flagged": False,
        }
        # Zero out all business activity fields
        for f in fieldnames:
            if f not in row:
                row[f] = 0.0

        # Sprinkle exclusion triggers
        if i == 5:
            row["tobacco_production_revenue_pct"] = 2.0
        elif i == 10:
            row["controversial_weapons_revenue_pct"] = 0.5
        elif i == 15:
            row["ungc_status"] = "Non-Compliant"
        elif i == 20:
            row["coal_revenue_pct"] = 3.0  # PAB only
        elif i == 25:
            row["oil_revenue_pct"] = 15.0  # PAB only
        elif i == 30:
            row["has_carbon_coverage"] = False
            row["scope_1_2_carbon_intensity"] = ""
            row["scope_3_carbon_intensity"] = ""
        elif i == 35:
            row["military_integral_weapons_revenue_pct"] = 2.0  # PAB only
        elif i == 40:
            row["gambling_operations_revenue_pct"] = 12.0  # PAB only
        elif i == 45:
            row["msa_flagged"] = True
        elif i == 50:
            row["alcohol_production_revenue_pct"] = 8.0  # PAB only

        rows.append(row)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
