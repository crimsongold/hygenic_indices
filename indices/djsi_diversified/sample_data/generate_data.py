"""Generate synthetic sample data for the DJSI Diversified index."""

from __future__ import annotations

import csv
import random
from pathlib import Path


def main() -> None:
    random.seed(42)
    out = Path(__file__).parent / "universe.csv"

    sectors = ["Information Technology", "Health Care", "Financials", "Energy", "Industrials"]
    regions = ["North America", "EMEA", "Asia/Pacific"]
    countries_by_region = {
        "North America": ["US", "CA"],
        "EMEA": ["GB", "DE", "FR"],
        "Asia/Pacific": ["JP", "AU", "KR"],
    }

    fieldnames = [
        "ticker", "company_name", "country", "region", "gics_sector", "gics_industry_group",
        "market_cap_usd", "float_ratio", "esg_score", "has_esg_coverage",
        "ungc_status", "msa_flagged",
        "controversial_weapons_revenue_pct", "controversial_weapons_ownership_pct",
        "tobacco_production_revenue_pct",
        "adult_entertainment_production_revenue_pct", "adult_entertainment_retail_revenue_pct",
        "alcohol_production_revenue_pct",
        "gambling_operations_revenue_pct", "gambling_equipment_revenue_pct",
        "military_integral_weapons_revenue_pct", "military_weapon_related_revenue_pct",
        "small_arms_civilian_production_revenue_pct", "small_arms_key_components_revenue_pct",
        "small_arms_noncivilian_production_revenue_pct", "small_arms_retail_revenue_pct",
    ]

    rows = []
    idx = 0
    for sector in sectors:
        for region in regions:
            n_stocks = random.randint(6, 10)
            for j in range(n_stocks):
                idx += 1
                country = random.choice(countries_by_region[region])
                esg = round(random.uniform(20, 95), 1)
                mcap = round(random.uniform(5e8, 1e11), 0)

                row = {
                    "ticker": f"D{idx:03d}",
                    "company_name": f"DJSICo {idx}",
                    "country": country,
                    "region": region,
                    "gics_sector": sector,
                    "gics_industry_group": f"{sector} Group A",
                    "market_cap_usd": mcap,
                    "float_ratio": round(random.uniform(0.5, 1.0), 2),
                    "esg_score": esg,
                    "has_esg_coverage": True,
                    "ungc_status": "Compliant",
                    "msa_flagged": False,
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

                # Sprinkle some exclusion triggers
                if idx == 3:
                    row["tobacco_production_revenue_pct"] = 5.0
                elif idx == 7:
                    row["ungc_status"] = "Non-Compliant"
                elif idx == 12:
                    row["controversial_weapons_revenue_pct"] = 1.0
                elif idx == 18:
                    row["gambling_operations_revenue_pct"] = 2.0
                elif idx == 25:
                    row["has_esg_coverage"] = False
                    row["esg_score"] = ""
                elif idx == 30:
                    row["military_integral_weapons_revenue_pct"] = 8.0
                elif idx == 35:
                    row["alcohol_production_revenue_pct"] = 3.0
                elif idx == 40:
                    row["msa_flagged"] = True

                rows.append(row)

    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
