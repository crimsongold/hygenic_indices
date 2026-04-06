# S&P Carbon Aware Index Series

Implementation of the **S&P Carbon Aware Index Series** rebalancing methodology.

**Source document:** `methodologies/methodology-sp-carbon-aware-index-series.pdf` (S&P Dow Jones Indices, March 2026)

---

## Index Objective

Each index in the series measures eligible securities from an underlying index, weighted to **minimize weighted-average carbon intensity** (WACI), subject to diversification and tracking constraints.

> "Each index measures the performance of eligible securities from an underlying index weighted to minimize the weighted average carbon intensity, subject to index active share, industry group weight, country weight, and diversification constraints."
> — Methodology, §Index Objective and Highlights

Two variants are implemented:

| Index | Underlying Index |
|---|---|
| S&P Developed ex-Australia LargeMidCap Carbon Aware | S&P Developed ex-Australia LargeMidCap |
| S&P Emerging LargeMidCap Carbon Aware | S&P Emerging LargeMidCap (incl. China A shares) |

---

## Pipeline Overview

```
CSV Input
   │
   ▼
1. Build universe (float-adjusted market cap weights)
   │
   ▼
2. Eligibility filters  ──► excluded_tickers dict (ticker → reason)
   │
   ▼
3. Optimizer (cvxpy / CLARABEL)
   │
   ▼
4. Post-process: minimum weight threshold (1 bps)
   │
   ▼
RebalanceResult (optimized_weights, diagnostics)
```

---

## Eligibility Criteria

### 1. ESG Score Exclusions
**Source:** §Exclusions Based on S&P Global ESG Score (p. 5)

- **All variants:** Companies without an S&P Global ESG score are excluded.
- **Developed:** Exclude companies whose score is strictly below the 25th percentile of their **global** GICS Industry Group (using combined S&P Global LargeMidCap + S&P Global 1200).
- **Emerging:** Exclude companies below the 25th percentile of their underlying index GICS Industry Group. Companies with no score are always treated as worst performers.

### 2. Business Activity Exclusions
**Source:** §Exclusions Based on Business Activities (pp. 5–6), Sustainalytics data

Companies are excluded if they exceed Sustainalytics involvement thresholds for:

| Activity | Metric | Threshold |
|---|---|---|
| Controversial Weapons (tailor-made/essential) | Revenue % | > 0% |
| Controversial Weapons (tailor-made/essential) | Significant ownership | ≥ 10% |
| Controversial Weapons (non-tailor-made) | Revenue % | > 0% |
| Controversial Weapons (non-tailor-made) | Significant ownership | ≥ 10% |
| Tobacco — Production | Revenue % | > 0% |
| Tobacco — Related Products/Services | Revenue % | ≥ 5% |
| Tobacco — Retail | Revenue % | ≥ 5% |
| Thermal Coal — Extraction | Revenue % | > 0% |
| Thermal Coal — Power Generation | Revenue % | > 0% |
| Oil Sands — Extraction | Revenue % | > 0% |
| Shale Energy — Extraction | Revenue % | > 0% |
| Arctic Oil & Gas — Extraction | Revenue % | > 0% |
| Oil & Gas — Production | Revenue % | > 0% |
| Oil & Gas — Electricity Generation | Revenue % | > 0% |
| Oil & Gas — Supporting Products/Services | Revenue % | ≥ 10% |
| Gambling — Operations | Revenue % | ≥ 5% |
| Gambling — Specialized Equipment | Revenue % | ≥ 10% |
| Gambling — Supporting Products/Services | Revenue % | ≥ 10% |
| Adult Entertainment — Production | Revenue % | > 0% |
| Adult Entertainment — Distribution | Revenue % | ≥ 5% |
| Alcoholic Beverages — Production | Revenue % | ≥ 5% |
| Alcoholic Beverages — Retail | Revenue % | ≥ 10% |
| Alcoholic Beverages — Related Products/Services | Revenue % | ≥ 10% |

Companies without Sustainalytics coverage are also excluded.

### 3. UNGC / Global Standards Screening
**Source:** §Exclusions Based on Sustainalytics' Global Standards Screening (p. 7)

Exclude companies classified as **Non-Compliant** with UNGC principles (per Sustainalytics GSS), and companies without GSS coverage.

Watchlist companies are **not** excluded at rebalancing (only at the quarterly UNGC review).

### 4. MSA Controversy Overlay
**Source:** §Controversies: Media and Stakeholder Analysis Overlay (p. 7)

Companies flagged by the Index Committee following a Media and Stakeholder Analysis (MSA) review are excluded. Excluded companies are ineligible for re-entry for at least one full calendar year.

---

## Optimization

### Objective Function
**Source:** §The Optimization Objective Function (p. 8)

Minimize the weighted-average carbon intensity of eligible constituents:

```
minimize  Σᵢ (optimized_weight_i × carbon_intensity_i)
```

**Carbon Intensity** = annual GHG emissions (tCO₂e, direct + first-tier indirect) ÷ annual revenue (millions USD). Provided by S&P Trucost.

**Missing carbon intensity** values are imputed with the median carbon intensity from stocks in the same GICS Industry Group. Source: footnote 2, p. 8.

### Constraints
**Source:** §Optimization Constraints (pp. 8–9)

| Constraint | Formula |
|---|---|
| Weights sum to 1 | Σwᵢ = 1 |
| Country weight lower | max(underlying − 5%, underlying × 0.75) |
| Country weight upper | min(underlying + 5%, underlying × 1.25) |
| Industry Group weight lower | min(Σ eligible × 10×, max(underlying − 5%, underlying × 0.75)) |
| Industry Group weight upper | min(underlying + 5%, underlying × 1.25) |
| Stock weight upper | min(underlying × 10, underlying + 2%) |
| Stock weight lower | max(0, underlying − 2%) |
| Min stock weight (hard) | 0.01% (1 bps) — achieved by post-processing |
| Diversification | Σ (W'ᵢ − wᵢ)² / W'ᵢ ≤ Σ (2T + ineligible/N)² / W'ᵢ |
| Active share | Σ|Wᵢ − wᵢ| / 2 ≤ T + ineligible weight |

Where:
- `W'ᵢ = Wᵢ + ineligible_weight / N_eligible`
- `T = max(5%, 25% − ineligible_weight)`

### Constraint Relaxation Hierarchy
**Source:** §Constraint Relaxation Hierarchy (p. 9)

If the optimization fails to find a feasible solution, constraints are relaxed in this order until a solution is found:

1. **Stock weight bounds** — excess weight relaxed from 2% → 4%; multiplier from 10× → 20×
2. **T (active share parameter)** — reduced toward the 5% floor
3. **Industry Group bounds** — further relaxed
4. **Country bounds** — further relaxed

Hard constraints that are never relaxed: minimum stock weight (1 bps).

---

## Index Maintenance

### Rebalancing Schedule
**Source:** §Rebalancing (p. 11)

- **Frequency:** Semi-annual
- **Effective date:** After close of last business day of **April** and **October**
- **Reference date:** Last trading day of **March** and **September**
- Weights calculated from reference date data are implemented using closing prices 7 business days prior to the effective date

### Quarterly UNGC Review
**Source:** §Quarterly UNGC Eligibility Review (p. 11)

UNGC compliance is reviewed quarterly, effective after the close of the third Friday of March, June, September, and December. Reference date: last business day of the previous month.

### Additions and Deletions
**Source:** §Additions and Deletions (p. 11)

- **No additions** between rebalancings except spin-offs (added at zero price, removed after at least one day of trading).
- **Deletions** occur when a stock drops from the underlying index, or due to corporate events (mergers, delistings, bankruptcies) or an MSA flag.

---

## CSV Input Format

The universe CSV must include these columns:

| Column | Type | Description |
|---|---|---|
| `ticker` | string | Stock identifier |
| `company_name` | string | Full company name |
| `country` | string | ISO country code |
| `gics_sector` | string | GICS Sector |
| `gics_industry_group` | string | GICS Industry Group |
| `market_cap_usd` | float | Total market cap in USD |
| `float_ratio` | float | Free-float fraction (0–1) |
| `esg_score` | float \| blank | S&P Global ESG Score (0–100) |
| `has_esg_coverage` | 0/1 | Whether ESG score exists |
| `carbon_intensity` | float \| blank | tCO₂e per million USD revenue (Trucost) |
| `ungc_status` | string | Compliant / Watchlist / Non-Compliant / No Coverage |
| `msa_flagged` | 0/1 | Whether flagged by MSA review |
| `has_sustainalytics_coverage` | 0/1 | Whether Sustainalytics data exists |
| `controversial_weapons_tailor_made_essential_pct` | float | % revenue from tailor-made/essential weapons |
| `controversial_weapons_tailor_made_essential_ownership_pct` | float | % ownership in tailor-made/essential weapons subsidiaries |
| `controversial_weapons_non_tailor_made_pct` | float | % revenue from non-tailor-made weapons |
| `controversial_weapons_non_tailor_made_ownership_pct` | float | % ownership in non-tailor-made weapons subsidiaries |
| `tobacco_production_revenue_pct` | float | % revenue from tobacco production |
| `tobacco_related_revenue_pct` | float | % revenue from tobacco-related products |
| `tobacco_retail_revenue_pct` | float | % revenue from tobacco retail |
| `thermal_coal_extraction_revenue_pct` | float | % revenue from thermal coal extraction |
| `thermal_coal_power_generation_revenue_pct` | float | % revenue from coal power generation |
| `oil_sands_extraction_revenue_pct` | float | % revenue from oil sands |
| `shale_energy_extraction_revenue_pct` | float | % revenue from shale energy |
| `arctic_oil_gas_extraction_revenue_pct` | float | % revenue from Arctic oil & gas |
| `oil_gas_production_revenue_pct` | float | % revenue from oil & gas production/E&P |
| `oil_gas_generation_revenue_pct` | float | % revenue from oil/gas power generation |
| `oil_gas_supporting_revenue_pct` | float | % revenue from oil & gas supporting services |
| `gambling_operations_revenue_pct` | float | % revenue from gambling operations |
| `gambling_equipment_revenue_pct` | float | % revenue from gambling equipment |
| `gambling_supporting_revenue_pct` | float | % revenue from gambling support |
| `adult_entertainment_production_revenue_pct` | float | % revenue from adult entertainment production |
| `adult_entertainment_distribution_revenue_pct` | float | % revenue from adult entertainment distribution |
| `alcoholic_beverages_production_revenue_pct` | float | % revenue from alcohol production |
| `alcoholic_beverages_retail_revenue_pct` | float | % revenue from alcohol retail |
| `alcoholic_beverages_related_revenue_pct` | float | % revenue from alcohol-related products |

A sample universe with 56 stocks is provided in `sample_data/universe.csv`.

---

## Usage

### CLI

```bash
# Activate virtual environment first
source venv/bin/activate       # Unix
venv\Scripts\activate          # Windows

# Run a rebalance (developed variant)
python cli.py sp-carbon-aware rebalance \
    --input indices/sp_carbon_aware/sample_data/universe.csv

# Run a rebalance (emerging variant) and export results
python cli.py sp-carbon-aware rebalance \
    --input universe.csv \
    --universe-type emerging \
    --output rebalanced.csv

# Inspect the raw universe
python cli.py sp-carbon-aware show-universe \
    --input indices/sp_carbon_aware/sample_data/universe.csv
```

### Python API

```python
from indices.sp_carbon_aware.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

universe = load_universe_from_csv("indices/sp_carbon_aware/sample_data/universe.csv")
result = rebalance(universe, universe_type="developed")

print(f"WACI reduction: {result.underlying_weighted_avg_carbon_intensity:.1f} → {result.weighted_avg_carbon_intensity:.1f} tCO2e/$M")
print(f"Eligible constituents: {len(result.eligible_tickers)}")
print(f"Excluded: {len(result.excluded_tickers)}")

df = result_to_dataframe(result, universe)
print(df.head())
```

---

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=indices/sp_carbon_aware --cov-report=term-missing
```

100 tests across models, eligibility, optimization, and the full rebalancing pipeline.

---

## Project Structure

```
indices/sp_carbon_aware/
├── models.py          # Stock, IndexUniverse, BusinessActivityExposures Pydantic models
├── eligibility.py     # All exclusion filter functions
├── optimization.py    # cvxpy-based optimizer with constraint relaxation
├── rebalancer.py      # Orchestrates eligibility → optimization → output
├── sample_data/
│   ├── universe.csv           # 56-stock sample universe
│   └── generate_data.py       # Script to regenerate universe.csv
└── README.md
```
