# S&P ESG Index Series

Implementation of the **S&P ESG Index Series** rebalancing methodology.

**Source document:** `methodologies/methodology-sp-esg-index-series.pdf` (S&P Dow Jones Indices)

---

## Index Objective

The S&P ESG Index Series measures the performance of securities meeting sustainability criteria, weighted by **float-adjusted market capitalisation** after applying eligibility screens. Unlike optimizer-based indices (S&P Carbon Aware, S&P PACT), this index uses pure rescaling — eligible stocks retain their relative size-based weights, with excluded stocks' weight redistributed proportionally.

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
3. Rescale surviving weights to sum to 1.0
   │
   ▼
RebalanceResult (rebalanced_weights, diagnostics)
```

---

## Eligibility Criteria

### 1. ESG Score Exclusions
**Source:** §Exclusions Based on S&P Global ESG Score

- **Missing ESG coverage:** Companies without an S&P Global ESG score are excluded.
- **Bottom quartile:** Companies scoring below the 25th percentile of their GICS Industry Group are excluded. The percentile is computed within each industry group, so a company is compared only against sector peers.

### 2. Business Activity Exclusions
**Source:** §Exclusions Based on Business Activities

Companies are excluded if they exceed involvement thresholds for:

| Activity | Metric | Threshold |
|---|---|---|
| Controversial Weapons | Revenue % | > 0% |
| Controversial Weapons | Significant ownership | ≥ 10% |
| Tobacco — Production | Revenue % | > 0% |
| Tobacco — Retail | Revenue % | ≥ 10% |
| Thermal Coal — Extraction | Revenue % | > 5% |
| Thermal Coal — Power Generation | Revenue % | > 25% |
| Small Arms — Manufacture | Revenue % | ≥ 5% |
| Small Arms — Retail | Revenue % | ≥ 10% |

### 3. UNGC / Global Standards Screening
**Source:** §Exclusions Based on Sustainalytics' Global Standards Screening

Exclude companies classified as **Non-Compliant** with UNGC principles or those without GSS coverage. Watchlist companies are **not** excluded.

### 4. MSA Controversy Overlay
**Source:** §Controversies: Media and Stakeholder Analysis Overlay

Companies flagged by the Index Committee following a Media and Stakeholder Analysis (MSA) review are excluded.

---

## Weighting

After exclusions, eligible stocks are weighted by their **float-adjusted market capitalisation**, rescaled so weights sum to 1.0:

```
w_i = underlying_weight_i / Σ(underlying_weight_j)    for all eligible j
```

No optimization or carbon tilt is applied. The index preserves the size-factor exposure of the underlying benchmark, simply removing ineligible companies and redistributing their weight proportionally.

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
| `ungc_status` | string | Compliant / Watchlist / Non-Compliant / No Coverage |
| `msa_flagged` | 0/1 | Whether flagged by MSA review |
| `controversial_weapons_revenue_pct` | float | % revenue from controversial weapons |
| `controversial_weapons_ownership_pct` | float | % ownership in controversial weapons subsidiaries |
| `tobacco_production_revenue_pct` | float | % revenue from tobacco production |
| `tobacco_retail_revenue_pct` | float | % revenue from tobacco retail |
| `thermal_coal_extraction_revenue_pct` | float | % revenue from thermal coal extraction |
| `thermal_coal_power_revenue_pct` | float | % revenue from coal power generation |
| `small_arms_manufacture_revenue_pct` | float | % revenue from small arms manufacture |
| `small_arms_retail_revenue_pct` | float | % revenue from small arms retail |

A sample universe with 40 stocks is provided in `sample_data/universe.csv`.

---

## Usage

### CLI

```bash
# Activate virtual environment first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Unix

# Run a rebalance
python cli.py sp-esg rebalance \
    --input indices/sp_esg/sample_data/universe.csv

# Export results to CSV
python cli.py sp-esg rebalance \
    --input universe.csv \
    --output rebalanced.csv

# Inspect the raw universe
python cli.py sp-esg show-universe \
    --input indices/sp_esg/sample_data/universe.csv
```

### Python API

```python
from indices.sp_esg.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

universe = load_universe_from_csv("indices/sp_esg/sample_data/universe.csv")
result = rebalance(universe)

print(f"Eligible constituents: {len(result.eligible_tickers)}")
print(f"Excluded: {len(result.excluded_tickers)}")

df = result_to_dataframe(result, universe)
print(df.head())
```

---

## Testing

```bash
python -m pytest tests/sp_esg/ -v
python -m pytest tests/sp_esg/ --cov=indices/sp_esg --cov-report=term-missing
```

44 tests across models, eligibility, and the full rebalancing pipeline.

---

## Project Structure

```
indices/sp_esg/
├── models.py          # Stock, IndexUniverse, BusinessActivityExposures, RebalanceResult
├── eligibility.py     # All exclusion filter functions + pipeline
├── rebalancer.py      # Orchestrates eligibility → rescaling → output
├── sample_data/
│   ├── universe.csv           # 40-stock sample universe
│   └── generate_data.py       # Script to regenerate universe.csv
└── README.md
```
