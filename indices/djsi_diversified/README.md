# Dow Jones Sustainability Diversified Indices

Implementation of the **Dow Jones Sustainability Diversified Indices** rebalancing methodology.

**Source document:** `methodologies/methodology-dj-sustainability-diversified-indices.pdf` (S&P Dow Jones Indices)

---

## Index Objective

The DJSI Diversified indices select the highest-scoring companies by S&P Global ESG Score (derived from the Corporate Sustainability Assessment) within each GICS Sector × Region group, targeting approximately **50% of each group's float-adjusted market capitalisation**. Selected companies are weighted by float-adjusted market cap, with a **single-stock cap of 10%**.

---

## Pipeline Overview

```
CSV Input
   │
   ▼
1. Build universe (float-adjusted market cap weights)
   │
   ▼
2. Hard exclusion screens  ──► excluded_tickers dict (ticker → reason)
   │
   ▼
3. Best-in-class ESG selection (per GICS Sector × Region)
   │
   ▼
4. Float-cap weighting of selected companies
   │
   ▼
5. Single-stock cap (10%) with iterative redistribution
   │
   ▼
RebalanceResult (rebalanced_weights, selected_tickers, capped_tickers)
```

---

## Eligibility Criteria (Hard Exclusions)

### Business Activity Exclusions
**Source:** §Exclusions Based on Business Activities

Companies are excluded if they exceed involvement thresholds for:

| Activity | Metric | Threshold |
|---|---|---|
| Controversial Weapons | Revenue % | > 0% |
| Controversial Weapons | Significant ownership | ≥ 10% |
| Tobacco — Production | Revenue % | > 0% |
| Adult Entertainment — Production | Revenue % | ≥ 5% |
| Adult Entertainment — Retail | Revenue % | ≥ 5% |
| Alcohol — Production | Revenue % | > 0% |
| Gambling — Operations | Revenue % | > 0% |
| Gambling — Equipment | Revenue % | > 0% |
| Military Contracting — Integral weapons | Revenue % | ≥ 5% |
| Military Contracting — Weapon-related | Revenue % | ≥ 5% |
| Small Arms — Civilian production | Revenue % | ≥ 5% |
| Small Arms — Key components | Revenue % | ≥ 5% |
| Small Arms — Non-civilian production | Revenue % | ≥ 5% |
| Small Arms — Retail | Revenue % | ≥ 5% |

### ESG Coverage
Companies without ESG score coverage are excluded.

### UNGC / Global Standards Screening
Exclude companies classified as **Non-Compliant** or without GSS coverage. Watchlist companies are retained.

### MSA Controversy Overlay
Companies flagged by the Index Committee via Media and Stakeholder Analysis are excluded.

---

## Best-in-Class Selection

**Source:** §Component Selection

After hard exclusions, the methodology selects the best ESG performers within each **GICS Sector × Region** group:

1. **Sort** stocks within each group by ESG score (descending).
2. **Greedily add** stocks until cumulative float-adjusted market cap reaches the **40% initial threshold** of the group total.
3. **Buffer zone (40%–60%):** Continue adding stocks in the boundary region, checking proximity to the 50% target.
4. **Target:** Approximately 50% of each group's float-adjusted market cap.

| Parameter | Value |
|---|---|
| Float-cap target | 50% |
| Initial threshold (greedy phase) | 40% |
| Buffer upper bound | 60% |

---

## Weight Capping

Selected companies are weighted by float-adjusted market capitalisation. A **10% single-stock cap** is enforced using an iterative locked-set algorithm:

1. Identify stocks exceeding the 10% cap.
2. Lock them at 10%.
3. Redistribute remaining budget proportionally to unlocked stocks.
4. Repeat until no stock exceeds the cap (max 50 iterations).

---

## CSV Input Format

The universe CSV must include these columns:

| Column | Type | Description |
|---|---|---|
| `ticker` | string | Stock identifier |
| `company_name` | string | Full company name |
| `country` | string | ISO country code |
| `region` | string | Geographic region (e.g. North America, EMEA, Asia/Pacific) |
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
| `adult_entertainment_production_revenue_pct` | float | % revenue from adult entertainment production |
| `adult_entertainment_retail_revenue_pct` | float | % revenue from adult entertainment retail |
| `alcohol_production_revenue_pct` | float | % revenue from alcohol production |
| `gambling_operations_revenue_pct` | float | % revenue from gambling operations |
| `gambling_equipment_revenue_pct` | float | % revenue from gambling equipment |
| `military_integral_weapons_revenue_pct` | float | % revenue from integral weapons systems |
| `military_weapon_related_revenue_pct` | float | % revenue from weapon-related products |
| `small_arms_civilian_production_revenue_pct` | float | % revenue from civilian small arms |
| `small_arms_key_components_revenue_pct` | float | % revenue from key small arms components |
| `small_arms_noncivilian_production_revenue_pct` | float | % revenue from non-civilian small arms |
| `small_arms_retail_revenue_pct` | float | % revenue from small arms retail |

A sample universe with 118 stocks is provided in `sample_data/universe.csv`.

---

## Usage

### CLI

```bash
# Activate virtual environment first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Unix

# Run a rebalance
python cli.py djsi-diversified rebalance \
    --input indices/djsi_diversified/sample_data/universe.csv

# Export results to CSV
python cli.py djsi-diversified rebalance \
    --input universe.csv \
    --output rebalanced.csv

# Inspect the raw universe
python cli.py djsi-diversified show-universe \
    --input indices/djsi_diversified/sample_data/universe.csv
```

### Python API

```python
from indices.djsi_diversified.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

universe = load_universe_from_csv("indices/djsi_diversified/sample_data/universe.csv")
result = rebalance(universe)

print(f"Selected: {len(result.selected_tickers)} companies")
print(f"Excluded: {len(result.excluded_tickers)}")
print(f"Capped at 10%: {len(result.capped_tickers)}")

df = result_to_dataframe(result, universe)
print(df.head())
```

---

## Testing

```bash
python -m pytest tests/djsi_diversified/ -v
python -m pytest tests/djsi_diversified/ --cov=indices/djsi_diversified --cov-report=term-missing
```

46 tests across models, eligibility (including all 10 exclusion filters and best-in-class selection), and the full rebalancing pipeline.

---

## Project Structure

```
indices/djsi_diversified/
├── models.py          # Stock, IndexUniverse, BusinessActivityExposures, RebalanceResult
├── eligibility.py     # Hard exclusion filters + best-in-class ESG selection
├── rebalancer.py      # Orchestrates exclusions → selection → capping → output
├── sample_data/
│   ├── universe.csv           # 118-stock sample universe
│   └── generate_data.py       # Script to regenerate universe.csv
└── README.md
```
