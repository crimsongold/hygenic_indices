# S&P Global Carbon Efficient Index Series

Implementation of the **S&P Global Carbon Efficient Index Series** rebalancing methodology.

**Source document:** `methodologies/methodology-sp-global-carbon-efficient-index-series.pdf` (S&P Dow Jones Indices)

---

## Index Objective

The Carbon Efficient index measures all securities from an underlying benchmark, weighted using a **carbon efficiency tilt**. Unlike exclusion-based indices, every benchmark constituent remains in the index — stocks with lower carbon intensity receive higher weights, and stocks with higher carbon intensity receive lower weights, relative to their GICS Industry Group peers.

The tilt is industry-group-neutral: total weight allocated to each GICS Industry Group matches the benchmark. Only the within-group distribution of weight changes based on carbon performance.

---

## Pipeline Overview

```
CSV Input
   │
   ▼
1. Build universe (float-adjusted market cap weights)
   │
   ▼
2. Compute Carbon Efficiency Factors (CEF) per GICS Industry Group
   │
   ▼
3. Apply multiplicative tilt: w_i = W_i × CEF_i / Σ(W_j × CEF_j)
   │
   ▼
4. Compute WACI diagnostics (pre-tilt vs. post-tilt)
   │
   ▼
RebalanceResult (tilted_weights, waci_reduction_pct)
```

---

## Carbon Efficiency Factor (CEF)

### Methodology

The published methodology uses a decile-based system where stocks within each GICS Industry Group are ranked by carbon-to-revenue footprint and assigned to deciles 1–10, each receiving a fixed percentage weight adjustment scaled by the group's Impact Factor.

Our implementation uses a simplified **exponential tilt** approximation:

```
z_i = (CI_i - mean(CI_group)) / std(CI_group)    [within-group z-score]
CEF_i = exp(-λ × z_i)
```

Where:
- `CI_i` = carbon intensity (tCO₂e per $M revenue, from Trucost)
- `λ` = tilt strength parameter (default: 0.5)
- `z_i > 0` → above-average CI (dirtier) → CEF < 1 → weight reduced
- `z_i < 0` → below-average CI (cleaner) → CEF > 1 → weight increased

### Special Cases

| Condition | Treatment |
|---|---|
| No Trucost coverage (CI = None) | CEF = 1.0 (neutral — weight unchanged) |
| Industry group with < 2 stocks with CI data | CEF = 1.0 for all members |
| All stocks in group have identical CI | CEF = 1.0 (zero variance) |

The neutral treatment of non-disclosed companies matches the methodology's handling of stocks in the 4th–7th decile "Not-disclosed" category with a 0% weight adjustment.

### Final Weights

```
w_i = W_i × CEF_i / Σ(W_j × CEF_j)
```

Normalized to sum to 1.0 across the entire universe.

---

## Tilt Parameter (λ)

| λ Value | Effect |
|---|---|
| 0.0 | No tilt — index equals benchmark |
| 0.5 | Standard tilt (default) |
| > 1.0 | Aggressive tilt — large overweight of clean companies |

Higher λ produces greater WACI reduction but increases tracking error relative to the benchmark.

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
| `carbon_intensity` | float \| blank | tCO₂e per million USD revenue (Trucost) |

A sample universe with 45 stocks is provided in `sample_data/universe.csv`.

---

## Usage

### CLI

```bash
# Activate virtual environment first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Unix

# Run a rebalance (default λ = 0.5)
python cli.py sp-carbon-efficient rebalance \
    --input indices/sp_carbon_efficient/sample_data/universe.csv

# Export results to CSV
python cli.py sp-carbon-efficient rebalance \
    --input universe.csv \
    --output rebalanced.csv

# Inspect the raw universe
python cli.py sp-carbon-efficient show-universe \
    --input indices/sp_carbon_efficient/sample_data/universe.csv
```

### Python API

```python
from indices.sp_carbon_efficient.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

universe = load_universe_from_csv("indices/sp_carbon_efficient/sample_data/universe.csv")
result = rebalance(universe, tilt_lambda=0.5)

print(f"WACI reduction: {result.underlying_waci:.1f} → {result.weighted_avg_carbon_intensity:.1f} tCO2e/$M")
print(f"Reduction: {result.waci_reduction_pct:.1f}%")

df = result_to_dataframe(result, universe)
print(df.head())
```

---

## Testing

```bash
python -m pytest tests/sp_carbon_efficient/ -v
python -m pytest tests/sp_carbon_efficient/ --cov=indices/sp_carbon_efficient --cov-report=term-missing
```

20 tests across models, weighting, and the full rebalancing pipeline.

---

## Project Structure

```
indices/sp_carbon_efficient/
├── models.py          # Stock, IndexUniverse, RebalanceResult Pydantic models
├── weighting.py       # CEF computation + multiplicative tilt logic
├── rebalancer.py      # Orchestrates tilt → WACI diagnostics → output
├── sample_data/
│   ├── universe.csv           # 45-stock sample universe
│   └── generate_data.py       # Script to regenerate universe.csv
└── README.md
```
