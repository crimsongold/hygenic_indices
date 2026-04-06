# S&P Paris-Aligned & Climate Transition (PACT) Indices

Implementation of the **S&P PACT Indices** rebalancing methodology.

**Source document:** `methodologies/methodology-sp-paris-aligned-climate-transition-pact-indices.pdf` (S&P Dow Jones Indices)

---

## Index Objective

The S&P PACT indices are designed to align with the objectives of the Paris Agreement by selecting and weighting securities to achieve significant carbon intensity reductions while maintaining broad market exposure. Two variants are implemented:

| Variant | Full Name | WACI Reduction Target |
|---|---|---|
| **CTB** | Climate Transition Benchmark | 30% vs. underlying |
| **PAB** | Paris-Aligned Benchmark | 50% vs. underlying |

Both variants enforce a **7% annual self-decarbonization** trajectory and apply increasingly strict exclusion screens (PAB excludes more activities than CTB).

---

## Pipeline Overview

```
CSV Input
   │
   ▼
1. Build universe (float-adjusted market cap weights)
   │
   ▼
2. Variant-specific eligibility filters  ──► excluded_tickers dict (ticker → reason)
   │
   ▼
3. Optimizer (cvxpy / CLARABEL): minimize tracking error
   subject to WACI, ESG, SBTI, sector, and diversification constraints
   │
   ▼
4. Post-process: minimum weight threshold (1 bps)
   │
   ▼
5. WACI diagnostics
   │
   ▼
RebalanceResult (optimized_weights, waci_reduction_pct, solver_status)
```

---

## Eligibility Criteria

### Shared Exclusions (CTB + PAB)

| Filter | Threshold |
|---|---|
| No Trucost carbon data | Excluded |
| Controversial Weapons — Revenue | > 0% |
| Controversial Weapons — Ownership | ≥ 25% |
| Tobacco — Production | > 0% |
| Tobacco — Related products | ≥ 10% |
| Tobacco — Retail | ≥ 10% (CTB) / ≥ 5% (PAB) |
| UNGC Non-Compliant / No Coverage | Excluded |
| MSA Flagged | Excluded |

### PAB-Only Exclusions

| Filter | Threshold |
|---|---|
| Small Arms (all categories) | > 0% |
| Military — Integral weapons | > 0% |
| Military — Weapon-related | ≥ 5% |
| Thermal Coal Generation | ≥ 5% |
| Oil Sands Extraction | ≥ 5% |
| Shale Oil/Gas Extraction | ≥ 5% |
| Gambling Operations | ≥ 10% |
| Alcohol — Production | ≥ 5% |
| Alcohol — Related/Retail | ≥ 10% |
| Fossil Fuel — Coal revenue | ≥ 1% |
| Fossil Fuel — Oil revenue | ≥ 10% |
| Fossil Fuel — Gas revenue | ≥ 50% |
| Fossil Fuel — Power generation | ≥ 50% |

---

## Optimization

### Objective Function

Minimize tracking error (squared relative deviations from parent index weights):

```
minimize  Σᵢ (w_i - W_i)² / W_i
```

Where `w_i` = optimized weight, `W_i` = underlying benchmark weight (rescaled to eligible stocks).

### Constraints

| Constraint | Formula / Target |
|---|---|
| Weights sum to 1 | Σw_i = 1 |
| WACI target | WACI ≤ underlying × (1 - reduction) × 0.95 buffer |
| CTB reduction | 30% |
| PAB reduction | 50% |
| Annual decarbonization | 7% per year (applied to anchor WACI) |
| SBTI weight | ≥ 120% of underlying SBTI weight |
| ESG score | Portfolio ESG ≥ underlying ESG |
| High climate impact sectors | Weight ≥ underlying weight in those sectors |
| Non-disclosing companies | Weight ≤ 110% of underlying |
| Country bounds | ±5% of underlying country weight |
| Stock weight lower | max(0, W_i - 2%) |
| Stock weight upper | min(W_i + 2%, 5%) |

**High climate impact sectors:** Energy, Materials, Utilities, Transportation, Capital Goods, Automobiles & Components.

### Constraint Relaxation Hierarchy

If the optimization fails to find a feasible solution, stock weight bounds are progressively relaxed across 3 levels:

| Level | Stock Lower | Stock Upper |
|---|---|---|
| 0 (default) | max(0, W_i - 2%) | min(W_i + 2%, 5%) |
| 1 | max(0, W_i - 4%) | min(W_i + 4%, 8%) |
| 2 | max(0, W_i - 6%) | min(W_i + 6%, 10%) |

If all levels fail, falls back to rescaled benchmark weights.

### Solver

Uses **CLARABEL** via cvxpy for convex quadratic optimization.

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
| `scope_1_2_carbon_intensity` | float \| blank | Scope 1+2 carbon intensity (tCO₂e/$M) |
| `scope_3_carbon_intensity` | float \| blank | Scope 3 carbon intensity (tCO₂e/$M) |
| `has_carbon_coverage` | 0/1 | Whether Trucost data exists |
| `has_sbti_target` | 0/1 | Whether company has Science Based Targets |
| `ungc_status` | string | Compliant / Watchlist / Non-Compliant / No Coverage |
| `msa_flagged` | 0/1 | Whether flagged by MSA review |
| `controversial_weapons_revenue_pct` | float | % revenue from controversial weapons |
| `controversial_weapons_ownership_pct` | float | % ownership in controversial weapons |
| `tobacco_production_revenue_pct` | float | % revenue from tobacco production |
| `tobacco_related_revenue_pct` | float | % revenue from tobacco-related products |
| `tobacco_retail_revenue_pct` | float | % revenue from tobacco retail |
| `small_arms_civilian_revenue_pct` | float | % revenue from civilian small arms |
| `small_arms_noncivilian_revenue_pct` | float | % revenue from non-civilian small arms |
| `small_arms_key_components_revenue_pct` | float | % revenue from small arms components |
| `small_arms_retail_revenue_pct` | float | % revenue from small arms retail |
| `military_integral_weapons_revenue_pct` | float | % revenue from integral weapons |
| `military_weapon_related_revenue_pct` | float | % revenue from weapon-related products |
| `thermal_coal_generation_revenue_pct` | float | % revenue from thermal coal generation |
| `oil_sands_extraction_revenue_pct` | float | % revenue from oil sands |
| `shale_oil_gas_extraction_revenue_pct` | float | % revenue from shale oil/gas |
| `gambling_operations_revenue_pct` | float | % revenue from gambling operations |
| `alcohol_production_revenue_pct` | float | % revenue from alcohol production |
| `alcohol_related_revenue_pct` | float | % revenue from alcohol-related products |
| `alcohol_retail_revenue_pct` | float | % revenue from alcohol retail |
| `coal_revenue_pct` | float | % revenue from coal (fossil fuel screen) |
| `oil_revenue_pct` | float | % revenue from oil (fossil fuel screen) |
| `natural_gas_revenue_pct` | float | % revenue from natural gas |
| `power_generation_revenue_pct` | float | % revenue from power generation |

A sample universe with 60 stocks is provided in `sample_data/universe.csv`.

---

## Usage

### CLI

```bash
# Activate virtual environment first
venv\Scripts\activate          # Windows
source venv/bin/activate       # Unix

# Run a CTB rebalance
python cli.py sp-pact rebalance \
    --input indices/sp_pact/sample_data/universe.csv \
    --variant ctb

# Run a PAB rebalance and export results
python cli.py sp-pact rebalance \
    --input indices/sp_pact/sample_data/universe.csv \
    --variant pab \
    --output rebalanced.csv

# Inspect the raw universe
python cli.py sp-pact show-universe \
    --input indices/sp_pact/sample_data/universe.csv
```

### Python API

```python
from indices.sp_pact.models import Variant
from indices.sp_pact.rebalancer import load_universe_from_csv, rebalance, result_to_dataframe

universe = load_universe_from_csv("indices/sp_pact/sample_data/universe.csv")
result = rebalance(universe, variant=Variant.PAB)

print(f"Variant: {result.variant}")
print(f"WACI reduction: {result.waci_reduction_pct:.1f}%")
print(f"Solver status: {result.solver_status}")
print(f"Relaxation level: {result.relaxation_level}")
print(f"Eligible: {len(result.eligible_tickers)}, Excluded: {len(result.excluded_tickers)}")

df = result_to_dataframe(result, universe)
print(df.head())
```

---

## Testing

```bash
python -m pytest tests/sp_pact/ -v
python -m pytest tests/sp_pact/ --cov=indices/sp_pact --cov-report=term-missing
```

37 tests across models, eligibility (shared + PAB-only filters), and the full rebalancing pipeline with both CTB and PAB variants.

---

## Project Structure

```
indices/sp_pact/
├── models.py          # Stock, IndexUniverse, BusinessActivityExposures, Variant, RebalanceResult
├── eligibility.py     # Shared + variant-specific exclusion filters
├── optimization.py    # cvxpy optimizer with WACI/ESG/SBTI/diversification constraints
├── rebalancer.py      # Orchestrates eligibility → optimization → diagnostics → output
├── sample_data/
│   ├── universe.csv           # 60-stock sample universe
│   └── generate_data.py       # Script to regenerate universe.csv
└── README.md
```
