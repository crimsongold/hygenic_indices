# hygenic_indices

Python implementations of S&P Global index rebalancing methodologies. Each index is a self-contained package that applies the published eligibility rules and optimization logic from the corresponding S&P Dow Jones Indices methodology document.

The goal is to make the business logic of these indices transparent, testable, and reproducible — without requiring access to proprietary data vendors. Sample universes are provided as CSVs so the full pipeline can be exercised out of the box.

---

## Indices

| Index | Approach | Methodology | Status |
|---|---|---|---|
| [S&P Carbon Aware Index Series](indices/sp_carbon_aware/README.md) | Exclusions + cvxpy optimizer (minimize WACI) | March 2026 | ✅ Implemented |
| [S&P ESG Index Series](indices/sp_esg/README.md) | Exclusions + float-cap rescaling | S&P DJI | ✅ Implemented |
| [S&P Global Carbon Efficient Index Series](indices/sp_carbon_efficient/README.md) | Carbon efficiency tilt (no exclusions) | S&P DJI | ✅ Implemented |
| [Dow Jones Sustainability Diversified Indices](indices/djsi_diversified/README.md) | Best-in-class ESG selection + weight cap | S&P DJI | ✅ Implemented |
| [S&P PACT Indices (CTB/PAB)](indices/sp_pact/README.md) | Exclusions + cvxpy optimizer (minimize tracking error) | S&P DJI | ✅ Implemented |

---

## Setup

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone <repo-url>
cd hygenic_indices

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

---

## Usage

### Run a rebalance

```bash
# S&P Carbon Aware — developed variant (default)
python cli.py sp-carbon-aware rebalance \
    --input indices/sp_carbon_aware/sample_data/universe.csv

# S&P ESG
python cli.py sp-esg rebalance \
    --input indices/sp_esg/sample_data/universe.csv

# S&P Carbon Efficient
python cli.py sp-carbon-efficient rebalance \
    --input indices/sp_carbon_efficient/sample_data/universe.csv

# DJSI Diversified
python cli.py djsi-diversified rebalance \
    --input indices/djsi_diversified/sample_data/universe.csv

# S&P PACT — Paris-Aligned Benchmark (PAB)
python cli.py sp-pact rebalance \
    --input indices/sp_pact/sample_data/universe.csv \
    --variant pab

# S&P PACT — Climate Transition Benchmark (CTB)
python cli.py sp-pact rebalance \
    --input indices/sp_pact/sample_data/universe.csv \
    --variant ctb

# Export results to CSV
python cli.py sp-carbon-aware rebalance \
    --input indices/sp_carbon_aware/sample_data/universe.csv \
    --output rebalanced.csv

# Inspect the raw universe and underlying weights
python cli.py sp-carbon-aware show-universe \
    --input indices/sp_carbon_aware/sample_data/universe.csv
```

### Download methodology PDFs

```bash
# List available methodologies
python cli.py download list

# Download all methodology PDFs
python cli.py download fetch --all

# Download a specific methodology
python cli.py download fetch --index sp-pact
```

### Run all tests

```bash
python -m pytest tests/ -v
```

### Run tests with coverage

```bash
python -m pytest tests/ --cov=indices --cov-report=term-missing
```

---

## Project Structure

```
hygenic_indices/
├── cli.py                              # Click CLI entry point (all index commands)
├── pyproject.toml                      # Package metadata and dependencies
├── methodologies/                      # Source methodology PDFs from S&P DJI
│   ├── downloader.py                   # Methodology PDF catalog + download logic
├── indices/
│   ├── sp_carbon_aware/                # S&P Carbon Aware Index Series
│   │   ├── models.py                   #   Data models (Stock, Universe, etc.)
│   │   ├── eligibility.py             #   Exclusion filter functions
│   │   ├── optimization.py            #   cvxpy optimizer (minimize WACI)
│   │   ├── rebalancer.py             #   Full pipeline orchestrator
│   │   ├── sample_data/              #   56-stock synthetic universe
│   │   └── README.md
│   ├── sp_esg/                         # S&P ESG Index Series
│   │   ├── models.py                   #   Data models
│   │   ├── eligibility.py             #   ESG + business activity exclusions
│   │   ├── rebalancer.py             #   Float-cap rescaling pipeline
│   │   ├── sample_data/              #   40-stock synthetic universe
│   │   └── README.md
│   ├── sp_carbon_efficient/            # S&P Global Carbon Efficient Index Series
│   │   ├── models.py                   #   Data models
│   │   ├── weighting.py              #   Carbon efficiency tilt (CEF)
│   │   ├── rebalancer.py             #   Tilt pipeline orchestrator
│   │   ├── sample_data/              #   45-stock synthetic universe
│   │   └── README.md
│   ├── djsi_diversified/               # Dow Jones Sustainability Diversified Indices
│   │   ├── models.py                   #   Data models
│   │   ├── eligibility.py             #   Hard exclusions + best-in-class selection
│   │   ├── rebalancer.py             #   Selection + weight cap pipeline
│   │   ├── sample_data/              #   118-stock synthetic universe
│   │   └── README.md
│   └── sp_pact/                        # S&P PACT Indices (CTB / PAB)
│       ├── models.py                   #   Data models (CTB/PAB variants)
│       ├── eligibility.py             #   Variant-specific exclusion filters
│       ├── optimization.py            #   cvxpy optimizer (minimize tracking error)
│       ├── rebalancer.py             #   Full pipeline orchestrator
│       ├── sample_data/              #   60-stock synthetic universe
│       └── README.md
└── tests/                              # 247 pytest tests across all indices
    ├── sp_carbon_aware/                #   100 tests
    ├── sp_esg/                         #   44 tests
    ├── sp_carbon_efficient/            #   20 tests
    ├── djsi_diversified/               #   46 tests
    └── sp_pact/                        #   37 tests
```

---

## Index Comparison

| Feature | Carbon Aware | ESG | Carbon Efficient | DJSI Diversified | PACT (CTB/PAB) |
|---|---|---|---|---|---|
| Exclusions | ESG + business activity | ESG + business activity | None | ESG + business activity | Variant-specific |
| Weighting | cvxpy optimizer | Float-cap rescaling | Carbon efficiency tilt | Float-cap + 10% cap | cvxpy optimizer |
| Objective | Minimize WACI | N/A | Reduce WACI via tilt | Best-in-class ESG | Minimize tracking error |
| Key constraint | Active share, diversification | Bottom 25% ESG removed | Industry-group neutral | 50% FMC target per group | WACI reduction (30%/50%) |
| Solver | CLARABEL | None | None | Iterative capping | CLARABEL |

---

## Adding a New Index

1. Create `indices/<index_name>/` with the same module structure (`models.py`, `eligibility.py`, `optimization.py` or `weighting.py`, `rebalancer.py`)
2. Add the methodology PDF to `methodologies/` and register it in `methodologies/downloader.py`
3. Generate a sample universe CSV in `indices/<index_name>/sample_data/`
4. Add a `tests/<index_name>/` suite
5. Register a new Click sub-group in `cli.py`
6. Write a `README.md` citing the relevant methodology sections
