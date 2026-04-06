# hygenic_indices

Python implementations of S&P Global index rebalancing methodologies. Each index is a self-contained package that applies the published eligibility rules and optimization logic from the corresponding S&P Dow Jones Indices methodology document.

The goal is to make the business logic of these indices transparent, testable, and reproducible — without requiring access to proprietary data vendors. Sample universes are provided as CSVs so the full pipeline can be exercised out of the box.

---

## Indices

| Index | Methodology | Status |
|---|---|---|
| [S&P Carbon Aware Index Series](indices/sp_carbon_aware/README.md) | March 2026 | ✅ Implemented |

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

# Emerging variant, export results to CSV
python cli.py sp-carbon-aware rebalance \
    --input indices/sp_carbon_aware/sample_data/universe.csv \
    --universe-type emerging \
    --output rebalanced.csv

# Inspect the raw universe and underlying weights
python cli.py sp-carbon-aware show-universe \
    --input indices/sp_carbon_aware/sample_data/universe.csv
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
├── cli.py                          # Click CLI entry point
├── pyproject.toml                  # Package metadata and dependencies
├── methodologies/                  # Source methodology PDFs from S&P DJI
├── indices/
│   └── sp_carbon_aware/            # S&P Carbon Aware Index Series
│       ├── models.py               # Data models (Stock, Universe, etc.)
│       ├── eligibility.py          # Exclusion filter functions
│       ├── optimization.py         # cvxpy-based optimizer
│       ├── rebalancer.py           # Full pipeline orchestrator
│       ├── sample_data/            # Synthetic universe CSVs
│       └── README.md               # Index-specific documentation
└── tests/
    └── sp_carbon_aware/            # pytest test suite
```

---

## Adding a New Index

1. Create `indices/<index_name>/` with the same module structure (`models.py`, `eligibility.py`, `optimization.py`, `rebalancer.py`)
2. Add the methodology PDF to `methodologies/`
3. Generate a sample universe CSV in `indices/<index_name>/sample_data/`
4. Add a `tests/<index_name>/` suite
5. Register a new Click sub-group in `cli.py`
6. Write a `README.md` citing the relevant methodology sections
