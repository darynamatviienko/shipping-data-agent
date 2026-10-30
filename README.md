# Shipping Data Agent

AI agent for shipment Q&A and invoice error checks.

## Layout

```
src/shipping_data_agent/   # Python package
data/raw/                  # Source CSVs (Step 2)
data/warehouse/            # DuckDB warehouse (Step 3)
tests/                     # Pytest suite
```

## Quick start

```bash
python3 -m venv .venv
make setup
make test
```

Steps 2–7 (synthetic data, ETL, agent tools, LangGraph graph, evaluation, CLI/UI) come later.
