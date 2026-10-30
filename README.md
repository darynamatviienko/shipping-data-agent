# Shipping Data Agent

AI agent for shipment Q&A and invoice error checks.

## Layout

```
src/shipping_data_agent/   # Python package
scripts/                   # Data generation and utilities
data/raw/                  # Source CSVs (Step 2)
data/warehouse/            # DuckDB warehouse (Step 3)
tests/                     # Pytest suite
```

## Quick start

```bash
python3 -m venv .venv
make setup
make data    # regenerate synthetic CSVs under data/raw/ (seed=42)
make etl     # clean + load DuckDB warehouse under data/warehouse/
make test
```

Or run the generator / ETL directly:

```bash
.venv/bin/python scripts/generate_synthetic_data.py
.venv/bin/python scripts/run_etl.py
```

### Raw CSVs (`data/raw/`)

| File | Role |
|------|------|
| `shipments.csv` | ~20k shipments (`id`, `date`, `carrier`, `country`, `weight_kg`, `parcel_type`, `status`, `delivery_days`) |
| `carriers.csv` | Carrier master data |
| `contract_rates.csv` | Contract rates by carrier / country / weight band |
| `carrier_invoices.csv` | Carrier bills (`shipment_id`, `carrier`, `billed_price`, `invoice_date`) |
| `carrier_invoices_labeled.csv` | Same invoices plus `error_type` labels (for later eval) |

Roughly 3% of invoices carry deliberate faults: duplicate billing, wrong weight class, price above contract, or a billed shipment that does not exist. ETL keeps those rows so later agent tools can discover them; only structural bad data is rejected.

### Warehouse (`data/warehouse/`)

| Artifact | Role |
|----------|------|
| `shipping.duckdb` | Star schema: `fact_shipments`, `fact_invoices`, `fact_contract_rates`, `dim_carrier`, `dim_country`, `dim_date` |
| `rejected_rows.csv` | Rows dropped during cleaning (`_source`, `_reason`, `_row_number` + original columns) |

`make etl` is idempotent (tables are replaced each run).

Steps 4–7 (agent tools, LangGraph graph, evaluation, CLI/UI) come later.
