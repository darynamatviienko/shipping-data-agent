# Shipping Data Agent

AI agent for shipment Q&A and invoice error checks.

## Layout

```
src/shipping_data_agent/   # Python package
  agent/                   # LangGraph graph (Step 5)
  tools/                   # Read-only SQL / schema / discrepancy tools
  etl/                     # Cleaning + DuckDB load
scripts/                   # Data generation, ETL, ask_agent
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

Copy `.env.example` → `.env` and set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` (plus optional `LLM_PROVIDER=anthropic|openai`). Keys are never hardcoded.

### Ask a sample question (Step 5)

```bash
# Invoice discrepancy route
make ask Q='Find invoice discrepancies and summarise the error types'

# SQL analytics route
.venv/bin/python scripts/ask_agent.py "How many shipments were delivered to Germany?"

# Print the compiled graph as Mermaid
.venv/bin/python scripts/ask_agent.py --print-diagram
```

Or from Python:

```python
from shipping_data_agent.agent import ask
print(ask("What is the average delivery days by carrier?"))
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

### Agent tools (`src/shipping_data_agent/tools/`)

| Tool | Role |
|------|------|
| `run_sql(query)` | Single read-only `SELECT` / `WITH … SELECT`; rejects writes & multi-statements; row limit + timeout |
| `get_schema()` | Table/column descriptions for the star schema |
| `find_invoice_discrepancies()` | Curated SQL for the four injected invoice faults; returns a summary table |

```bash
.venv/bin/python -c "from shipping_data_agent.tools import find_invoice_discrepancies; print(find_invoice_discrepancies())"
```

### LangGraph agent (`src/shipping_data_agent/agent/`)

Explicit nodes:

1. **understand_question** — classify as `invoice_check`, `sql_qa`, or `unanswerable`
2. **run_discrepancy** — invoice audits call `find_invoice_discrepancies`
3. **get_schema → write_sql → validate_sql → execute_sql** — analytics path using Step 4 tools
4. **maybe_retry_gate / bump_retry** — on SQL guard or execution errors, retry writing SQL up to **2** times with the error message
5. **summarise** — plain-English answer; SQL analytics answers always show the SQL used
6. **refuse** — if the question cannot be answered from the warehouse, say so (no guessing)

Compiled graph (from `export_graph_mermaid()` / `scripts/ask_agent.py --print-diagram`):

```mermaid
flowchart TD
  start([start]) --> understand_question
  understand_question -->|sql_qa| get_schema
  understand_question -->|invoice_check| run_discrepancy
  understand_question -->|unanswerable| refuse
  get_schema --> write_sql --> validate_sql
  validate_sql -->|ok| execute_sql
  validate_sql -->|unsupported| summarise
  validate_sql -->|error| maybe_retry_gate
  execute_sql -->|ok| summarise
  execute_sql -->|error| maybe_retry_gate
  maybe_retry_gate -->|retries remaining| bump_retry --> write_sql
  maybe_retry_gate -->|retries exhausted| summarise
  run_discrepancy --> summarise
  refuse --> fin([end])
  summarise --> fin
```

Steps 6–7 (evaluation harness, Streamlit UI) come later.
