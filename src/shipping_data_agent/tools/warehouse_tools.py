"""Agent tools backed by the DuckDB shipping warehouse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from langchain_core.tools import StructuredTool

from shipping_data_agent.etl.pipeline import DEFAULT_DB_PATH as ETL_DEFAULT_DB_PATH
from shipping_data_agent.tools.guards import SqlGuardError, validate_readonly_select

DEFAULT_DB_PATH = ETL_DEFAULT_DB_PATH
DEFAULT_ROW_LIMIT = 200
DEFAULT_TIMEOUT_MS = 5_000

# Country multipliers used when synthetic contract rates were generated.
# Needed to reconstruct expected invoice price from mid-band published rates.
_COUNTRY_MULT_SQL = """
CASE dc.country_code
  WHEN 'DE' THEN 1.00 WHEN 'NL' THEN 1.02 WHEN 'BE' THEN 1.05
  WHEN 'FR' THEN 1.08 WHEN 'PL' THEN 0.95 WHEN 'CZ' THEN 0.93
  WHEN 'IT' THEN 1.12 WHEN 'ES' THEN 1.10 WHEN 'GB' THEN 1.18
  WHEN 'US' THEN 1.35 ELSE 1.00
END
"""

# Shared CTE chain that classifies each invoice into at most one injected fault.
DISCREPANCY_CLASSIFIED_SQL = f"""
WITH invoice_enriched AS (
  SELECT
    i.invoice_id,
    i.shipment_id,
    i.billed_price,
    s.shipment_id AS matched_shipment,
    s.weight_kg,
    s.carrier_sk,
    s.country_sk,
    r.rate_eur,
    r.weight_band AS expected_band,
    {_COUNTRY_MULT_SQL} AS country_mult,
    CASE
      WHEN r.max_kg IS NULL THEN r.min_kg + 10.0
      ELSE (r.min_kg + r.max_kg) / 2.0
    END AS ref_kg
  FROM fact_invoices i
  LEFT JOIN fact_shipments s ON i.shipment_id = s.shipment_id
  LEFT JOIN dim_country dc ON s.country_sk = dc.country_sk
  LEFT JOIN fact_contract_rates r
    ON r.carrier_sk = s.carrier_sk
   AND r.country_sk = s.country_sk
   AND s.weight_kg >= r.min_kg
   AND (r.max_kg IS NULL OR s.weight_kg < r.max_kg)
),
with_expected AS (
  SELECT
    *,
    CASE
      WHEN rate_eur IS NOT NULL THEN
        rate_eur + (weight_kg - ref_kg) * 0.35 * country_mult
    END AS expected_rate
  FROM invoice_enriched
),
dupes AS (
  SELECT shipment_id
  FROM fact_invoices
  GROUP BY shipment_id
  HAVING COUNT(*) > 1
),
other_bands AS (
  SELECT
    e.invoice_id,
    MIN(ABS(e.billed_price - o.rate_eur)) AS min_other_diff
  FROM with_expected e
  JOIN fact_contract_rates o
    ON o.carrier_sk = e.carrier_sk
   AND o.country_sk = e.country_sk
   AND o.weight_band <> e.expected_band
  WHERE e.matched_shipment IS NOT NULL
  GROUP BY e.invoice_id
),
classified AS (
  SELECT
    e.invoice_id,
    e.shipment_id,
    e.billed_price,
    e.expected_rate,
    CASE
      WHEN e.matched_shipment IS NULL THEN 'nonexistent_shipment'
      WHEN d.shipment_id IS NOT NULL THEN 'duplicate_billing'
      WHEN e.expected_rate IS NOT NULL
           AND ABS(e.billed_price - e.expected_rate)
               / NULLIF(e.expected_rate, 0) > 0.02
           AND ob.min_other_diff IS NOT NULL
           AND ob.min_other_diff <= 0.01
        THEN 'wrong_weight_class'
      WHEN e.expected_rate IS NOT NULL
           AND e.billed_price > e.expected_rate * 1.10
        THEN 'price_above_contract'
      ELSE NULL
    END AS error_type
  FROM with_expected e
  LEFT JOIN dupes d ON e.shipment_id = d.shipment_id
  LEFT JOIN other_bands ob ON e.invoice_id = ob.invoice_id
)
"""

DISCREPANCY_SUMMARY_SQL = (
    DISCREPANCY_CLASSIFIED_SQL
    + """
SELECT
  error_type,
  COUNT(*) AS discrepancy_count,
  ROUND(SUM(billed_price), 2) AS total_billed_eur,
  ROUND(AVG(billed_price), 2) AS avg_billed_eur
FROM classified
WHERE error_type IS NOT NULL
GROUP BY error_type
ORDER BY error_type
"""
)

DISCREPANCY_DETAIL_SQL = (
    DISCREPANCY_CLASSIFIED_SQL
    + """
SELECT invoice_id, shipment_id, billed_price, expected_rate, error_type
FROM classified
WHERE error_type IS NOT NULL
ORDER BY error_type, invoice_id
"""
)

TABLE_DESCRIPTIONS: dict[str, str] = {
    "dim_carrier": (
        "Carrier dimension: one row per carrier (DHL, UPS, FDX, TNT, GLS). "
        "Join to facts on carrier_sk."
    ),
    "dim_country": (
        "Country dimension for destination / HQ countries. "
        "Join on country_sk; country_code is ISO-2."
    ),
    "dim_date": (
        "Calendar dimension derived from shipment, invoice, and rate dates. "
        "Join on date_sk (YYYYMMDD integer)."
    ),
    "fact_shipments": (
        "One row per shipment. Measures: weight_kg, delivery_days. "
        "Attributes: parcel_type, status (includes cancelled)."
    ),
    "fact_invoices": (
        "One row per carrier invoice line. Grain is invoice_id. "
        "May include orphan shipment_ids and duplicate bills for the same shipment "
        "(intentional business faults for discrepancy detection)."
    ),
    "fact_contract_rates": (
        "Contract rate lines by carrier × country × weight_band. "
        "rate_eur is the mid-band published EUR rate; use min_kg/max_kg to match "
        "a shipment weight. max_kg is NULL for the open-ended 50+ band."
    ),
}

COLUMN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "dim_carrier": {
        "carrier_sk": "Surrogate key",
        "carrier_id": "Natural carrier code (e.g. DHL)",
        "name": "Carrier display name",
        "hq_country": "Headquarters country code",
        "service_tier": "express | standard | economy",
    },
    "dim_country": {
        "country_sk": "Surrogate key",
        "country_code": "ISO-2 country code",
        "country_name": "Country display name",
    },
    "dim_date": {
        "date_sk": "Surrogate key YYYYMMDD",
        "date": "Calendar date",
        "year": "Year",
        "month": "Month 1-12",
        "day": "Day of month",
        "quarter": "Quarter 1-4",
        "day_of_week": "Monday=0 .. Sunday=6",
        "iso_week": "ISO week number",
    },
    "fact_shipments": {
        "shipment_id": "Natural shipment id (SHP-######)",
        "date_sk": "FK → dim_date (ship date)",
        "carrier_sk": "FK → dim_carrier",
        "country_sk": "FK → dim_country (destination)",
        "weight_kg": "Parcel weight in kilograms",
        "parcel_type": "envelope | small_box | medium_box | large_box | pallet",
        "status": "delivered | in_transit | delayed | returned | cancelled",
        "delivery_days": "Days to deliver (0 if cancelled)",
    },
    "fact_invoices": {
        "invoice_id": "Natural invoice id (INV-######)",
        "shipment_id": "Claimed shipment id (may not exist)",
        "carrier_sk": "FK → dim_carrier (billing carrier)",
        "invoice_date_sk": "FK → dim_date",
        "billed_price": "Amount billed by the carrier (EUR)",
    },
    "fact_contract_rates": {
        "contract_rate_sk": "Surrogate key",
        "carrier_sk": "FK → dim_carrier",
        "country_sk": "FK → dim_country",
        "weight_band": "Band label (0-1, 1-5, 5-20, 20-50, 50+)",
        "min_kg": "Inclusive lower bound of the band",
        "max_kg": "Exclusive upper bound; NULL for 50+",
        "rate_eur": "Published mid-band contract rate (EUR)",
        "currency": "Currency code (EUR)",
        "effective_from": "Rate effective date",
    },
}


class SqlTimeoutError(TimeoutError):
    """Raised when a warehouse query exceeds the configured timeout."""


def _connect(db_path: Path, *, timeout_ms: int) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Warehouse not found: {path}. Run `make etl` first."
        )
    con = duckdb.connect(str(path), read_only=True)
    con.execute(f"SET max_execution_time = {int(timeout_ms)}")
    return con


def run_sql(
    query: str,
    *,
    db_path: Path | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Any]:
    """Execute a single read-only SELECT against the warehouse.

    Guards: reject non-SELECT / multi-statement SQL. Caps returned rows and
    enforces a DuckDB ``max_execution_time`` timeout.
    """
    if row_limit < 1:
        raise ValueError("row_limit must be >= 1")
    if timeout_ms < 1:
        raise ValueError("timeout_ms must be >= 1")

    cleaned = validate_readonly_select(query)
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    wrapped = (
        f"SELECT * FROM (\n{cleaned}\n) AS _agent_query\n"
        f"LIMIT {int(row_limit) + 1}"
    )

    con = _connect(path, timeout_ms=timeout_ms)
    try:
        try:
            frame = con.execute(wrapped).fetchdf()
        except duckdb.InterruptException as exc:
            raise SqlTimeoutError(
                f"Query exceeded timeout of {timeout_ms} ms"
            ) from exc
        except duckdb.Error as exc:
            # DuckDB may surface timeouts as generic errors depending on version.
            message = str(exc)
            if "maximum execution time" in message.lower() or "interrupt" in message.lower():
                raise SqlTimeoutError(
                    f"Query exceeded timeout of {timeout_ms} ms"
                ) from exc
            raise
    finally:
        con.close()

    truncated = len(frame) > row_limit
    if truncated:
        frame = frame.iloc[:row_limit].copy()

    # Convert timestamps/dates to ISO strings for JSON-friendly agent output.
    rows = []
    for record in frame.to_dict(orient="records"):
        rows.append(
            {
                key: (
                    value.isoformat()
                    if hasattr(value, "isoformat")
                    else (None if pd.isna(value) else value)
                )
                for key, value in record.items()
            }
        )

    return {
        "columns": list(frame.columns),
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
        "row_limit": row_limit,
    }


def get_schema(*, db_path: Path | None = None) -> dict[str, Any]:
    """Return table and column descriptions for the warehouse star schema."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    con = _connect(path, timeout_ms=DEFAULT_TIMEOUT_MS)
    try:
        live = con.execute(
            """
            SELECT table_name, column_name, data_type, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchdf()
        counts = {
            row[0]: int(row[1])
            for row in con.execute(
                """
                SELECT table_name, estimated_size
                FROM duckdb_tables()
                WHERE schema_name = 'main'
                """
            ).fetchall()
        }
    finally:
        con.close()

    tables: list[dict[str, Any]] = []
    for table_name, group in live.groupby("table_name", sort=True):
        columns = []
        col_docs = COLUMN_DESCRIPTIONS.get(str(table_name), {})
        for _, row in group.iterrows():
            col_name = str(row["column_name"])
            columns.append(
                {
                    "name": col_name,
                    "type": str(row["data_type"]),
                    "description": col_docs.get(col_name, ""),
                }
            )
        tables.append(
            {
                "name": str(table_name),
                "description": TABLE_DESCRIPTIONS.get(
                    str(table_name), "Warehouse table"
                ),
                "row_count_estimate": counts.get(str(table_name)),
                "columns": columns,
            }
        )

    return {
        "database": str(path),
        "tables": tables,
    }


def find_invoice_discrepancies(
    *,
    db_path: Path | None = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> pd.DataFrame:
    """Detect injected invoice error types with plain SQL; return a summary table.

    Error types (mutually exclusive, matching synthetic injection):
      - nonexistent_shipment: invoice.shipment_id missing from fact_shipments
      - duplicate_billing: more than one invoice for the same shipment_id
      - wrong_weight_class: billed_price matches a different weight band's
        published mid-band rate (and not the correct expected price)
      - price_above_contract: billed_price > 110% of reconstructed expected price
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    con = _connect(path, timeout_ms=timeout_ms)
    try:
        try:
            return con.execute(DISCREPANCY_SUMMARY_SQL).fetchdf()
        except duckdb.InterruptException as exc:
            raise SqlTimeoutError(
                f"Query exceeded timeout of {timeout_ms} ms"
            ) from exc
    finally:
        con.close()


def _format_schema_for_agent(schema: dict[str, Any]) -> str:
    lines = [f"DuckDB warehouse: {schema['database']}", ""]
    for table in schema["tables"]:
        est = table.get("row_count_estimate")
        est_s = f" (~{est} rows)" if est is not None else ""
        lines.append(f"## {table['name']}{est_s}")
        lines.append(table["description"])
        for col in table["columns"]:
            desc = f" — {col['description']}" if col["description"] else ""
            lines.append(f"- {col['name']} ({col['type']}){desc}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_sql_result(result: dict[str, Any]) -> str:
    if result["row_count"] == 0:
        body = "(no rows)"
    else:
        frame = pd.DataFrame(result["rows"], columns=result["columns"])
        body = frame.to_string(index=False)
    flag = " truncated=yes" if result["truncated"] else ""
    return (
        f"row_count={result['row_count']} row_limit={result['row_limit']}{flag}\n"
        f"{body}"
    )


def build_langchain_tools(
    *,
    db_path: Path | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> list[StructuredTool]:
    """Build LangChain tool wrappers (no agent graph — that is Step 5)."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH

    def _run_sql(query: str) -> str:
        try:
            return _format_sql_result(
                run_sql(
                    query,
                    db_path=path,
                    row_limit=row_limit,
                    timeout_ms=timeout_ms,
                )
            )
        except (SqlGuardError, SqlTimeoutError, duckdb.Error, FileNotFoundError) as exc:
            return f"ERROR: {exc}"

    def _get_schema() -> str:
        try:
            return _format_schema_for_agent(get_schema(db_path=path))
        except (FileNotFoundError, duckdb.Error) as exc:
            return f"ERROR: {exc}"

    def _find_invoice_discrepancies() -> str:
        try:
            frame = find_invoice_discrepancies(
                db_path=path, timeout_ms=timeout_ms
            )
            if frame.empty:
                return "No invoice discrepancies found."
            return frame.to_string(index=False)
        except (SqlTimeoutError, duckdb.Error, FileNotFoundError) as exc:
            return f"ERROR: {exc}"

    return [
        StructuredTool.from_function(
            func=_run_sql,
            name="run_sql",
            description=(
                "Run a single read-only SQL SELECT (WITH/CTE allowed) against the "
                f"shipping DuckDB warehouse. Results are capped at {row_limit} rows "
                f"and time out after {timeout_ms} ms. Do not use INSERT/UPDATE/DELETE "
                "or multiple statements."
            ),
        ),
        StructuredTool.from_function(
            func=_get_schema,
            name="get_schema",
            description=(
                "Return warehouse table and column descriptions (star schema: "
                "fact_shipments, fact_invoices, fact_contract_rates, dim_carrier, "
                "dim_country, dim_date)."
            ),
        ),
        StructuredTool.from_function(
            func=_find_invoice_discrepancies,
            name="find_invoice_discrepancies",
            description=(
                "Run curated SQL that flags invoice faults: duplicate_billing, "
                "wrong_weight_class, price_above_contract, nonexistent_shipment. "
                "Returns a summary table (error_type, counts, billed totals)."
            ),
        ),
    ]
