"""Pytest coverage for Step 4 agent tool guards and warehouse tools."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from shipping_data_agent.tools import (
    DEFAULT_DB_PATH,
    SqlGuardError,
    SqlTimeoutError,
    build_langchain_tools,
    find_invoice_discrepancies,
    get_schema,
    run_sql,
    validate_readonly_select,
)


WAREHOUSE = DEFAULT_DB_PATH


pytestmark = pytest.mark.skipif(
    not WAREHOUSE.exists(),
    reason="DuckDB warehouse missing; run `make etl` first",
)


# --- SQL guards ---


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "select carrier_id from dim_carrier limit 5",
        "WITH c AS (SELECT * FROM dim_carrier) SELECT carrier_id FROM c",
        "SELECT COUNT(*) AS n FROM fact_invoices;  ",
        "SELECT REPLACE(carrier_id, 'D', 'X') FROM dim_carrier ORDER BY carrier_id DESC",
    ],
)
def test_validate_allows_single_select(query: str):
    cleaned = validate_readonly_select(query)
    assert cleaned.upper().lstrip().startswith(("SELECT", "WITH"))
    assert ";" not in cleaned


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "-- just a comment",
        "DELETE FROM fact_invoices",
        "INSERT INTO dim_carrier VALUES (99, 'X', 'X', 'DE', 'express')",
        "UPDATE fact_invoices SET billed_price = 0",
        "DROP TABLE fact_invoices",
        "SELECT 1; SELECT 2",
        "CREATE TABLE hack AS SELECT 1",
        "ATTACH 'other.duckdb'",
        "COPY fact_invoices TO 'out.csv'",
        "PRAGMA show_tables",
        "EXPLAIN SELECT 1",
        "SET max_execution_time = 1",
        "SHOW TABLES",
    ],
)
def test_validate_rejects_non_readonly(query: str):
    with pytest.raises(SqlGuardError):
        validate_readonly_select(query)


def test_run_sql_rejects_write_attempt():
    with pytest.raises(SqlGuardError, match="Forbidden|read-only|SELECT"):
        run_sql("DELETE FROM fact_invoices WHERE 1=1")


def test_run_sql_select_and_row_limit():
    result = run_sql(
        "SELECT invoice_id FROM fact_invoices ORDER BY invoice_id",
        row_limit=10,
    )
    assert result["row_count"] == 10
    assert result["truncated"] is True
    assert result["columns"] == ["invoice_id"]
    assert len(result["rows"]) == 10


def test_run_sql_not_truncated_when_under_limit():
    result = run_sql("SELECT carrier_id FROM dim_carrier ORDER BY carrier_id")
    assert result["row_count"] == 5
    assert result["truncated"] is False


def test_run_sql_timeout():
    with pytest.raises(SqlTimeoutError):
        run_sql(
            """
            WITH RECURSIVE t(n) AS (
              SELECT 1
              UNION ALL
              SELECT n + 1 FROM t WHERE n < 100000000
            )
            SELECT count(*) AS n FROM t
            """,
            timeout_ms=50,
            row_limit=1,
        )


def test_get_schema_lists_star_tables():
    schema = get_schema()
    names = {t["name"] for t in schema["tables"]}
    assert names == {
        "dim_carrier",
        "dim_country",
        "dim_date",
        "fact_shipments",
        "fact_invoices",
        "fact_contract_rates",
    }
    invoices = next(t for t in schema["tables"] if t["name"] == "fact_invoices")
    col_names = {c["name"] for c in invoices["columns"]}
    assert {"invoice_id", "shipment_id", "billed_price"} <= col_names
    assert "description" in invoices and invoices["description"]


def test_find_invoice_discrepancies_summary():
    summary = find_invoice_discrepancies()
    assert list(summary.columns) == [
        "error_type",
        "discrepancy_count",
        "total_billed_eur",
        "avg_billed_eur",
    ]
    counts = dict(
        zip(summary["error_type"], summary["discrepancy_count"], strict=True)
    )
    # Seed-42 synthetic volumes (see Step 2 report).
    assert counts["duplicate_billing"] == 292
    assert counts["nonexistent_shipment"] == 145
    assert counts["wrong_weight_class"] == 146
    assert counts["price_above_contract"] == 145
    assert int(summary["discrepancy_count"].sum()) == 728


def test_discrepancies_match_labeled_file():
    """Classifier labels should match carrier_invoices_labeled.csv exactly."""
    from shipping_data_agent.tools.warehouse_tools import (
        DEFAULT_TIMEOUT_MS,
        DISCREPANCY_DETAIL_SQL,
        _connect,
    )

    labels = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "carrier_invoices_labeled.csv"
    )
    labels["error_type"] = labels["error_type"].fillna("")

    con = _connect(WAREHOUSE, timeout_ms=DEFAULT_TIMEOUT_MS)
    try:
        pred = con.execute(DISCREPANCY_DETAIL_SQL).fetchdf()[
            ["invoice_id", "error_type"]
        ]
    finally:
        con.close()

    merged = labels.merge(pred, on="invoice_id", how="left", suffixes=("_true", "_pred"))
    merged["error_type_pred"] = merged["error_type_pred"].fillna("")
    assert (merged["error_type_true"] == merged["error_type_pred"]).all()


def test_build_langchain_tools_smoke():
    tools = build_langchain_tools()
    by_name = {t.name: t for t in tools}
    assert set(by_name) == {"run_sql", "get_schema", "find_invoice_discrepancies"}

    schema_text = by_name["get_schema"].invoke({})
    assert "fact_invoices" in schema_text

    blocked = by_name["run_sql"].invoke({"query": "DROP TABLE fact_invoices"})
    assert blocked.startswith("ERROR:")

    ok = by_name["run_sql"].invoke({"query": "SELECT COUNT(*) AS n FROM dim_carrier"})
    assert "row_count=1" in ok

    summary = by_name["find_invoice_discrepancies"].invoke({})
    assert "duplicate_billing" in summary
    assert "wrong_weight_class" in summary


def test_read_only_connection_blocks_ddl(tmp_path: Path):
    """Defense in depth: even if a guard slipped, connection is read_only."""
    con = duckdb.connect(str(WAREHOUSE), read_only=True)
    try:
        with pytest.raises(duckdb.Error):
            con.execute("CREATE TABLE should_fail AS SELECT 1")
    finally:
        con.close()
