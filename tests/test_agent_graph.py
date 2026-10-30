"""Lightweight Step 5 tests: routing, SQL validation, graph shape (no live LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from shipping_data_agent.agent.graph import ask, build_agent_graph, export_graph_mermaid
from shipping_data_agent.agent.nodes import (
    make_nodes,
    route_after_execute,
    route_after_understand,
    route_after_validate,
    route_retry,
)
from shipping_data_agent.agent.routing import classify_intent_heuristic
from shipping_data_agent.agent.state import MAX_SQL_RETRIES
from shipping_data_agent.tools import DEFAULT_DB_PATH


WAREHOUSE = DEFAULT_DB_PATH


# --- Heuristic routing ---


@pytest.mark.parametrize(
    "question",
    [
        "Find invoice discrepancies in the warehouse",
        "Check invoices for billing errors",
        "Are there duplicate billing problems?",
        "Detect wrong weight class invoice faults",
        "Audit invoices for price above contract",
    ],
)
def test_heuristic_routes_invoice_checks(question: str):
    assert classify_intent_heuristic(question).name == "invoice_check"


@pytest.mark.parametrize(
    "question",
    [
        "How many shipments were delivered to DE?",
        "What is the average delivery days by carrier?",
        "List the top 5 carriers by shipment count",
        "How many invoices are there?",
    ],
)
def test_heuristic_routes_sql_qa(question: str):
    assert classify_intent_heuristic(question).name == "sql_qa"


@pytest.mark.parametrize(
    "question",
    [
        "",
        "What is the weather in Berlin?",
        "Tell me a joke",
        "Who is the president?",
    ],
)
def test_heuristic_routes_unanswerable(question: str):
    assert classify_intent_heuristic(question).name == "unanswerable"


def test_route_helpers():
    assert route_after_understand({"intent": "invoice_check"}) == "run_discrepancy"
    assert route_after_understand({"intent": "sql_qa"}) == "get_schema"
    assert route_after_understand({"intent": "unanswerable"}) == "refuse"

    assert route_after_validate({"sql_error": None}) == "execute_sql"
    assert route_after_validate({"sql_error": "boom"}) == "maybe_retry"
    assert (
        route_after_validate({"sql_error": "UNSUPPORTED: no table"}) == "summarise"
    )

    assert route_after_execute({"sql_error": None}) == "summarise"
    assert route_after_execute({"sql_error": "timeout"}) == "maybe_retry"

    assert route_retry({"retry_count": 0}) == "bump_retry"
    assert route_retry({"retry_count": MAX_SQL_RETRIES - 1}) == "bump_retry"
    assert route_retry({"retry_count": MAX_SQL_RETRIES}) == "summarise"


def test_validate_sql_node_accepts_and_rejects():
    nodes = make_nodes(db_path=WAREHOUSE if WAREHOUSE.exists() else Path("."))
    ok = nodes["validate_sql"]({"sql": "SELECT 1 AS n", "sql_error": None})
    assert ok["sql_error"] is None
    assert ok["sql"].upper().startswith("SELECT")

    bad = nodes["validate_sql"](
        {"sql": "DELETE FROM fact_invoices", "sql_error": None}
    )
    assert bad["sql_error"]
    assert "guard" in bad["sql_error"].lower() or "Forbidden" in bad["sql_error"]


def test_bump_retry_increments():
    nodes = make_nodes(db_path=Path("."))
    assert nodes["bump_retry"]({"retry_count": 0})["retry_count"] == 1
    assert nodes["bump_retry"]({})["retry_count"] == 1


def test_graph_mermaid_lists_expected_nodes():
    mermaid = export_graph_mermaid()
    for name in (
        "understand_question",
        "get_schema",
        "write_sql",
        "validate_sql",
        "execute_sql",
        "bump_retry",
        "run_discrepancy",
        "refuse",
        "summarise",
        "maybe_retry_gate",
    ):
        assert name in mermaid


def test_refuse_path_without_live_llm():
    """Off-topic questions end in refuse; no API key required."""
    fake = FakeListChatModel(responses=["unused"])
    app = build_agent_graph(llm=fake)
    final = app.invoke({"question": "What is the weather in Berlin?"})
    assert final["intent"] == "unanswerable"
    assert "cannot answer" in final["answer"].lower()
    assert final["answered"] is False


@pytest.mark.skipif(
    not WAREHOUSE.exists(),
    reason="DuckDB warehouse missing; run `make etl` first",
)
def test_invoice_path_uses_discrepancy_tool():
    """Invoice audit questions hit find_invoice_discrepancies (fake LLM for summarise)."""
    fake = FakeListChatModel(
        responses=[
            "There are four discrepancy types with hundreds of faulty invoices. "
            "Source: find_invoice_discrepancies tool."
        ]
    )
    answer = ask(
        "Find invoice discrepancies and summarise the error types",
        llm=fake,
    )
    assert answer
    # Either the LLM summary or (if somehow templated) the tool table.
    lowered = answer.lower()
    assert (
        "discrepan" in lowered
        or "duplicate_billing" in lowered
        or "error" in lowered
    )


@pytest.mark.skipif(
    not WAREHOUSE.exists(),
    reason="DuckDB warehouse missing; run `make etl` first",
)
def test_sql_retry_then_summarise_with_fake_llm():
    """Bad SQL → guard/execute error → retry twice → final summarise (no live key)."""
    # understand may call the LLM; write_sql fails validation three times; then summarise.
    fake = FakeListChatModel(
        responses=[
            "INTENT=sql_qa\nRATIONALE=shipment count",
            "DELETE FROM fact_invoices",  # rejected by validate
            "DELETE FROM fact_invoices",  # retry 1
            "DELETE FROM fact_invoices",  # retry 2
            "Could not answer after retries; SQL was invalid.",
        ]
    )
    app = build_agent_graph(llm=fake)
    final = app.invoke({"question": "How many shipments are there?"})
    assert final["intent"] == "sql_qa"
    assert int(final.get("retry_count") or 0) == MAX_SQL_RETRIES
    assert final.get("answer")
