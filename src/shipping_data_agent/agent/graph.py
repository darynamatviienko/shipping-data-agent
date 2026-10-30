"""Compile the shipping LangGraph: understand → schema/SQL or discrepancy → answer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from shipping_data_agent.agent.nodes import (
    make_nodes,
    route_after_execute,
    route_after_understand,
    route_after_validate,
    route_retry,
)
from shipping_data_agent.agent.state import AgentState
from shipping_data_agent.tools.warehouse_tools import (
    DEFAULT_DB_PATH,
    DEFAULT_ROW_LIMIT,
    DEFAULT_TIMEOUT_MS,
)


def build_agent_graph(
    *,
    db_path: Path | None = None,
    llm: BaseChatModel | None = None,
    llm_factory: Callable[[], BaseChatModel] | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
):
    """Build and compile the explicit shipping agent StateGraph."""
    nodes = make_nodes(
        db_path=db_path,
        llm=llm,
        llm_factory=llm_factory,
        row_limit=row_limit,
        timeout_ms=timeout_ms,
    )

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("understand_question", nodes["understand_question"])
    graph.add_node("get_schema", nodes["get_schema"])
    graph.add_node("write_sql", nodes["write_sql"])
    graph.add_node("validate_sql", nodes["validate_sql"])
    graph.add_node("execute_sql", nodes["execute_sql"])
    graph.add_node("bump_retry", nodes["bump_retry"])
    graph.add_node("run_discrepancy", nodes["run_discrepancy"])
    graph.add_node("refuse", nodes["refuse"])
    graph.add_node("summarise", nodes["summarise"])

    graph.add_edge(START, "understand_question")
    graph.add_conditional_edges(
        "understand_question",
        route_after_understand,
        {
            "run_discrepancy": "run_discrepancy",
            "get_schema": "get_schema",
            "refuse": "refuse",
        },
    )
    graph.add_edge("get_schema", "write_sql")
    graph.add_edge("write_sql", "validate_sql")
    graph.add_conditional_edges(
        "validate_sql",
        route_after_validate,
        {
            "execute_sql": "execute_sql",
            "maybe_retry": "maybe_retry_gate",
            "summarise": "summarise",
        },
    )
    # Named gate node keeps the diagram readable for the retry decision.
    graph.add_node("maybe_retry_gate", lambda state: {})
    graph.add_conditional_edges(
        "maybe_retry_gate",
        route_retry,
        {
            "bump_retry": "bump_retry",
            "summarise": "summarise",
        },
    )
    graph.add_edge("bump_retry", "write_sql")
    graph.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "summarise": "summarise",
            "maybe_retry": "maybe_retry_gate",
        },
    )
    graph.add_edge("run_discrepancy", "summarise")
    graph.add_edge("refuse", END)
    graph.add_edge("summarise", END)

    return graph.compile(name="shipping_data_agent")


def export_graph_mermaid(**kwargs: Any) -> str:
    """Return a Mermaid flowchart for the compiled agent graph."""
    app = build_agent_graph(**kwargs)
    return app.get_graph().draw_mermaid()


def ask(
    question: str,
    *,
    db_path: Path | None = None,
    llm: BaseChatModel | None = None,
    llm_factory: Callable[[], BaseChatModel] | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> str:
    """Run one question through the agent graph and return the final answer text."""
    app = build_agent_graph(
        db_path=db_path if db_path is not None else DEFAULT_DB_PATH,
        llm=llm,
        llm_factory=llm_factory,
        row_limit=row_limit,
        timeout_ms=timeout_ms,
    )
    final: AgentState = app.invoke({"question": question})
    return str(final.get("answer") or "").strip()
