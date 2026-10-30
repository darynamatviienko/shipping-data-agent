"""Shared LangGraph state for the shipping data agent."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

IntentName = Literal["invoice_check", "sql_qa", "unanswerable"]
MAX_SQL_RETRIES = 2


class AgentState(TypedDict, total=False):
    """Graph state passed between nodes."""

    question: str
    intent: IntentName
    intent_rationale: str
    schema_text: str
    sql: str
    sql_error: str | None
    retry_count: int
    query_result: dict[str, Any] | None
    query_result_text: str
    discrepancy_text: str
    answer: str
    answered: bool
