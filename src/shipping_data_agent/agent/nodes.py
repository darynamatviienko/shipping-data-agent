"""LangGraph node implementations for the shipping data agent."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from shipping_data_agent.agent.routing import classify_intent
from shipping_data_agent.agent.state import AgentState, MAX_SQL_RETRIES
from shipping_data_agent.tools.guards import SqlGuardError, validate_readonly_select
from shipping_data_agent.tools.warehouse_tools import (
    DEFAULT_DB_PATH,
    DEFAULT_ROW_LIMIT,
    DEFAULT_TIMEOUT_MS,
    SqlTimeoutError,
    _format_schema_for_agent,
    _format_sql_result,
    find_invoice_discrepancies,
    get_schema,
    run_sql,
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply (fenced or bare)."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Fall back to first SELECT/WITH block.
    bare = re.search(
        r"((?:WITH|SELECT)\b.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if bare:
        return bare.group(1).strip()
    return text.strip()


def make_nodes(
    *,
    db_path: Path | None = None,
    llm: BaseChatModel | None = None,
    llm_factory: Callable[[], BaseChatModel] | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> dict[str, Callable[[AgentState], dict[str, Any]]]:
    """Build node callables closed over warehouse + LLM settings."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH

    def _llm() -> BaseChatModel:
        if llm is not None:
            return llm
        if llm_factory is not None:
            return llm_factory()
        from shipping_data_agent.agent.llm import get_chat_model

        return get_chat_model()

    def understand_question(state: AgentState) -> dict[str, Any]:
        from shipping_data_agent.agent.llm import llm_configured

        model: BaseChatModel | None = None
        if llm is not None or llm_factory is not None:
            model = _llm()
        elif llm_configured():
            model = _llm()
        intent = classify_intent(state["question"], llm=model)
        return {
            "intent": intent.name,
            "intent_rationale": intent.rationale,
            "retry_count": 0,
            "sql_error": None,
            "answered": False,
        }

    def get_schema_node(state: AgentState) -> dict[str, Any]:
        schema = get_schema(db_path=path)
        return {"schema_text": _format_schema_for_agent(schema)}

    def write_sql(state: AgentState) -> dict[str, Any]:
        schema_text = state.get("schema_text") or ""
        prior_error = state.get("sql_error")
        prior_sql = state.get("sql") or ""
        system = (
            "You write a single read-only DuckDB SQL SELECT (WITH/CTE allowed) "
            "to answer shipping warehouse questions.\n"
            "Rules:\n"
            "- Output ONLY the SQL (optional ```sql fence). No commentary.\n"
            "- One statement only. No INSERT/UPDATE/DELETE/DDL/PRAGMA.\n"
            "- Use only tables/columns from the provided schema.\n"
            "- If the question cannot be answered from the schema, output exactly:\n"
            "  UNSUPPORTED: <short reason>\n"
            "- Prefer aggregates for 'how many / average / total' questions.\n"
            "- Keep result sets small; use LIMIT when listing rows.\n"
        )
        user_parts = [
            f"Question: {state['question']}",
            "",
            "Schema:",
            schema_text,
        ]
        if prior_error:
            user_parts.extend(
                [
                    "",
                    "Previous SQL failed. Fix it using the error.",
                    f"Previous SQL:\n{prior_sql}",
                    f"Error:\n{prior_error}",
                ]
            )
        response = _llm().invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content="\n".join(user_parts)),
            ]
        )
        raw = _content_to_text(response.content).strip()
        if raw.upper().startswith("UNSUPPORTED:"):
            return {
                "sql": "",
                "sql_error": raw,
                "answered": False,
            }
        return {"sql": _extract_sql(raw), "sql_error": None}

    def validate_sql(state: AgentState) -> dict[str, Any]:
        sql = (state.get("sql") or "").strip()
        prior = state.get("sql_error") or ""
        if prior.upper().startswith("UNSUPPORTED:"):
            return {"sql_error": prior}
        if not sql:
            return {"sql_error": "No SQL was produced"}
        try:
            cleaned = validate_readonly_select(sql)
            return {"sql": cleaned, "sql_error": None}
        except SqlGuardError as exc:
            return {"sql_error": f"SQL guard rejected query: {exc}"}

    def execute_sql(state: AgentState) -> dict[str, Any]:
        sql = state.get("sql") or ""
        try:
            result = run_sql(
                sql,
                db_path=path,
                row_limit=row_limit,
                timeout_ms=timeout_ms,
            )
            return {
                "query_result": result,
                "query_result_text": _format_sql_result(result),
                "sql_error": None,
            }
        except (SqlGuardError, SqlTimeoutError, FileNotFoundError, Exception) as exc:
            return {
                "query_result": None,
                "query_result_text": "",
                "sql_error": str(exc),
            }

    def bump_retry(state: AgentState) -> dict[str, Any]:
        return {"retry_count": int(state.get("retry_count") or 0) + 1}

    def run_discrepancy(state: AgentState) -> dict[str, Any]:
        try:
            frame = find_invoice_discrepancies(db_path=path, timeout_ms=timeout_ms)
            if frame.empty:
                text = "No invoice discrepancies found."
            else:
                text = frame.to_string(index=False)
            return {"discrepancy_text": text, "sql_error": None}
        except Exception as exc:
            return {"discrepancy_text": "", "sql_error": str(exc)}

    def refuse(state: AgentState) -> dict[str, Any]:
        reason = state.get("intent_rationale") or "outside warehouse scope"
        answer = (
            "I cannot answer that from the shipping warehouse data. "
            f"({reason}) "
            "I can answer questions about shipments, carriers, contract rates, "
            "and invoice discrepancies — nothing else."
        )
        return {"answer": answer, "answered": False}

    def summarise(state: AgentState) -> dict[str, Any]:
        intent = state.get("intent")
        sql = state.get("sql") or ""
        sql_error = state.get("sql_error")
        unsupported = bool(sql_error and str(sql_error).upper().startswith("UNSUPPORTED:"))

        if intent == "invoice_check":
            payload = state.get("discrepancy_text") or "(no discrepancy output)"
            system = (
                "Summarise invoice discrepancy findings in plain English for a "
                "business reader. Mention each error_type and its count. "
                "Do not invent numbers. End with a short note that results came "
                "from the find_invoice_discrepancies tool (curated SQL)."
            )
            user = (
                f"Question: {state['question']}\n\n"
                f"Discrepancy summary table:\n{payload}"
            )
        elif unsupported:
            reason = str(sql_error).split(":", 1)[-1].strip() or "not in schema"
            return {
                "answer": (
                    "I cannot answer that from the available warehouse tables. "
                    f"{reason} "
                    "Ask about shipments, invoices, carriers, countries, dates, "
                    "or contract rates."
                ),
                "answered": False,
            }
        elif sql_error:
            system = (
                "Explain that the agent could not produce a reliable SQL answer "
                "after retries. Be honest; do not guess numbers. Mention the "
                "last SQL and the error briefly."
            )
            user = (
                f"Question: {state['question']}\n"
                f"Last SQL:\n{sql}\n"
                f"Error:\n{sql_error}\n"
                f"Retries used: {state.get('retry_count', 0)}/{MAX_SQL_RETRIES}"
            )
        else:
            result_text = state.get("query_result_text") or "(no rows)"
            system = (
                "Answer the user's shipping question in plain English using ONLY "
                "the SQL result below. If the result is empty or insufficient, "
                "say you cannot tell from the data — do not guess.\n"
                "Always include a short section titled 'SQL used:' followed by "
                "the SQL in a fenced ```sql block."
            )
            user = (
                f"Question: {state['question']}\n\n"
                f"SQL:\n{sql}\n\n"
                f"Result:\n{result_text}"
            )

        # Template fallback when LLM is unavailable (tests / offline).
        try:
            model = _llm()
        except Exception:
            model = None

        if model is None:
            if intent == "invoice_check":
                answer = (
                    "Invoice discrepancy summary:\n"
                    f"{state.get('discrepancy_text') or '(none)'}\n\n"
                    "Source: find_invoice_discrepancies tool."
                )
            elif sql_error:
                answer = (
                    "Could not answer from the warehouse after SQL retries.\n"
                    f"Error: {sql_error}\n"
                    f"SQL used:\n```sql\n{sql}\n```"
                )
            else:
                answer = (
                    f"{state.get('query_result_text') or '(no rows)'}\n\n"
                    f"SQL used:\n```sql\n{sql}\n```"
                )
            return {"answer": answer, "answered": not bool(sql_error)}

        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        answer = _content_to_text(response.content).strip()
        # Guarantee SQL visibility for sql_qa successes.
        if intent == "sql_qa" and sql and "```sql" not in answer.lower() and not sql_error:
            answer = f"{answer}\n\nSQL used:\n```sql\n{sql}\n```"
        return {"answer": answer, "answered": not bool(sql_error)}

    return {
        "understand_question": understand_question,
        "get_schema": get_schema_node,
        "write_sql": write_sql,
        "validate_sql": validate_sql,
        "execute_sql": execute_sql,
        "bump_retry": bump_retry,
        "run_discrepancy": run_discrepancy,
        "refuse": refuse,
        "summarise": summarise,
    }


def route_after_understand(state: AgentState) -> str:
    intent = state.get("intent") or "unanswerable"
    if intent == "invoice_check":
        return "run_discrepancy"
    if intent == "sql_qa":
        return "get_schema"
    return "refuse"


def route_after_validate(state: AgentState) -> str:
    err = state.get("sql_error")
    if err and str(err).upper().startswith("UNSUPPORTED:"):
        return "summarise"
    if err:
        return "maybe_retry"
    return "execute_sql"


def route_after_execute(state: AgentState) -> str:
    if state.get("sql_error"):
        return "maybe_retry"
    return "summarise"


def route_retry(state: AgentState) -> str:
    retries = int(state.get("retry_count") or 0)
    if retries < MAX_SQL_RETRIES:
        return "bump_retry"
    return "summarise"
