"""Intent classification: heuristic (always) + optional LLM refinement."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

IntentName = Literal["invoice_check", "sql_qa", "unanswerable"]

_INVOICE_PATTERNS = (
    r"\binvoice\b",
    r"\bbilling\b",
    r"\bbilled\b",
    r"\bovercharg",
    r"\bundercharg",
    r"\bdiscrepanc",
    r"\bduplicate\s+bill",
    r"\bwrong\s+weight\s+class",
    r"\bprice\s+above\s+contract",
    r"\bnonexistent\s+shipment",
    r"\bfaulty\s+invoice",
    r"\binvoice\s+error",
    r"\bbilling\s+error",
    r"\baudit\s+invoice",
    r"\bcheck\s+invoice",
    r"\binvoice\s+check",
    r"\bfind\s+.*\berror",
    r"\bdetect\s+.*\berror",
)

_OFFTOPIC_PATTERNS = (
    r"\bweather\b",
    r"\bstock\s+price\b",
    r"\bcapital\s+of\b",
    r"\bwho\s+is\s+the\s+president\b",
    r"\brecipe\b",
    r"\bjoke\b",
    r"\bwrite\s+(me\s+)?a\s+poem\b",
    r"\btranslate\b",
)

_SHIPPING_HINTS = (
    r"\bshipment",
    r"\bshipments?\b",
    r"\bcarrier",
    r"\bparcel",
    r"\bdeliver",
    r"\bweight",
    r"\bcountry",
    r"\binvoice",
    r"\brate",
    r"\bcontract",
    r"\bwarehouse",
    r"\bduckdb\b",
    r"\bbilled",
    r"\bbilling",
)


@dataclass(frozen=True)
class Intent:
    name: IntentName
    rationale: str


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def classify_intent_heuristic(question: str) -> Intent:
    """Fast, deterministic intent router used by tests and as a LLM fallback."""
    q = (question or "").strip()
    if not q:
        return Intent("unanswerable", "Empty question")

    if _matches_any(q, _INVOICE_PATTERNS):
        # Prefer discrepancy tool when the user is clearly auditing invoices.
        invoice_audit = _matches_any(
            q,
            (
                r"\bdiscrepanc",
                r"\berror",
                r"\bfault",
                r"\bovercharg",
                r"\bduplicate",
                r"\bwrong\s+weight",
                r"\babove\s+contract",
                r"\bnonexistent",
                r"\baudit",
                r"\bcheck\b",
                r"\bfind\b",
                r"\bdetect\b",
                r"\bflag\b",
                r"\bproblem",
                r"\bissue",
                r"\bmismatch",
                r"\binvalid",
            ),
        )
        invoice_word = bool(re.search(r"\binvoice", q, flags=re.IGNORECASE))
        invoice_problem = _matches_any(
            q, (r"\berror", r"\bwrong", r"\bbad", r"\binvalid", r"\bissue")
        )
        if invoice_audit or (invoice_word and invoice_problem):
            return Intent(
                "invoice_check",
                "Question asks to check / find invoice billing faults",
            )

    if _matches_any(q, _OFFTOPIC_PATTERNS) and not _matches_any(q, _SHIPPING_HINTS):
        return Intent(
            "unanswerable",
            "Question appears unrelated to the shipping warehouse",
        )

    if _matches_any(q, _SHIPPING_HINTS) or re.search(
        r"\b(how many|average|total|count|sum|list|show|which|what)\b",
        q,
        flags=re.IGNORECASE,
    ):
        return Intent("sql_qa", "Treat as a warehouse SQL analytics question")

    return Intent(
        "unanswerable",
        "No clear link to shipments, invoices, carriers, or rates",
    )


_LLM_SYSTEM = """You classify user questions for a shipping analytics agent.
The agent can ONLY use a DuckDB warehouse with:
- fact_shipments, fact_invoices, fact_contract_rates
- dim_carrier, dim_country, dim_date
and a curated invoice-discrepancy detector for billing faults
(duplicate billing, wrong weight class, price above contract, nonexistent shipment).

Respond with exactly one line:
INTENT=<invoice_check|sql_qa|unanswerable>
RATIONALE=<short reason>

Rules:
- invoice_check: user wants invoice errors / discrepancies / overcharges / audits.
- sql_qa: factual analytics answerable with SELECT against the warehouse.
- unanswerable: anything outside this data (or impossible to answer from it).
Do not invent capabilities the warehouse does not have.
"""


def classify_intent_with_llm(question: str, llm: BaseChatModel) -> Intent:
    """Ask the LLM to classify intent; fall back to heuristic on parse failure."""
    fallback = classify_intent_heuristic(question)
    response = llm.invoke(
        [
            SystemMessage(content=_LLM_SYSTEM),
            HumanMessage(content=question),
        ]
    )
    text = getattr(response, "content", str(response))
    if isinstance(text, list):
        text = " ".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part)
            for part in text
        )
    text = str(text)

    intent_match = re.search(
        r"INTENT\s*=\s*(invoice_check|sql_qa|unanswerable)",
        text,
        flags=re.IGNORECASE,
    )
    rationale_match = re.search(r"RATIONALE\s*=\s*(.+)", text, flags=re.IGNORECASE)
    if not intent_match:
        return fallback
    name = intent_match.group(1).lower()  # type: ignore[assignment]
    rationale = (
        rationale_match.group(1).strip()
        if rationale_match
        else "LLM classification"
    )
    return Intent(name=name, rationale=rationale)  # type: ignore[arg-type]


def classify_intent(
    question: str, *, llm: BaseChatModel | None = None
) -> Intent:
    """Classify intent. Uses LLM when provided; otherwise heuristic only."""
    if llm is None:
        return classify_intent_heuristic(question)

    heuristic = classify_intent_heuristic(question)
    # High-precision heuristic wins for clear invoice audits and empty input.
    if heuristic.name == "invoice_check" or not (question or "").strip():
        return heuristic
    try:
        return classify_intent_with_llm(question, llm)
    except Exception:
        return heuristic
