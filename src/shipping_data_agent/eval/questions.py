"""Fifteen eval questions with DuckDB SQL that defines the correct answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Reuse the curated discrepancy SQL from Step 4 tools so ground truth matches
# the invoice-check tool the agent is expected to call.
from shipping_data_agent.tools.warehouse_tools import DISCREPANCY_SUMMARY_SQL

GradeKind = Literal["scalar", "rows", "discrepancy"]


@dataclass(frozen=True)
class EvalQuestion:
    """One known question + SQL that yields the expected answer."""

    id: str
    question: str
    sql: str
    kind: GradeKind
    intent: Literal["sql_qa", "invoice_check"]
    notes: str = ""


EVAL_QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        id="q01_shipment_count",
        question="How many shipments are in the warehouse?",
        sql="SELECT COUNT(*) AS n FROM fact_shipments",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q02_delivered_germany",
        question="How many shipments were delivered to Germany?",
        sql="""
        SELECT COUNT(*) AS n
        FROM fact_shipments s
        JOIN dim_country c ON s.country_sk = c.country_sk
        WHERE c.country_code = 'DE' AND s.status = 'delivered'
        """,
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q03_invoice_count",
        question="How many invoices are there?",
        sql="SELECT COUNT(*) AS n FROM fact_invoices",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q04_total_billed",
        question=(
            "What is the total billed amount across all invoices in EUR, "
            "rounded to 2 decimal places?"
        ),
        sql="SELECT ROUND(SUM(billed_price), 2) AS total_eur FROM fact_invoices",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q05_avg_delivery_by_carrier",
        question=(
            "What is the average delivery_days by carrier, excluding cancelled "
            "shipments, rounded to 2 decimals?"
        ),
        sql="""
        SELECT dc.carrier_id,
               ROUND(AVG(s.delivery_days), 2) AS avg_delivery_days
        FROM fact_shipments s
        JOIN dim_carrier dc ON s.carrier_sk = dc.carrier_sk
        WHERE s.status <> 'cancelled'
        GROUP BY dc.carrier_id
        ORDER BY dc.carrier_id
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q06_top_carriers_by_shipments",
        question="List the top 5 carriers by shipment count with their counts.",
        sql="""
        SELECT dc.carrier_id, COUNT(*) AS shipment_count
        FROM fact_shipments s
        JOIN dim_carrier dc ON s.carrier_sk = dc.carrier_sk
        GROUP BY dc.carrier_id
        ORDER BY shipment_count DESC, dc.carrier_id
        LIMIT 5
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q07_delayed_count",
        question="How many shipments have status delayed?",
        sql="SELECT COUNT(*) AS n FROM fact_shipments WHERE status = 'delayed'",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q08_avg_weight",
        question="What is the average shipment weight_kg, rounded to 2 decimals?",
        sql="SELECT ROUND(AVG(weight_kg), 2) AS avg_weight_kg FROM fact_shipments",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q09_carrier_dim_count",
        question="How many carriers are in the carrier dimension?",
        sql="SELECT COUNT(*) AS n FROM dim_carrier",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q10_shipments_2024",
        question="How many shipments were shipped in calendar year 2024?",
        sql="""
        SELECT COUNT(*) AS n
        FROM fact_shipments s
        JOIN dim_date d ON s.date_sk = d.date_sk
        WHERE d.year = 2024
        """,
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q11_dhl_invoices",
        question="How many invoices were billed by DHL?",
        sql="""
        SELECT COUNT(*) AS n
        FROM fact_invoices i
        JOIN dim_carrier c ON i.carrier_sk = c.carrier_sk
        WHERE c.carrier_id = 'DHL'
        """,
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q12_top_parcel_type",
        question="What is the most common parcel_type and how many shipments use it?",
        sql="""
        SELECT parcel_type, COUNT(*) AS n
        FROM fact_shipments
        GROUP BY parcel_type
        ORDER BY n DESC, parcel_type
        LIMIT 1
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q13_max_weight",
        question="What is the maximum shipment weight_kg, rounded to 2 decimals?",
        sql="SELECT ROUND(MAX(weight_kg), 2) AS max_weight_kg FROM fact_shipments",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q14_contract_rate_rows",
        question="How many contract rate rows are in the warehouse?",
        sql="SELECT COUNT(*) AS n FROM fact_contract_rates",
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q15_invoice_discrepancies",
        question=(
            "Find invoice discrepancies and report the count for each error type "
            "(duplicate_billing, wrong_weight_class, price_above_contract, "
            "nonexistent_shipment)."
        ),
        sql=DISCREPANCY_SUMMARY_SQL,
        kind="discrepancy",
        intent="invoice_check",
        notes="Ground truth uses the same curated SQL as find_invoice_discrepancies.",
    ),
)


def get_eval_questions() -> tuple[EvalQuestion, ...]:
    """Return the fixed 15-question eval bank."""
    return EVAL_QUESTIONS
