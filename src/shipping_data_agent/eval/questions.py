"""Eval questions with DuckDB SQL that defines the correct answers."""

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
    # Harder follow-ons (multi-table / window / relative periods). Live scores for
    # these are not in README until a new eval run is recorded.
    EvalQuestion(
        id="q16_top_carrier_avg_delivery_per_country",
        question=(
            "For each destination country, which carrier has the highest average "
            "delivery_days (excluding cancelled), and what is that average rounded "
            "to 2 decimals?"
        ),
        sql="""
        WITH ranked AS (
          SELECT c.country_code,
                 dc.carrier_id,
                 ROUND(AVG(s.delivery_days), 2) AS avg_delivery_days,
                 ROW_NUMBER() OVER (
                   PARTITION BY c.country_code
                   ORDER BY AVG(s.delivery_days) DESC, dc.carrier_id
                 ) AS rn
          FROM fact_shipments s
          JOIN dim_country c ON s.country_sk = c.country_sk
          JOIN dim_carrier dc ON s.carrier_sk = dc.carrier_sk
          WHERE s.status <> 'cancelled'
          GROUP BY c.country_code, dc.carrier_id
        )
        SELECT country_code, carrier_id, avg_delivery_days
        FROM ranked
        WHERE rn = 1
        ORDER BY country_code
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q17_delayed_share_per_carrier",
        question=(
            "What share of shipments is delayed for each carrier, as a percentage "
            "rounded to 2 decimals?"
        ),
        sql="""
        SELECT dc.carrier_id,
               ROUND(
                 100.0 * SUM(CASE WHEN s.status = 'delayed' THEN 1 ELSE 0 END)
                 / COUNT(*),
                 2
               ) AS delayed_pct
        FROM fact_shipments s
        JOIN dim_carrier dc ON s.carrier_sk = dc.carrier_sk
        GROUP BY dc.carrier_id
        ORDER BY dc.carrier_id
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q18_mom_billed_jan2025",
        question=(
            "What was the month-over-month change in total billed EUR for "
            "January 2025 versus December 2024, rounded to 2 decimals?"
        ),
        sql="""
        WITH monthly AS (
          SELECT d.year, d.month, ROUND(SUM(i.billed_price), 2) AS total_billed
          FROM fact_invoices i
          JOIN dim_date d ON i.invoice_date_sk = d.date_sk
          GROUP BY d.year, d.month
        )
        SELECT ROUND(
          (SELECT total_billed FROM monthly WHERE year = 2025 AND month = 1)
          - (SELECT total_billed FROM monthly WHERE year = 2024 AND month = 12),
          2
        ) AS mom_change_eur
        """,
        kind="scalar",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q19_over_contract_by_carrier",
        question=(
            "How many invoices are billed more than 10% above the reconstructed "
            "contract price, grouped by carrier?"
        ),
        sql="""
        WITH country_mult AS (
          SELECT country_sk,
            CASE country_code
              WHEN 'DE' THEN 1.00 WHEN 'NL' THEN 1.02 WHEN 'BE' THEN 1.05
              WHEN 'FR' THEN 1.08 WHEN 'PL' THEN 0.95 WHEN 'CZ' THEN 0.93
              WHEN 'IT' THEN 1.12 WHEN 'ES' THEN 1.10 WHEN 'GB' THEN 1.18
              WHEN 'US' THEN 1.35 ELSE 1.00
            END AS mult
          FROM dim_country
        ),
        enriched AS (
          SELECT carr.carrier_id,
                 i.billed_price,
                 r.rate_eur
                   + (
                       s.weight_kg
                       - CASE
                           WHEN r.max_kg IS NULL THEN r.min_kg + 10.0
                           ELSE (r.min_kg + r.max_kg) / 2.0
                         END
                     ) * 0.35 * cm.mult AS expected_rate
          FROM fact_invoices i
          JOIN fact_shipments s ON i.shipment_id = s.shipment_id
          JOIN dim_carrier carr ON i.carrier_sk = carr.carrier_sk
          JOIN country_mult cm ON s.country_sk = cm.country_sk
          JOIN fact_contract_rates r
            ON r.carrier_sk = s.carrier_sk
           AND r.country_sk = s.country_sk
           AND s.weight_kg >= r.min_kg
           AND (r.max_kg IS NULL OR s.weight_kg < r.max_kg)
        )
        SELECT carrier_id, COUNT(*) AS n_above_contract
        FROM enriched
        WHERE billed_price > expected_rate * 1.10
        GROUP BY carrier_id
        ORDER BY carrier_id
        """,
        kind="rows",
        intent="sql_qa",
    ),
    EvalQuestion(
        id="q20_shipments_without_invoice",
        question="How many shipments have no matching invoice?",
        sql="""
        SELECT COUNT(*) AS n
        FROM fact_shipments s
        LEFT JOIN fact_invoices i ON s.shipment_id = i.shipment_id
        WHERE i.invoice_id IS NULL
        """,
        kind="scalar",
        intent="sql_qa",
    ),
)


def get_eval_questions() -> tuple[EvalQuestion, ...]:
    """Return the fixed eval question bank."""
    return EVAL_QUESTIONS
