"""Step 6 eval: question bank, SQL ground truth, offline grading (no live LLM)."""

from __future__ import annotations

from pathlib import Path

import pytest

from shipping_data_agent.eval.grading import (
    answer_contains_number,
    extract_numbers,
    grade_answer,
    sql_rows_to_expected,
)
from shipping_data_agent.eval.harness import compute_ground_truth, format_readme_section, run_eval
from shipping_data_agent.eval.questions import EVAL_QUESTIONS, get_eval_questions
from shipping_data_agent.tools import DEFAULT_DB_PATH


WAREHOUSE = DEFAULT_DB_PATH


@pytest.fixture(scope="module")
def warehouse() -> Path:
    if not WAREHOUSE.exists():
        pytest.skip("warehouse missing — run make etl")
    return WAREHOUSE


def test_eval_bank_has_expected_questions():
    bank = get_eval_questions()
    assert len(bank) == 20
    assert len({q.id for q in bank}) == 20
    assert bank is EVAL_QUESTIONS
    assert {q.id for q in bank} >= {
        "q01_shipment_count",
        "q15_invoice_discrepancies",
        "q16_top_carrier_avg_delivery_per_country",
        "q20_shipments_without_invoice",
    }


def test_ground_truth_sql_runs(warehouse: Path):
    for q in get_eval_questions():
        expected = compute_ground_truth(q, db_path=warehouse)
        assert expected["kind"] == q.kind
        assert expected["display"]
        if q.kind == "scalar":
            assert expected["value"] is not None
        elif q.kind == "discrepancy":
            assert len(expected["pairs"]) == 4
            types = {p["error_type"] for p in expected["pairs"]}
            assert types == {
                "duplicate_billing",
                "wrong_weight_class",
                "price_above_contract",
                "nonexistent_shipment",
            }
        else:
            assert expected["rows"]


def test_extract_and_match_numbers():
    assert extract_numbers("There are 1,472 delivered shipments.") == [1472.0]
    assert answer_contains_number("avg weight is 16.74 kg", 16.74)
    assert answer_contains_number("total 624580.26 EUR", 624580.26)
    assert not answer_contains_number("count is 10", 11)


def test_grade_scalar_and_discrepancy(warehouse: Path):
    q_scalar = next(q for q in get_eval_questions() if q.id == "q01_shipment_count")
    expected = compute_ground_truth(q_scalar, db_path=warehouse)
    ok = grade_answer(q_scalar, expected, f"There are {expected['value']} shipments.")
    assert ok.passed

    bad = grade_answer(q_scalar, expected, "There are 0 shipments.")
    assert not bad.passed

    q_disc = next(q for q in get_eval_questions() if q.id == "q15_invoice_discrepancies")
    disc_expected = compute_ground_truth(q_disc, db_path=warehouse)
    # Build an answer that mentions every type + count.
    parts = [
        f"{p['error_type']}: {p['count']}" for p in disc_expected["pairs"]
    ]
    good = grade_answer(q_disc, disc_expected, "Discrepancies — " + "; ".join(parts))
    assert good.passed

    # Summariser often humanizes codes (FedEx / medium box / Duplicate billing).
    human = grade_answer(
        q_disc,
        disc_expected,
        "Duplicate billing: 292; Wrong weight class: 146; "
        "Nonexistent shipment: 145; Price above contract: 145",
    )
    assert human.passed

    q_rows = next(q for q in get_eval_questions() if q.id == "q06_top_carriers_by_shipments")
    rows_expected = compute_ground_truth(q_rows, db_path=warehouse)
    named = grade_answer(
        q_rows,
        rows_expected,
        "FedEx International 4068, TNT Express 4002, UPS Worldwide 3998, "
        "DHL Express 3979, GLS Parcel 3953",
    )
    assert named.passed

    q_parcel = next(q for q in get_eval_questions() if q.id == "q12_top_parcel_type")
    parcel_expected = compute_ground_truth(q_parcel, db_path=warehouse)
    assert grade_answer(
        q_parcel, parcel_expected, "Most common is medium box with 5981 shipments."
    ).passed


def test_sql_rows_to_expected_helpers():
    scalar = sql_rows_to_expected("scalar", ["n"], [(42,)])
    assert scalar["value"] == 42
    rows = sql_rows_to_expected(
        "rows",
        ["carrier_id", "n"],
        [("DHL", 10), ("UPS", 9)],
    )
    assert len(rows["rows"]) == 2


def test_run_eval_ground_truth_only(warehouse: Path):
    report = run_eval(db_path=warehouse, live=False)
    assert report.question_count == 20
    assert report.live_agent_run is False
    assert report.skipped == 20
    assert report.accuracy is None
    assert all(c.expected_display for c in report.cases)
    assert report.warehouse == "data/warehouse/shipping.duckdb"
    assert "/home/" not in report.warehouse
    assert "/Users/" not in report.warehouse
    md = format_readme_section(report)
    assert "## Evaluation (Step 6)" in md
    assert "q01_shipment_count" in md
    assert "pending live run" in md
