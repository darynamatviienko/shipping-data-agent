"""Unit tests for ETL cleaning rules (types, nulls, duplicates, invalid values)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shipping_data_agent.etl.cleaning import (
    clean_carriers,
    clean_contract_rates,
    clean_invoices,
    clean_shipments,
)
from shipping_data_agent.etl.pipeline import run_etl


def _carriers_row(**overrides):
    base = {
        "carrier_id": "DHL",
        "name": "DHL Express",
        "hq_country": "DE",
        "service_tier": "express",
    }
    base.update(overrides)
    return base


def _shipment_row(**overrides):
    base = {
        "id": "SHP-000001",
        "date": "2024-03-06",
        "carrier": "DHL",
        "country": "DE",
        "weight_kg": 2.5,
        "parcel_type": "small_box",
        "status": "delivered",
        "delivery_days": 3,
    }
    base.update(overrides)
    return base


def _invoice_row(**overrides):
    base = {
        "invoice_id": "INV-000001",
        "shipment_id": "SHP-000001",
        "carrier": "DHL",
        "billed_price": 12.50,
        "invoice_date": "2024-03-10",
    }
    base.update(overrides)
    return base


def _rate_row(**overrides):
    base = {
        "carrier_id": "DHL",
        "country": "DE",
        "weight_band": "1-5",
        "min_kg": 1.0,
        "max_kg": 5.0,
        "rate_eur": 14.7,
        "currency": "EUR",
        "effective_from": "2024-01-01",
    }
    base.update(overrides)
    return base


# --- carriers ---


def test_carriers_reject_null_and_invalid_tier():
    df = pd.DataFrame(
        [
            _carriers_row(),
            _carriers_row(carrier_id="UPS", name="UPS", service_tier=None),
            _carriers_row(carrier_id="BAD", name="Bad", service_tier="premium"),
        ]
    )
    result = clean_carriers(df)
    assert result.n_accepted == 1
    assert set(result.rejected["_reason"]) == {"null_service_tier", "invalid_service_tier"}


def test_carriers_reject_duplicate_id():
    df = pd.DataFrame([_carriers_row(), _carriers_row(name="DHL Dup")])
    result = clean_carriers(df)
    assert result.n_accepted == 1
    assert result.rejected["_reason"].tolist() == ["duplicate_carrier_id"]


# --- shipments ---


def test_shipments_reject_null_required():
    df = pd.DataFrame([_shipment_row(id=None), _shipment_row(id="SHP-000002")])
    result = clean_shipments(df)
    assert result.n_accepted == 1
    assert "null_id" in set(result.rejected["_reason"])


def test_shipments_reject_invalid_date_and_weight():
    df = pd.DataFrame(
        [
            _shipment_row(id="SHP-1", date="not-a-date"),
            _shipment_row(id="SHP-2", weight_kg=-1),
            _shipment_row(id="SHP-3", weight_kg="heavy"),
            _shipment_row(id="SHP-4"),
        ]
    )
    result = clean_shipments(df)
    assert result.n_accepted == 1
    reasons = set(result.rejected["_reason"])
    assert "invalid_date" in reasons
    assert "non_positive_weight_kg" in reasons
    assert "invalid_weight_kg_type" in reasons


def test_shipments_reject_invalid_enums_and_duplicate():
    df = pd.DataFrame(
        [
            _shipment_row(id="SHP-1", parcel_type="crate"),
            _shipment_row(id="SHP-2", status="lost"),
            _shipment_row(id="SHP-3", country="Deutschland"),
            _shipment_row(id="SHP-4"),
            _shipment_row(id="SHP-4", weight_kg=9.0),
        ]
    )
    result = clean_shipments(df)
    assert result.n_accepted == 1
    reasons = set(result.rejected["_reason"])
    assert reasons >= {
        "invalid_parcel_type",
        "invalid_status",
        "invalid_country",
        "duplicate_shipment_id",
    }


def test_shipments_reject_negative_delivery_days():
    df = pd.DataFrame([_shipment_row(delivery_days=-2), _shipment_row(id="SHP-2")])
    result = clean_shipments(df)
    assert result.n_accepted == 1
    assert result.rejected["_reason"].tolist() == ["negative_delivery_days"]


# --- invoices ---


def test_invoices_reject_nulls_negative_price_bad_date():
    df = pd.DataFrame(
        [
            _invoice_row(invoice_id=None),
            _invoice_row(invoice_id="INV-2", billed_price=-5),
            _invoice_row(invoice_id="INV-3", billed_price="x"),
            _invoice_row(invoice_id="INV-4", invoice_date="33/33/3333"),
            _invoice_row(invoice_id="INV-5"),
        ]
    )
    result = clean_invoices(df)
    assert result.n_accepted == 1
    reasons = set(result.rejected["_reason"])
    assert reasons >= {
        "null_invoice_id",
        "negative_billed_price",
        "invalid_billed_price_type",
        "invalid_invoice_date",
    }


def test_invoices_reject_duplicate_invoice_id_keeps_first():
    df = pd.DataFrame(
        [
            _invoice_row(invoice_id="INV-1", billed_price=10.0),
            _invoice_row(invoice_id="INV-1", billed_price=99.0),
        ]
    )
    result = clean_invoices(df)
    assert result.n_accepted == 1
    assert float(result.accepted.iloc[0]["billed_price"]) == 10.0
    assert result.rejected["_reason"].tolist() == ["duplicate_invoice_id"]


def test_invoices_keep_orphan_shipment_and_multi_bill():
    """Business-logic faults stay in the warehouse for the agent to find."""
    df = pd.DataFrame(
        [
            _invoice_row(invoice_id="INV-1", shipment_id="SHP-MISSING"),
            _invoice_row(invoice_id="INV-2", shipment_id="SHP-000001"),
            _invoice_row(invoice_id="INV-3", shipment_id="SHP-000001"),
        ]
    )
    result = clean_invoices(df)
    assert result.n_accepted == 3
    assert result.n_rejected == 0


# --- contract rates ---


def test_contract_rates_reject_bad_band_currency_and_bounds():
    df = pd.DataFrame(
        [
            _rate_row(),
            _rate_row(weight_band="100-200", country="FR"),
            _rate_row(currency="USD", country="NL"),
            _rate_row(min_kg=10, max_kg=5, country="BE"),
            _rate_row(rate_eur=0, country="PL"),
            _rate_row(weight_band="50+", min_kg=50.0, max_kg=None, country="IT"),
        ]
    )
    result = clean_contract_rates(df)
    assert result.n_accepted == 2  # good + open 50+ band
    reasons = set(result.rejected["_reason"])
    assert reasons >= {
        "invalid_weight_band",
        "unsupported_currency",
        "max_kg_not_above_min_kg",
        "non_positive_rate_eur",
    }


# --- end-to-end pipeline ---


def test_run_etl_idempotent_and_writes_rejected(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()

    pd.DataFrame(
        [
            _carriers_row(),
            _carriers_row(carrier_id="UPS", name="UPS", hq_country="US"),
        ]
    ).to_csv(raw / "carriers.csv", index=False)

    pd.DataFrame(
        [
            _rate_row(),
            _rate_row(carrier_id="UPS", country="US"),
            _rate_row(carrier_id="UPS", country="US", rate_eur=-1),  # reject
        ]
    ).to_csv(raw / "contract_rates.csv", index=False)

    pd.DataFrame(
        [
            _shipment_row(id="SHP-1", carrier="DHL"),
            _shipment_row(id="SHP-2", carrier="UPS", country="US"),
            _shipment_row(id="SHP-3", carrier="DHL", weight_kg=0),  # reject
        ]
    ).to_csv(raw / "shipments.csv", index=False)

    pd.DataFrame(
        [
            _invoice_row(invoice_id="INV-1", shipment_id="SHP-1", carrier="DHL"),
            _invoice_row(invoice_id="INV-2", shipment_id="SHP-2", carrier="UPS"),
            _invoice_row(invoice_id="INV-2", shipment_id="SHP-1", carrier="DHL"),  # dup id
        ]
    ).to_csv(raw / "carrier_invoices.csv", index=False)

    db_path = warehouse / "shipping.duckdb"
    rejected_path = warehouse / "rejected_rows.csv"

    first = run_etl(raw_dir=raw, db_path=db_path, rejected_path=rejected_path)
    second = run_etl(raw_dir=raw, db_path=db_path, rejected_path=rejected_path)

    assert first["table_counts"] == second["table_counts"]
    assert first["table_counts"]["fact_shipments"] == 2
    assert first["table_counts"]["fact_invoices"] == 2
    assert first["table_counts"]["dim_carrier"] == 2
    assert first["n_rejected"] >= 2

    rejected = pd.read_csv(rejected_path)
    assert "_reason" in rejected.columns
    assert set(rejected["_reason"]) >= {"non_positive_weight_kg", "duplicate_invoice_id"}

    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        n_ship = con.execute("SELECT COUNT(*) FROM fact_shipments").fetchone()[0]
        n_inv = con.execute("SELECT COUNT(*) FROM fact_invoices").fetchone()[0]
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
    finally:
        con.close()

    assert n_ship == 2
    assert n_inv == 2
    assert tables >= {
        "dim_carrier",
        "dim_country",
        "dim_date",
        "fact_shipments",
        "fact_invoices",
        "fact_contract_rates",
    }


def test_clean_shipments_normalizes_types():
    df = pd.DataFrame([_shipment_row(country="de", parcel_type="Small_Box", status="DELIVERED")])
    result = clean_shipments(df)
    assert result.n_accepted == 1
    row = result.accepted.iloc[0]
    assert row["country"] == "DE"
    assert row["parcel_type"] == "small_box"
    assert row["status"] == "delivered"
    assert pd.api.types.is_datetime64_any_dtype(result.accepted["date"])
