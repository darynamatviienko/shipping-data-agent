"""DuckDB star-schema DDL and load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from shipping_data_agent.etl.cleaning import COUNTRY_NAMES

SCHEMA_SQL = """
CREATE OR REPLACE TABLE dim_carrier (
    carrier_sk INTEGER PRIMARY KEY,
    carrier_id VARCHAR NOT NULL UNIQUE,
    name VARCHAR NOT NULL,
    hq_country VARCHAR NOT NULL,
    service_tier VARCHAR NOT NULL
);

CREATE OR REPLACE TABLE dim_country (
    country_sk INTEGER PRIMARY KEY,
    country_code VARCHAR NOT NULL UNIQUE,
    country_name VARCHAR NOT NULL
);

CREATE OR REPLACE TABLE dim_date (
    date_sk INTEGER PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    year INTEGER NOT NULL,
    month INTEGER NOT NULL,
    day INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    iso_week INTEGER NOT NULL
);

CREATE OR REPLACE TABLE fact_shipments (
    shipment_id VARCHAR PRIMARY KEY,
    date_sk INTEGER NOT NULL,
    carrier_sk INTEGER NOT NULL,
    country_sk INTEGER NOT NULL,
    weight_kg DOUBLE NOT NULL,
    parcel_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    delivery_days INTEGER NOT NULL
);

CREATE OR REPLACE TABLE fact_invoices (
    invoice_id VARCHAR PRIMARY KEY,
    shipment_id VARCHAR NOT NULL,
    carrier_sk INTEGER NOT NULL,
    invoice_date_sk INTEGER NOT NULL,
    billed_price DOUBLE NOT NULL
);

CREATE OR REPLACE TABLE fact_contract_rates (
    contract_rate_sk INTEGER PRIMARY KEY,
    carrier_sk INTEGER NOT NULL,
    country_sk INTEGER NOT NULL,
    weight_band VARCHAR NOT NULL,
    min_kg DOUBLE NOT NULL,
    max_kg DOUBLE,
    rate_eur DOUBLE NOT NULL,
    currency VARCHAR NOT NULL,
    effective_from DATE NOT NULL
);
"""


def connect(db_path: Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotent full replace of warehouse tables."""
    con.execute(SCHEMA_SQL)


def build_dim_date(dates: pd.Series) -> pd.DataFrame:
    unique = (
        pd.to_datetime(pd.Series(dates).dropna(), errors="coerce")
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(unique) == 0:
        return pd.DataFrame(
            columns=[
                "date_sk",
                "date",
                "year",
                "month",
                "day",
                "quarter",
                "day_of_week",
                "iso_week",
            ]
        )
    frame = pd.DataFrame({"date": unique})
    frame["date_sk"] = (
        frame["date"].dt.year * 10_000
        + frame["date"].dt.month * 100
        + frame["date"].dt.day
    ).astype(int)
    frame["year"] = frame["date"].dt.year.astype(int)
    frame["month"] = frame["date"].dt.month.astype(int)
    frame["day"] = frame["date"].dt.day.astype(int)
    frame["quarter"] = frame["date"].dt.quarter.astype(int)
    frame["day_of_week"] = frame["date"].dt.day_of_week.astype(int)
    frame["iso_week"] = frame["date"].dt.isocalendar().week.astype(int)
    return frame[
        [
            "date_sk",
            "date",
            "year",
            "month",
            "day",
            "quarter",
            "day_of_week",
            "iso_week",
        ]
    ]


def build_dim_carrier(carriers: pd.DataFrame) -> pd.DataFrame:
    out = carriers[["carrier_id", "name", "hq_country", "service_tier"]].copy()
    out = out.reset_index(drop=True)
    out.insert(0, "carrier_sk", out.index + 1)
    return out


def build_dim_country(country_codes: Iterable[str] | pd.Series) -> pd.DataFrame:
    codes = sorted({str(c).upper() for c in country_codes if pd.notna(c) and str(c)})
    rows = [
        {
            "country_sk": i,
            "country_code": code,
            "country_name": COUNTRY_NAMES.get(code, code),
        }
        for i, code in enumerate(codes, start=1)
    ]
    return pd.DataFrame(rows)


def load_star_schema(
    con: duckdb.DuckDBPyConnection,
    *,
    carriers: pd.DataFrame,
    contract_rates: pd.DataFrame,
    shipments: pd.DataFrame,
    invoices: pd.DataFrame,
) -> dict[str, int]:
    """Replace warehouse contents with cleaned frames (idempotent via CREATE OR REPLACE)."""
    create_schema(con)

    dim_carrier = build_dim_carrier(carriers)
    country_codes = set(carriers["hq_country"]) | set(shipments["country"]) | set(
        contract_rates["country"]
    )
    dim_country = build_dim_country(country_codes)

    all_dates = pd.concat(
        [
            shipments["date"],
            invoices["invoice_date"],
            contract_rates["effective_from"],
        ],
        ignore_index=True,
    )
    dim_date = build_dim_date(all_dates)

    carrier_sk = dim_carrier.set_index("carrier_id")["carrier_sk"]
    country_sk = dim_country.set_index("country_code")["country_sk"]
    date_sk = dim_date.set_index("date")["date_sk"]

    fact_shipments = pd.DataFrame(
        {
            "shipment_id": shipments["id"],
            "date_sk": shipments["date"].dt.normalize().map(date_sk).astype(int),
            "carrier_sk": shipments["carrier"].map(carrier_sk).astype(int),
            "country_sk": shipments["country"].map(country_sk).astype(int),
            "weight_kg": shipments["weight_kg"].astype(float),
            "parcel_type": shipments["parcel_type"],
            "status": shipments["status"],
            "delivery_days": shipments["delivery_days"].astype(int),
        }
    )

    fact_invoices = pd.DataFrame(
        {
            "invoice_id": invoices["invoice_id"],
            "shipment_id": invoices["shipment_id"],
            "carrier_sk": invoices["carrier"].map(carrier_sk).astype(int),
            "invoice_date_sk": invoices["invoice_date"]
            .dt.normalize()
            .map(date_sk)
            .astype(int),
            "billed_price": invoices["billed_price"].astype(float),
        }
    )

    rates = contract_rates.reset_index(drop=True)
    fact_contract_rates = pd.DataFrame(
        {
            "contract_rate_sk": rates.index + 1,
            "carrier_sk": rates["carrier_id"].map(carrier_sk).astype(int),
            "country_sk": rates["country"].map(country_sk).astype(int),
            "weight_band": rates["weight_band"],
            "min_kg": rates["min_kg"].astype(float),
            "max_kg": rates["max_kg"].astype(float),
            "rate_eur": rates["rate_eur"].astype(float),
            "currency": rates["currency"],
            "effective_from": rates["effective_from"].dt.normalize(),
        }
    )

    con.register("_dim_carrier", dim_carrier)
    con.register("_dim_country", dim_country)
    con.register("_dim_date", dim_date)
    con.register("_fact_shipments", fact_shipments)
    con.register("_fact_invoices", fact_invoices)
    con.register("_fact_contract_rates", fact_contract_rates)

    con.execute("INSERT INTO dim_carrier SELECT * FROM _dim_carrier")
    con.execute("INSERT INTO dim_country SELECT * FROM _dim_country")
    con.execute("INSERT INTO dim_date SELECT * FROM _dim_date")
    con.execute("INSERT INTO fact_shipments SELECT * FROM _fact_shipments")
    con.execute("INSERT INTO fact_invoices SELECT * FROM _fact_invoices")
    con.execute("INSERT INTO fact_contract_rates SELECT * FROM _fact_contract_rates")

    return {
        "dim_carrier": len(dim_carrier),
        "dim_country": len(dim_country),
        "dim_date": len(dim_date),
        "fact_shipments": len(fact_shipments),
        "fact_invoices": len(fact_invoices),
        "fact_contract_rates": len(fact_contract_rates),
    }
