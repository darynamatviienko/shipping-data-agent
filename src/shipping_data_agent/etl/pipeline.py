"""Orchestrate raw CSV → clean → DuckDB warehouse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shipping_data_agent.etl.cleaning import (
    CleanResult,
    clean_carriers,
    clean_contract_rates,
    clean_invoices,
    clean_shipments,
    combine_rejected,
)
from shipping_data_agent.etl.warehouse import connect, load_star_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "warehouse" / "shipping.duckdb"
DEFAULT_REJECTED_PATH = REPO_ROOT / "data" / "warehouse" / "rejected_rows.csv"

RAW_FILES = {
    "carriers": "carriers.csv",
    "contract_rates": "contract_rates.csv",
    "shipments": "shipments.csv",
    "invoices": "carrier_invoices.csv",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    return pd.read_csv(path)


def _reject_unknown_carriers(
    source: str,
    result: CleanResult,
    carrier_col: str,
    known: set[str],
) -> CleanResult:
    """Drop rows whose carrier is not in dim_carrier; append to rejected."""
    accepted = result.accepted
    if accepted.empty:
        return result
    bad_mask = ~accepted[carrier_col].isin(known)
    if not bad_mask.any():
        return result
    bad = accepted.loc[bad_mask].copy()
    # Recover approximate CSV row numbers when available via original index is lost;
    # mark as post-clean referential check.
    bad["_source"] = source
    bad["_reason"] = "unknown_carrier"
    bad["_row_number"] = -1
    rejected = (
        pd.concat([result.rejected, bad], ignore_index=True)
        if len(result.rejected)
        else bad
    )
    return CleanResult(accepted=accepted.loc[~bad_mask].reset_index(drop=True), rejected=rejected)


def run_etl(
    raw_dir: Path | None = None,
    db_path: Path | None = None,
    rejected_path: Path | None = None,
) -> dict:
    """Load, clean, and warehouse raw CSVs. Safe to re-run (tables replaced)."""
    raw_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    rejected_path = Path(rejected_path) if rejected_path else DEFAULT_REJECTED_PATH

    raw_carriers = _read_csv(raw_dir / RAW_FILES["carriers"])
    raw_rates = _read_csv(raw_dir / RAW_FILES["contract_rates"])
    raw_shipments = _read_csv(raw_dir / RAW_FILES["shipments"])
    raw_invoices = _read_csv(raw_dir / RAW_FILES["invoices"])

    carriers = clean_carriers(raw_carriers)
    known_carriers = set(carriers.accepted["carrier_id"])

    rates = clean_contract_rates(raw_rates)
    rates = _reject_unknown_carriers(
        "contract_rates.csv", rates, "carrier_id", known_carriers
    )

    shipments = clean_shipments(raw_shipments)
    shipments = _reject_unknown_carriers(
        "shipments.csv", shipments, "carrier", known_carriers
    )

    invoices = clean_invoices(raw_invoices)
    invoices = _reject_unknown_carriers(
        "carrier_invoices.csv", invoices, "carrier", known_carriers
    )

    rejected = combine_rejected(
        [carriers.rejected, rates.rejected, shipments.rejected, invoices.rejected]
    )
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected.to_csv(rejected_path, index=False)

    con = connect(db_path)
    try:
        counts = load_star_schema(
            con,
            carriers=carriers.accepted,
            contract_rates=rates.accepted,
            shipments=shipments.accepted,
            invoices=invoices.accepted,
        )
    finally:
        con.close()

    summary = {
        "raw_dir": str(raw_dir),
        "db_path": str(db_path),
        "rejected_path": str(rejected_path),
        "n_rejected": int(len(rejected)),
        "rejected_by_source": (
            rejected.groupby("_source").size().to_dict() if len(rejected) else {}
        ),
        "table_counts": counts,
        "accepted": {
            "carriers": carriers.n_accepted,
            "contract_rates": rates.n_accepted,
            "shipments": shipments.n_accepted,
            "invoices": invoices.n_accepted,
        },
    }
    return summary
