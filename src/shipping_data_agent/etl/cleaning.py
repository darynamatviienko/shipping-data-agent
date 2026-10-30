"""Validate and clean raw shipping DataFrames; collect rejected rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

VALID_PARCEL_TYPES = frozenset(
    {"envelope", "small_box", "medium_box", "large_box", "pallet"}
)
VALID_STATUSES = frozenset(
    {"delivered", "in_transit", "delayed", "returned", "cancelled"}
)
VALID_SERVICE_TIERS = frozenset({"express", "standard", "economy"})
VALID_WEIGHT_BANDS = frozenset({"0-1", "1-5", "5-20", "20-50", "50+"})

COUNTRY_NAMES = {
    "DE": "Germany",
    "FR": "France",
    "NL": "Netherlands",
    "BE": "Belgium",
    "PL": "Poland",
    "IT": "Italy",
    "ES": "Spain",
    "GB": "United Kingdom",
    "US": "United States",
    "CZ": "Czechia",
}


@dataclass
class CleanResult:
    """Accepted rows plus a frame of rejected rows with reason metadata."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_accepted(self) -> int:
        return len(self.accepted)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)


def _empty_rejected(columns: Iterable[str]) -> pd.DataFrame:
    cols = list(columns) + ["_source", "_reason", "_row_number"]
    return pd.DataFrame(columns=cols)


def _reject(
    source: str,
    original: pd.DataFrame,
    mask: pd.Series,
    reason: str,
    rejected_parts: list[pd.DataFrame],
) -> pd.Series:
    """Append masked rows to rejected_parts; return remaining keep-mask."""
    if not mask.any():
        return ~mask
    bad = original.loc[mask].copy()
    bad["_source"] = source
    bad["_reason"] = reason
    bad["_row_number"] = bad.index.astype(int) + 2  # 1-based CSV row (header=1)
    rejected_parts.append(bad)
    return ~mask


def _finalize(
    source: str,
    working: pd.DataFrame,
    original: pd.DataFrame,
    keep: pd.Series,
    rejected_parts: list[pd.DataFrame],
) -> CleanResult:
    accepted = working.loc[keep].reset_index(drop=True)
    if rejected_parts:
        rejected = pd.concat(rejected_parts, ignore_index=True)
    else:
        rejected = _empty_rejected(original.columns)
    return CleanResult(accepted=accepted, rejected=rejected)


def clean_carriers(df: pd.DataFrame) -> CleanResult:
    source = "carriers.csv"
    required = ["carrier_id", "name", "hq_country", "service_tier"]
    original = df.copy()
    working = df.copy()
    keep = pd.Series(True, index=working.index)
    rejected: list[pd.DataFrame] = []

    for col in required:
        if col not in working.columns:
            raise ValueError(f"{source}: missing required column '{col}'")
        keep &= _reject(
            source, original, keep & working[col].isna(), f"null_{col}", rejected
        )

    working["carrier_id"] = working["carrier_id"].astype(str).str.strip()
    working["name"] = working["name"].astype(str).str.strip()
    working["hq_country"] = working["hq_country"].astype(str).str.strip().str.upper()
    working["service_tier"] = working["service_tier"].astype(str).str.strip().str.lower()

    keep &= _reject(
        source,
        original,
        keep & (working["carrier_id"] == ""),
        "empty_carrier_id",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["hq_country"].str.fullmatch(r"[A-Z]{2}", na=False),
        "invalid_hq_country",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["service_tier"].isin(VALID_SERVICE_TIERS),
        "invalid_service_tier",
        rejected,
    )

    dup = keep & working["carrier_id"].duplicated(keep="first")
    keep &= _reject(source, original, dup, "duplicate_carrier_id", rejected)

    return _finalize(source, working, original, keep, rejected)


def clean_contract_rates(df: pd.DataFrame) -> CleanResult:
    source = "contract_rates.csv"
    required = [
        "carrier_id",
        "country",
        "weight_band",
        "min_kg",
        "max_kg",
        "rate_eur",
        "currency",
        "effective_from",
    ]
    original = df.copy()
    working = df.copy()
    keep = pd.Series(True, index=working.index)
    rejected: list[pd.DataFrame] = []

    for col in required:
        if col not in working.columns:
            raise ValueError(f"{source}: missing required column '{col}'")

    for col in ("carrier_id", "country", "weight_band", "currency", "effective_from"):
        keep &= _reject(
            source, original, keep & working[col].isna(), f"null_{col}", rejected
        )

    keep &= _reject(
        source, original, keep & working["min_kg"].isna(), "null_min_kg", rejected
    )
    keep &= _reject(
        source, original, keep & working["rate_eur"].isna(), "null_rate_eur", rejected
    )

    working["carrier_id"] = working["carrier_id"].astype(str).str.strip()
    working["country"] = working["country"].astype(str).str.strip().str.upper()
    working["weight_band"] = working["weight_band"].astype(str).str.strip()
    working["currency"] = working["currency"].astype(str).str.strip().str.upper()
    working["min_kg"] = pd.to_numeric(working["min_kg"], errors="coerce")
    working["max_kg"] = pd.to_numeric(working["max_kg"], errors="coerce")
    working["rate_eur"] = pd.to_numeric(working["rate_eur"], errors="coerce")
    working["effective_from"] = pd.to_datetime(
        working["effective_from"], errors="coerce", format="mixed"
    )

    keep &= _reject(
        source,
        original,
        keep & working["min_kg"].isna(),
        "invalid_min_kg_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & working["rate_eur"].isna(),
        "invalid_rate_eur_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & working["effective_from"].isna(),
        "invalid_effective_from",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["min_kg"] < 0),
        "negative_min_kg",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["rate_eur"] <= 0),
        "non_positive_rate_eur",
        rejected,
    )
    # Open-ended 50+ band has null max_kg; otherwise max must be > min.
    bad_max = keep & working["max_kg"].notna() & (working["max_kg"] <= working["min_kg"])
    keep &= _reject(source, original, bad_max, "max_kg_not_above_min_kg", rejected)
    keep &= _reject(
        source,
        original,
        keep & ~working["country"].str.fullmatch(r"[A-Z]{2}", na=False),
        "invalid_country",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["weight_band"].isin(VALID_WEIGHT_BANDS),
        "invalid_weight_band",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["currency"] != "EUR"),
        "unsupported_currency",
        rejected,
    )

    key_cols = ["carrier_id", "country", "weight_band", "effective_from"]
    dup = keep & working.duplicated(subset=key_cols, keep="first")
    keep &= _reject(source, original, dup, "duplicate_contract_rate_key", rejected)

    return _finalize(source, working, original, keep, rejected)


def clean_shipments(df: pd.DataFrame) -> CleanResult:
    source = "shipments.csv"
    required = [
        "id",
        "date",
        "carrier",
        "country",
        "weight_kg",
        "parcel_type",
        "status",
        "delivery_days",
    ]
    original = df.copy()
    working = df.copy()
    keep = pd.Series(True, index=working.index)
    rejected: list[pd.DataFrame] = []

    for col in required:
        if col not in working.columns:
            raise ValueError(f"{source}: missing required column '{col}'")
        keep &= _reject(
            source, original, keep & working[col].isna(), f"null_{col}", rejected
        )

    working["id"] = working["id"].astype(str).str.strip()
    working["carrier"] = working["carrier"].astype(str).str.strip()
    working["country"] = working["country"].astype(str).str.strip().str.upper()
    working["parcel_type"] = working["parcel_type"].astype(str).str.strip().str.lower()
    working["status"] = working["status"].astype(str).str.strip().str.lower()
    working["weight_kg"] = pd.to_numeric(working["weight_kg"], errors="coerce")
    working["delivery_days"] = pd.to_numeric(working["delivery_days"], errors="coerce")
    working["date"] = pd.to_datetime(working["date"], errors="coerce", format="mixed")

    keep &= _reject(
        source, original, keep & (working["id"] == ""), "empty_id", rejected
    )
    keep &= _reject(
        source, original, keep & working["date"].isna(), "invalid_date", rejected
    )
    keep &= _reject(
        source,
        original,
        keep & working["weight_kg"].isna(),
        "invalid_weight_kg_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["weight_kg"] <= 0),
        "non_positive_weight_kg",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & working["delivery_days"].isna(),
        "invalid_delivery_days_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["delivery_days"] < 0),
        "negative_delivery_days",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["country"].str.fullmatch(r"[A-Z]{2}", na=False),
        "invalid_country",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["parcel_type"].isin(VALID_PARCEL_TYPES),
        "invalid_parcel_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & ~working["status"].isin(VALID_STATUSES),
        "invalid_status",
        rejected,
    )

    dup = keep & working["id"].duplicated(keep="first")
    keep &= _reject(source, original, dup, "duplicate_shipment_id", rejected)

    working.loc[keep, "delivery_days"] = working.loc[keep, "delivery_days"].astype(int)
    return _finalize(source, working, original, keep, rejected)


def clean_invoices(df: pd.DataFrame) -> CleanResult:
    """Clean public invoice rows.

    Deliberate business faults (orphan shipment_id, multi-invoice shipments) are
    kept — only structural / type / domain violations are rejected.
    """
    source = "carrier_invoices.csv"
    required = [
        "invoice_id",
        "shipment_id",
        "carrier",
        "billed_price",
        "invoice_date",
    ]
    original = df.copy()
    working = df.copy()
    keep = pd.Series(True, index=working.index)
    rejected: list[pd.DataFrame] = []

    for col in required:
        if col not in working.columns:
            raise ValueError(f"{source}: missing required column '{col}'")
        keep &= _reject(
            source, original, keep & working[col].isna(), f"null_{col}", rejected
        )

    working["invoice_id"] = working["invoice_id"].astype(str).str.strip()
    working["shipment_id"] = working["shipment_id"].astype(str).str.strip()
    working["carrier"] = working["carrier"].astype(str).str.strip()
    working["billed_price"] = pd.to_numeric(working["billed_price"], errors="coerce")
    working["invoice_date"] = pd.to_datetime(
        working["invoice_date"], errors="coerce", format="mixed"
    )

    keep &= _reject(
        source,
        original,
        keep & (working["invoice_id"] == ""),
        "empty_invoice_id",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["shipment_id"] == ""),
        "empty_shipment_id",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & working["billed_price"].isna(),
        "invalid_billed_price_type",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & (working["billed_price"] < 0),
        "negative_billed_price",
        rejected,
    )
    keep &= _reject(
        source,
        original,
        keep & working["invoice_date"].isna(),
        "invalid_invoice_date",
        rejected,
    )

    dup = keep & working["invoice_id"].duplicated(keep="first")
    keep &= _reject(source, original, dup, "duplicate_invoice_id", rejected)

    return _finalize(source, working, original, keep, rejected)


def combine_rejected(parts: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [p for p in parts if len(p)]
    if not nonempty:
        return pd.DataFrame(columns=["_source", "_reason", "_row_number"])
    return pd.concat(nonempty, ignore_index=True)
