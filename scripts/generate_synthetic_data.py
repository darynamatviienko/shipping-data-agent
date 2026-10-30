#!/usr/bin/env python3
"""Generate synthetic shipping CSVs with deliberate invoice errors (~3%).

Outputs under data/raw/:
  carriers.csv
  contract_rates.csv
  shipments.csv          (~20,000 rows)
  carrier_invoices.csv

Reproducible via FIXED_SEED. Only dependency: pandas (see requirements.txt).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FIXED_SEED = 42
N_SHIPMENTS = 20_000
ERROR_RATE = 0.03

CARRIERS = [
    {"carrier_id": "DHL", "name": "DHL Express", "hq_country": "DE", "service_tier": "express"},
    {"carrier_id": "UPS", "name": "UPS Worldwide", "hq_country": "US", "service_tier": "express"},
    {"carrier_id": "FDX", "name": "FedEx International", "hq_country": "US", "service_tier": "express"},
    {"carrier_id": "TNT", "name": "TNT Express", "hq_country": "NL", "service_tier": "standard"},
    {"carrier_id": "GLS", "name": "GLS Parcel", "hq_country": "DE", "service_tier": "economy"},
]

COUNTRIES = ["DE", "FR", "NL", "BE", "PL", "IT", "ES", "GB", "US", "CZ"]
PARCEL_TYPES = ["envelope", "small_box", "medium_box", "large_box", "pallet"]
STATUSES = ["delivered", "in_transit", "delayed", "returned", "cancelled"]
STATUS_WEIGHTS = [0.72, 0.14, 0.07, 0.04, 0.03]

# Weight bands used by contract rates (kg). Wrong-weight-class errors bill a mismatched band.
WEIGHT_BANDS: list[tuple[str, float, float | None]] = [
    ("0-1", 0.0, 1.0),
    ("1-5", 1.0, 5.0),
    ("5-20", 5.0, 20.0),
    ("20-50", 20.0, 50.0),
    ("50+", 50.0, None),
]

# Base EUR rates by (service_tier, weight_band) before country multipliers.
BASE_RATES: dict[tuple[str, str], float] = {
    ("express", "0-1"): 8.50,
    ("express", "1-5"): 14.00,
    ("express", "5-20"): 28.00,
    ("express", "20-50"): 55.00,
    ("express", "50+"): 95.00,
    ("standard", "0-1"): 6.00,
    ("standard", "1-5"): 10.50,
    ("standard", "5-20"): 21.00,
    ("standard", "20-50"): 42.00,
    ("standard", "50+"): 75.00,
    ("economy", "0-1"): 4.50,
    ("economy", "1-5"): 7.50,
    ("economy", "5-20"): 15.00,
    ("economy", "20-50"): 32.00,
    ("economy", "50+"): 58.00,
}

COUNTRY_MULTIPLIER = {
    "DE": 1.00,
    "NL": 1.02,
    "BE": 1.05,
    "FR": 1.08,
    "PL": 0.95,
    "CZ": 0.93,
    "IT": 1.12,
    "ES": 1.10,
    "GB": 1.18,
    "US": 1.35,
}

PARCEL_TYPE_WEIGHT_RANGE = {
    "envelope": (0.05, 0.8),
    "small_box": (0.5, 4.5),
    "medium_box": (3.0, 18.0),
    "large_box": (12.0, 45.0),
    "pallet": (40.0, 120.0),
}

ERROR_TYPES = (
    "duplicate_billing",
    "wrong_weight_class",
    "price_above_contract",
    "nonexistent_shipment",
)


def weight_band(weight_kg: float) -> str:
    for name, lo, hi in WEIGHT_BANDS:
        if hi is None:
            if weight_kg >= lo:
                return name
        elif lo <= weight_kg < hi:
            return name
    return WEIGHT_BANDS[-1][0]


def contract_price(carrier_id: str, country: str, weight_kg: float, carriers_df: pd.DataFrame) -> float:
    tier = carriers_df.set_index("carrier_id").loc[carrier_id, "service_tier"]
    band = weight_band(weight_kg)
    base = BASE_RATES[(tier, band)]
    # Small per-kg add-on within band so rates look realistic.
    add_on = max(0.0, weight_kg - next(lo for name, lo, _ in WEIGHT_BANDS if name == band)) * 0.35
    price = (base + add_on) * COUNTRY_MULTIPLIER[country]
    return round(price, 2)


def build_carriers() -> pd.DataFrame:
    return pd.DataFrame(CARRIERS)


def build_contract_rates(carriers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for _, carrier in carriers.iterrows():
        for country in COUNTRIES:
            for band_name, lo, hi in WEIGHT_BANDS:
                base = BASE_RATES[(carrier["service_tier"], band_name)]
                # Mid-band reference weight for the published contract line.
                if hi is None:
                    ref_kg = lo + 10.0
                else:
                    ref_kg = (lo + hi) / 2.0
                rate = round((base + max(0.0, ref_kg - lo) * 0.35) * COUNTRY_MULTIPLIER[country], 2)
                rows.append(
                    {
                        "carrier_id": carrier["carrier_id"],
                        "country": country,
                        "weight_band": band_name,
                        "min_kg": lo,
                        "max_kg": hi if hi is not None else "",
                        "rate_eur": rate,
                        "currency": "EUR",
                        "effective_from": "2024-01-01",
                    }
                )
    return pd.DataFrame(rows)


def build_shipments(rng: np.random.Generator, carriers: pd.DataFrame) -> pd.DataFrame:
    carrier_ids = carriers["carrier_id"].tolist()
    start = np.datetime64("2024-01-01")
    day_offsets = rng.integers(0, 730, size=N_SHIPMENTS)

    parcel_types = rng.choice(PARCEL_TYPES, size=N_SHIPMENTS, p=[0.12, 0.28, 0.30, 0.22, 0.08])
    weights = np.empty(N_SHIPMENTS, dtype=float)
    for i, pt in enumerate(parcel_types):
        lo, hi = PARCEL_TYPE_WEIGHT_RANGE[pt]
        weights[i] = round(float(rng.uniform(lo, hi)), 2)

    statuses = rng.choice(STATUSES, size=N_SHIPMENTS, p=STATUS_WEIGHTS)
    delivery_days = np.where(
        statuses == "cancelled",
        0,
        rng.integers(1, 12, size=N_SHIPMENTS),
    )
    # Delayed shipments tend to take longer.
    delayed_mask = statuses == "delayed"
    delivery_days[delayed_mask] = rng.integers(8, 21, size=int(delayed_mask.sum()))

    return pd.DataFrame(
        {
            "id": [f"SHP-{i:06d}" for i in range(1, N_SHIPMENTS + 1)],
            "date": (start + day_offsets.astype("timedelta64[D]")).astype(str),
            "carrier": rng.choice(carrier_ids, size=N_SHIPMENTS),
            "country": rng.choice(COUNTRIES, size=N_SHIPMENTS),
            "weight_kg": weights,
            "parcel_type": parcel_types,
            "status": statuses,
            "delivery_days": delivery_days.astype(int),
        }
    )


def build_invoices(
    rng: np.random.Generator,
    shipments: pd.DataFrame,
    carriers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Build invoices for billable shipments, then inject ~3% deliberate errors."""
    billable = shipments[shipments["status"] != "cancelled"].reset_index(drop=True)
    ship_by_id = shipments.set_index("id")

    prices = [
        contract_price(c, country, float(w), carriers)
        for c, country, w in zip(
            billable["carrier"], billable["country"], billable["weight_kg"], strict=True
        )
    ]
    noise = rng.uniform(0.985, 1.015, size=len(billable))
    billed = np.round(np.asarray(prices, dtype=float) * noise, 2)
    lags = rng.integers(1, 21, size=len(billable))
    ship_dates = billable["date"].to_numpy(dtype="datetime64[D]")
    invoice_dates = (ship_dates + lags.astype("timedelta64[D]")).astype(str)

    invoice_rows: list[dict] = [
        {
            "invoice_id": "",
            "shipment_id": sid,
            "carrier": carrier,
            "billed_price": float(price),
            "invoice_date": inv_date,
            "error_type": "",
        }
        for sid, carrier, price, inv_date in zip(
            billable["id"], billable["carrier"], billed, invoice_dates, strict=True
        )
    ]

    n_base = len(invoice_rows)
    n_errors = max(1, int(round(n_base * ERROR_RATE)))
    per_type = n_errors // len(ERROR_TYPES)
    remainder = n_errors - per_type * len(ERROR_TYPES)
    quotas = {et: per_type for et in ERROR_TYPES}
    for et in ERROR_TYPES[:remainder]:
        quotas[et] += 1

    clean_indices = list(range(n_base))
    rng.shuffle(clean_indices)
    cursor = 0

    def take(n: int) -> list[int]:
        nonlocal cursor
        chosen = clean_indices[cursor : cursor + n]
        cursor += n
        return chosen

    # --- duplicate_billing: second invoice for the same shipment ---
    for idx in take(quotas["duplicate_billing"]):
        dup = dict(invoice_rows[idx])
        ship_date = np.datetime64(dup["invoice_date"])
        dup["invoice_date"] = (ship_date + np.timedelta64(int(rng.integers(3, 15)), "D")).astype(str)
        dup["error_type"] = "duplicate_billing"
        invoice_rows[idx]["error_type"] = "duplicate_billing"
        invoice_rows.append(dup)

    # --- wrong_weight_class: bill a mismatched weight band ---
    for idx in take(quotas["wrong_weight_class"]):
        row = invoice_rows[idx]
        ship = ship_by_id.loc[row["shipment_id"]]
        true_band = weight_band(float(ship["weight_kg"]))
        other_bands = [b for b, _, _ in WEIGHT_BANDS if b != true_band]
        wrong_band = str(rng.choice(other_bands))
        lo, hi = next((lo, hi) for name, lo, hi in WEIGHT_BANDS if name == wrong_band)
        fake_kg = (lo + 10.0) if hi is None else (lo + hi) / 2.0
        row["billed_price"] = contract_price(row["carrier"], ship["country"], fake_kg, carriers)
        row["error_type"] = "wrong_weight_class"

    # --- price_above_contract: clear markup over contract ---
    for idx in take(quotas["price_above_contract"]):
        row = invoice_rows[idx]
        ship = ship_by_id.loc[row["shipment_id"]]
        correct = contract_price(row["carrier"], ship["country"], float(ship["weight_kg"]), carriers)
        markup = float(rng.uniform(1.15, 1.45))
        row["billed_price"] = round(correct * markup, 2)
        row["error_type"] = "price_above_contract"

    # --- nonexistent_shipment: invoice IDs absent from shipments ---
    for i in range(quotas["nonexistent_shipment"]):
        fake_id = f"SHP-{N_SHIPMENTS + 1000 + i:06d}"
        carrier = str(rng.choice(carriers["carrier_id"].tolist()))
        country = str(rng.choice(COUNTRIES))
        fake_weight = float(rng.uniform(0.5, 40.0))
        price = contract_price(carrier, country, fake_weight, carriers)
        day = int(rng.integers(0, 730))
        invoice_date = (np.datetime64("2024-01-01") + np.timedelta64(day, "D")).astype(str)
        invoice_rows.append(
            {
                "invoice_id": "",
                "shipment_id": fake_id,
                "carrier": carrier,
                "billed_price": price,
                "invoice_date": invoice_date,
                "error_type": "nonexistent_shipment",
            }
        )

    invoices = pd.DataFrame(invoice_rows)
    invoices = invoices.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    invoices["invoice_id"] = [f"INV-{i:06d}" for i in range(1, len(invoices) + 1)]

    public = invoices[["invoice_id", "shipment_id", "carrier", "billed_price", "invoice_date"]].copy()
    summary = {
        "base_invoices": n_base,
        "total_invoices": len(public),
        "target_error_rate": ERROR_RATE,
        "n_error_rows": int((invoices["error_type"] != "").sum()),
        **{f"n_{et}": int((invoices["error_type"] == et).sum()) for et in ERROR_TYPES},
    }
    return public, summary, invoices


def generate(output_dir: Path) -> dict:
    rng = np.random.default_rng(FIXED_SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    carriers = build_carriers()
    contract_rates = build_contract_rates(carriers)
    shipments = build_shipments(rng, carriers)
    invoices, summary, labeled = build_invoices(rng, shipments, carriers)

    carriers.to_csv(output_dir / "carriers.csv", index=False)
    contract_rates.to_csv(output_dir / "contract_rates.csv", index=False)
    shipments.to_csv(output_dir / "shipments.csv", index=False)
    invoices.to_csv(output_dir / "carrier_invoices.csv", index=False)
    # Labels kept out of the public schema but available for Step 6 eval scaffolding.
    labeled.to_csv(output_dir / "carrier_invoices_labeled.csv", index=False)

    summary.update(
        {
            "n_carriers": len(carriers),
            "n_contract_rates": len(contract_rates),
            "n_shipments": len(shipments),
            "seed": FIXED_SEED,
            "output_dir": str(output_dir),
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw",
        help="Directory for CSV outputs (default: data/raw)",
    )
    args = parser.parse_args()
    summary = generate(args.output_dir)

    print("Synthetic shipping data written.")
    print(f"  seed:              {summary['seed']}")
    print(f"  output:            {summary['output_dir']}")
    print(f"  carriers:          {summary['n_carriers']}")
    print(f"  contract_rates:    {summary['n_contract_rates']}")
    print(f"  shipments:         {summary['n_shipments']}")
    print(f"  invoices (total):  {summary['total_invoices']}")
    print(f"  base (pre-inject): {summary['base_invoices']}")
    print(f"  error rows:        {summary['n_error_rows']} "
          f"({summary['n_error_rows'] / summary['total_invoices']:.1%} of invoices)")
    for et in ERROR_TYPES:
        print(f"    {et}: {summary[f'n_{et}']}")


if __name__ == "__main__":
    main()
