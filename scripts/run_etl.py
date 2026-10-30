#!/usr/bin/env python3
"""CLI entrypoint: load data/raw CSVs into the DuckDB warehouse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shipping_data_agent.etl import run_etl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=root / "data" / "raw",
        help="Directory with source CSVs (default: data/raw)",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=root / "data" / "warehouse" / "shipping.duckdb",
        help="DuckDB warehouse path (default: data/warehouse/shipping.duckdb)",
    )
    parser.add_argument(
        "--rejected-path",
        type=Path,
        default=root / "data" / "warehouse" / "rejected_rows.csv",
        help="CSV for rejected rows (default: data/warehouse/rejected_rows.csv)",
    )
    args = parser.parse_args()

    summary = run_etl(
        raw_dir=args.raw_dir,
        db_path=args.db_path,
        rejected_path=args.rejected_path,
    )

    print("ETL complete.")
    print(f"  db:        {summary['db_path']}")
    print(f"  rejected:  {summary['n_rejected']} rows → {summary['rejected_path']}")
    for table, n in summary["table_counts"].items():
        print(f"  {table}: {n}")
    print(json.dumps({"accepted": summary["accepted"], "rejected_by_source": summary["rejected_by_source"]}))


if __name__ == "__main__":
    main()
