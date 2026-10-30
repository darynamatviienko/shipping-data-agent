"""ETL: clean raw CSVs and load a DuckDB star-schema warehouse."""

from shipping_data_agent.etl.pipeline import run_etl

__all__ = ["run_etl"]
