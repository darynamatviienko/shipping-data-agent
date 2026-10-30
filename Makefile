.PHONY: setup etl test run

setup:
	@test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

etl:
	@echo "Step 3 will implement ETL"

test:
	.venv/bin/pytest

run:
	@echo "Step 7 will implement the CLI/UI"
