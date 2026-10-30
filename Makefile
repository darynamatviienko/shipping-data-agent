.PHONY: setup data etl test run

PYTHON := .venv/bin/python
export PYTHONPATH := src

setup:
	@test -d .venv || python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) scripts/generate_synthetic_data.py

etl:
	$(PYTHON) scripts/run_etl.py

test:
	.venv/bin/pytest

run:
	@echo "Step 7 will implement the CLI/UI"
