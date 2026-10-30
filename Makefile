.PHONY: setup data etl test ask run

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

# Example: make ask Q='How many shipments were delivered to Germany?'
ask:
	@test -n "$(Q)" || (echo 'Usage: make ask Q="your question"'; exit 1)
	$(PYTHON) scripts/ask_agent.py "$(Q)"

run:
	@echo "Step 7 will implement the full CLI/UI; for now use: make ask Q='...'"
