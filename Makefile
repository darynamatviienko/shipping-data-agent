.PHONY: setup data etl test ask eval run

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

# Step 6: live agent scoring when .env has API keys; else SQL ground truth only.
# Pass EXTRA='--live' to require keys, or EXTRA='--ground-truth-only' to skip LLM.
eval:
	$(PYTHON) scripts/run_eval.py --update-readme $(EXTRA)

# Step 7: Streamlit chat UI (requires .env API key + warehouse from make etl).
run:
	.venv/bin/streamlit run scripts/streamlit_app.py
