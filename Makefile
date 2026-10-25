PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
export PYTHONPATH := src

.PHONY: install profile ingest-sample ingest-full inspect test

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

profile:
	$(PY) scripts/profile_data.py

ingest-sample:
	$(PY) scripts/ingest.py --sample 1000

ingest-full:
	$(PY) scripts/ingest.py

inspect:
	$(PY) scripts/inspect_collection.py

test:
	$(PY) -m pytest
