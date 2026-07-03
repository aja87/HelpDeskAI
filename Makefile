PYTHON ?= .venv/bin/python

.PHONY: ingest

ingest:
	PYTHONPATH=. $(PYTHON) scripts/run_ingestion.py