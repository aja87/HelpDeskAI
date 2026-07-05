PYTHON ?= .venv/bin/python

.PHONY: ingest index

ingest:
	PYTHONPATH=. $(PYTHON) scripts/run_ingestion.py

index:
	PYTHONPATH=. $(PYTHON) scripts/run_indexing.py