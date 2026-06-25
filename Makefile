PYTHON ?= .venv/bin/python

.PHONY: ingest

ingest:
	$(PYTHON) -m helpdeskai.ingestion.pipeline