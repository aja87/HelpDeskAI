PYTHON ?= .venv/bin/python

.PHONY: ingest index rag evaluate

ingest:
	PYTHONPATH=. $(PYTHON) scripts/run_ingestion.py

index:
	PYTHONPATH=. $(PYTHON) scripts/run_indexing.py

rag:
	PYTHONPATH=. $(PYTHON) scripts/run_rag.py answer --query "How do I reset my MQ queue manager password?"

evaluate:
	PYTHONPATH=. $(PYTHON) scripts/run_rag.py evaluate