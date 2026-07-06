# HelpDeskAI

[![CI](https://github.com/aja87/HelpDeskAI/actions/workflows/ci.yml/badge.svg)](https://github.com/aja87/HelpDeskAI/actions/workflows/ci.yml)


HelpDeskAI est un assistant local de support N1 pour le contexte fictif
NovaCloud. Il combine retrieval, RAG, agent LangGraph, outils MCP CRM/Knowledge
et observabilite MLflow/Langfuse.

## Fonctionnalites

- Preparation et indexation d'un corpus documentaire TechQA.
- Recherche dense Qdrant, sparse BM25 et hybride.
- Reponses RAG avec citations de chunks.
- Agent support avec appels MCP en chaine.
- Validation humaine avant action sensible.
- Demo Streamlit locale.
- Tracking MLflow, traces Langfuse et estimation FinOps.

## Prerequis

- Python 3.11
- `uv`
- Docker avec Docker Compose
- `ANTHROPIC_API_KEY` pour les executions avec Claude

## Demarrage Rapide

Installer les dependances :

```bash
uv sync --dev
```

Demarrer la stack locale :

```bash
docker compose up --build -d
```

Services principaux :

- Demo Streamlit : http://localhost:8501
- Qdrant : http://localhost:6333
- MLflow : http://127.0.0.1:5000
- Langfuse : http://localhost:3000
- MinIO Langfuse : http://localhost:9091

Arreter la stack :

```bash
docker compose down
```

## Configuration

Variables utiles :

```powershell
$env:ANTHROPIC_API_KEY = "..."
$env:HELPDESKAI_MCP_TOKEN = "helpdeskai-dev-token"
$env:QDRANT_URL = "http://localhost:6333"
$env:PGVECTOR_DSN = "postgresql://postgres:postgres@localhost:5433/helpdeskai"
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
$env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
$env:LANGFUSE_SECRET_KEY = "sk-lf-..."
$env:LANGFUSE_HOST = "http://localhost:3000"
```

## Donnees Et Indexation

```bash
uv run python scripts/download_corpus.py
uv run python scripts/analyze_corpus.py
uv run python scripts/prepare_corpus.py --force
uv run python scripts/index_retrieval.py
```

Pour indexer uniquement Qdrant :

```bash
uv run python scripts/index_retrieval.py --no-pgvector
```

Benchmark retrieval :

```bash
uv run python scripts/benchmark_retrieval.py
```

Le golden dataset contient 100 questions : 75 cas TechQA alignes avec les
documents indexes pour le retrieval/RAG, et 25 cas Bitext pour les intents
support non-retrieval.

## Utilisation

Interface Streamlit :

```bash
uv run streamlit run scripts/demo_streamlit.py
```

Question RAG directe :

```bash
uv run python scripts/run_rag.py --question "How do I configure SAML?"
```

Agent avec MCP :

```bash
uv run python scripts/run_agent.py --question "Quel est le statut de cust_acme ?"
```

Action sensible avec validation humaine :

```bash
uv run python scripts/run_agent.py --thread-id ticket-1 --question "Escalade le compte cust_acme pour acces admin bloque"
uv run python scripts/run_agent.py --thread-id ticket-1 --approve
```

Demo MCP deterministe sans appel LLM :

```bash
uv run python scripts/demo_agent_mcp.py
```

## Observabilite

Evaluation RAG loggee dans MLflow :

```bash
uv run python scripts/evaluate_rag.py --prompt strict --limit 2 --mlflow-tracking-uri http://127.0.0.1:5000
```

Registry de prompts :

```bash
uv run python scripts/register_prompts.py --promote-from-eval
```

Registry du modele RAG pyfunc :

```bash
uv run python scripts/register_rag_model.py --tracking-uri http://127.0.0.1:5000
```

Trace agent Langfuse :

```bash
uv run python scripts/run_agent_langfuse.py
```

Rapports FinOps :

```bash
uv run python scripts/finops_dashboard.py --csv reports/finops/scenarios.csv
```

## Documentation

- Architecture : `docs/ARCHITECTURE.md`
- Graphe agent : `docs/agent_graph.mmd`
- Outils MCP : `docs/mcp_tools.md`
- FinOps : `docs/FINOPS.md`

## Validation

```bash
uv run ruff check .
uv run pytest
docker compose config --services
```
