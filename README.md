# HelpDeskAI

Projet fil rouge du cursus AI Engineer Neosoft.

L'objectif est de concevoir et industrialiser un assistant de support N1
augmente par RAG et agents pour le contexte fictif NovaCloud.

## Prerequis

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker avec Docker Compose

## Installation

```bash
uv sync --dev
```

## Preparation des corpus

La preparation des corpus est separee en scripts independants. Le
telechargement n'est pas inclus dans le flow Prefect d'ingestion.

### 1. Telecharger les donnees sources

Le script telecharge les sources necessaires au projet :

- TechQA documents, utilises pour l'indexation vectorielle ;
- TechQA Q/A, reserve au golden dataset et aux evaluations RAG ;
- Bitext, reserve aux tests d'intention et demonstrations ;
- MSDialog, reserve aux tests agent multi-tours.

```bash
uv run python scripts/download_corpus.py
```

Les fichiers JSONL et le manifeste sont produits sous `data/raw/` :

```text
data/raw/techqa/documents.jsonl
data/raw/techqa/qa.jsonl
data/raw/bitext/tickets.jsonl
data/raw/msdialog/conversations.jsonl
data/raw/manifest.json
```

La graine vaut `42` par defaut. Les principales options sont :

```bash
uv run python scripts/download_corpus.py --seed 123 --data-dir data/raw
uv run python scripts/download_corpus.py --skip-existing
uv run python scripts/download_corpus.py --force
```

Sans `--force`, le script refuse d'ecraser des sorties existantes.

### 2. Analyser les corpus telecharges

Le script d'analyse charge les quatre fichiers JSONL avec Pandas et exporte
les statistiques, doublons et distributions sous
`docs/corpus_preparation/` :

```bash
uv run python scripts/analyze_corpus.py
```

### 3. Preparer le corpus indexable TechQA

Le flow Prefect part de `data/raw/techqa/documents.jsonl`, normalise et enrichit
uniquement les documents TechQA, applique le chunking recursive retenu,
deduplique les chunks, puis genere automatiquement un rapport qualite Evidently :

```bash
uv run python scripts/prepare_corpus.py --force
```

Il produit sous `data/processed/techqa/` :

- `documents.jsonl` : documents canoniques enrichis ;
- `chunks.jsonl` : chunks recursive dedupliques, prets a indexer ;
- `manifest.json` : statistiques et configuration du chunking.

Il produit egalement sous `docs/corpus_preparation/` :

- `docs/corpus_preparation/corpus_quality_report.html` ;
- `docs/corpus_preparation/corpus_quality_summary.json`.

Le flow accepte `--skip-existing` et `--force`.

Les étapes de la pipeline:


```text
extract -> normalize -> document dedup -> enrich -> chunk -> chunk dedup
-> persist -> quality
```

Les Q/A TechQA, Bitext et MSDialog ne sont pas transformes par ce pipeline.
Ils restent des donnees d'evaluation ou de demonstration pour les modules
suivants.

### 4. Comparer les strategies de chunking

La comparaison des chunkings est un script independant. Elle utilise les
documents prepares, compare `fixed`, `recursive` et `semantic`, et charge
localement `BAAI/bge-m3` pour le chunking semantique :

```bash
uv run python scripts/compare_chunking.py --force
```

Sorties :

- `docs/corpus_preparation/chunking_benchmark.json` ;
- `docs/corpus_preparation/chunking_benchmark.md` ;
- `docs/corpus_preparation/chunking_comparison.png`.

Les indicateurs compares sont structurels : nombre de chunks, distribution des
tokens, chunks trop petits ou trop grands, doublons exacts et temps d'execution.
La qualite retrieval/RAG sera mesuree dans les modules suivants, quand l'index et
les jeux d'evaluation seront disponibles.

## Retrieval

Le module retrieval indexe les chunks TechQA prepares dans Qdrant et pgvector,
puis expose une recherche dense, sparse ou hybride.

Demarrer les services necessaires :

```bash
docker compose --profile retrieval up -d qdrant pgvector
```

Indexer le corpus prepare :

```bash
uv run python scripts/index_retrieval.py
```

Parametres principaux :

```bash
uv run python scripts/index_retrieval.py --collection helpdeskai_techqa_chunks
uv run python scripts/index_retrieval.py --no-pgvector
uv run python scripts/index_retrieval.py --no-qdrant
uv run python scripts/index_retrieval.py --append
```

Le comportement par defaut :

- lit `data/processed/techqa/chunks.jsonl` ;
- encode les chunks avec `BAAI/bge-m3` ;
- recrée la collection Qdrant `helpdeskai_techqa_chunks` ;
- alimente aussi la table pgvector `retrieval_chunks` ;
- conserve les metadonnees utiles au filtrage : produit, version, date, categorie
  et tenant.

La recherche publique est exposee par `helpdeskai.retrieval.search.search(...)`.
Elle supporte trois modes :

- `dense` : recherche vectorielle Qdrant ;
- `sparse` : BM25 local en memoire ;
- `hybrid` : fusion dense + sparse par Reciprocal Rank Fusion.

Exemple Python :

```python
from helpdeskai.retrieval.search import search

results = search("How do I configure SAML?", top_k=5, mode="hybrid")
```

Benchmark retrieval :

```bash
uv run python scripts/benchmark_retrieval.py
```

Le benchmark lit `tests/golden/questions.jsonl` et produit :

- `reports/retrieval/benchmark_results.csv` ;
- `reports/retrieval/benchmark_report.md`.

Les indicateurs mesures sont `Recall@5`, `Recall@10`, `MRR`, `p50_ms` et
`p95_ms`, pour les modes dense, sparse et hybride.

## RAG avance et evaluation

Le module RAG s'appuie sur le retrieval existant et ajoute :

- query rewriting avec Claude ;
- retrieval dense/sparse/hybride via `helpdeskai.retrieval` ;
- re-ranking avec `BAAI/bge-reranker-v2-m3` ;
- compression contextuelle des chunks retenus ;
- generation Claude avec citations `[chunk_id]` ;
- evaluation Ragas sur les questions TechQA eligibles.

Configurer la cle Anthropic :

```bash
$env:ANTHROPIC_API_KEY = "..."
```

Executer une question :

```bash
uv run python scripts/run_rag.py --question "How do I configure SAML?"
```

Options utiles :

```bash
uv run python scripts/run_rag.py --prompt strict --mode hybrid
uv run python scripts/run_rag.py --prompt pedagogical --final-k 5
uv run python scripts/run_rag.py --questions-file questions.txt
```

Comparer les trois prompts avec Ragas :

```bash
uv run python scripts/evaluate_rag.py --limit 5
```

Sorties sous `reports/rag/` :

- `rag_results_<prompt>.jsonl` ;
- `ragas_results_<prompt>.csv` ;
- `ragas_comparison.md` ;
- `ragas_comparison.json`.

Les prompts versionnes sont :

- `strict` : citations obligatoires, refus si contexte insuffisant ;
- `pedagogical` : explication support simple avec citations ;
- `concise` : reponse operationnelle courte avec citations.

La CI execute uniquement les tests offline. L'evaluation Ragas reelle appelle
Claude et doit etre lancee manuellement.

## Agent et orchestration

Le module `helpdeskai.agents` expose un agent LangGraph de support N1. Le graphe
enchaine classification d'intention metier par LLM, controle de budget,
clarification si la demande est ambigue, appel au RAG pour les questions
documentees et escalade avec validation humaine pour les actions sensibles. Les
intentions metier (`technical_question`, `crm_question`, etc.) sont separees
des routes internes (`answer_with_rag`, `sensitive_action`, `clarification`).

Demo simple :

```bash
uv run python scripts/run_agent.py --question "How do I configure SAML login in NovaCloud?"
```

Demo human-in-the-loop avec checkpoint SQLite :

```bash
uv run python scripts/run_agent.py --thread-id ticket-1 --question "Create an escalation ticket for this login issue"
uv run python scripts/run_agent.py --thread-id ticket-1 --approve
```

Exporter la visualisation Mermaid du graphe :

```bash
uv run python scripts/run_agent.py --export-mermaid docs/agent_graph.mmd
```

La comparaison LangGraph vs CrewAI est documentee dans
`docs/agents_langgraph_vs_crewai.md`.

## Validation

```bash
uv run ruff check .
uv run pytest
```

## Donnees et artefacts generes

Les donnees brutes, donnees traitees, caches de modeles et rapports generes ne
doivent pas etre versionnes, sauf artefacts explicitement demandes dans
`docs/corpus_preparation/` ou `tests/golden/`.


