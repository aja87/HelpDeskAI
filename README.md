# HelpDeskAI

Projet fil rouge du cursus AI Engineer Neosoft.

L'objectif est de concevoir et industrialiser un assistant de support N1
augmente par RAG et agents pour le contexte fictif NovaCloud.

Le depot contient uniquement le squelette initial. Les pipelines d'ingestion,
de retrieval, de RAG, les agents, les serveurs MCP, l'observabilite et les
tests restent a implementer au fil des phases du projet.

## Prerequis

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- Docker avec Docker Compose

## Installation

```bash
uv sync --dev
```

## Telechargement des corpus

Le script telecharge 5 000 documents TechQA depuis le corpus officiel
`PrimeQA/TechQA`, les 621 questions/reponses de `rojagtap/tech-qa`, puis cree
des echantillons deterministes de 2 000 tickets Bitext et 500 conversations
MSDialog :

```bash
uv run python scripts/download_corpus.py
```

Les fichiers JSONL et leur manifeste de checksums SHA-256 sont produits sous
`data/raw/`. La graine vaut `42` par defaut. Les principales options sont :

```bash
uv run python scripts/download_corpus.py --seed 123 --data-dir data/raw
uv run python scripts/download_corpus.py --skip-existing
uv run python scripts/download_corpus.py --force
```

Sans `--force`, le script refuse d'ecraser des sorties existantes.

## Analyse locale des corpus

Le script d'analyse charge les quatre fichiers JSONL avec Pandas et exporte
les statistiques, doublons et distributions sous
`docs/corpus_preparation/` :

```bash
uv run python scripts/analyze_corpus.py
```

## Pipeline de preparation du corpus

Le telechargement reste une operation independante :

```bash
uv run python scripts/download_corpus.py
```

Le flow Prefect part ensuite de `data/raw`, normalise et enrichit TechQA,
applique le chunking recursive retenu, deduplique les chunks, puis genere
automatiquement un rapport qualite Evidently sur le corpus final :

```bash
uv run python scripts/prepare_corpus.py --force
```

Il produit sous `data/processed/techqa/` :

- `documents.jsonl` : documents canoniques enrichis ;
- `chunks.jsonl` : chunks recursive dedupliques, prets a indexer ;
- `manifest.json` : statistiques et configuration du chunking.

Il produit egalement :

- `docs/corpus_preparation/corpus_quality_report.html` ;
- `docs/corpus_preparation/corpus_quality_summary.json`.

Le flow accepte `--skip-existing` et `--force`.

Les étapes de la pipeline:


```text
extract -> normalize -> document dedup -> enrich -> chunk -> chunk dedup
-> persist -> quality
```


## Operations independantes

La comparaison des chunkings utilise localement `BAAI/bge-m3` et n'est pas
executee par le pipeline :

```bash
uv run python scripts/compare_chunking.py --force
```



