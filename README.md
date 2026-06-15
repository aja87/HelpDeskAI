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

## Services disponibles

Le squelette Qdrant et MLflow peut etre demarre avec :

```bash
docker compose up -d
```

Les profils `app` et `observability` sont des placeholders qui seront
completes dans les phases suivantes.
