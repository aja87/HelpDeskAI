# Architecture HelpDeskAI

## Statut

Document de reference pour l'organisation du code de la phase actuelle.

## Objectif produit

HelpDeskAI construit un assistant N1 enterprise base sur:

- une base de connaissances preparee par pipeline reproductible
- un moteur de retrieval hybride
- une orchestration agentique avec outils MCP
- une observabilite de bout en bout (qualite, cout, traces)

## Principe d'organisation du code

Deux niveaux de responsabilite sont imposes:

1. Les scripts executables sont dans le dossier scripts.
2. La logique metier est dans le package helpdeskai, decoupee en sous-modules.

Consequence: un script dans scripts ne doit contenir que l'entree CLI, puis deleguer vers helpdeskai.<domaine>.<module>.

## Vue d'ensemble des modules

| Module | Responsabilite principale | Entree/Sortie |
| --- | --- | --- |
| scripts/ | Points d'entree d'execution locale et CI | CLI utilisateur |
| helpdeskai/corpus/ | Telechargement et sous-echantillonnage des corpus bruts | data/raw/*.jsonl + checksums |
| helpdeskai/ingestion/ | Normalisation, chunking, controle qualite, golden set | data/processed + reports/ingestion + tests/golden |
| helpdeskai/retrieval/ | Construction et interrogation des index | Index/vector store + metriques retrieval |
| helpdeskai/rag/ | Rewriting, rerank, compression, generation | Reponses contextualisees |
| helpdeskai/agents/ | Orchestration des etats et chemins de decision | Tool calls + reponses agent |
| helpdeskai/mcp_servers/ | Exposition des outils metier (CRM, KB, etc.) | Contrats MCP valides |
| helpdeskai/observability/ | Traces, metriques, evaluation offline/online | Runs MLflow/Langfuse + rapports |

## Focus: architecture corpus

Le telechargement de corpus est maintenant organise ainsi:

- scripts/download_corpus.py: script executable mince, delegue au package.
- helpdeskai/corpus/downloader.py: orchestration complete du workflow de download.
- helpdeskai/corpus/datasets.py: selection de split et logique de sampling.
- helpdeskai/corpus/transforms.py: mapping schema source vers schema brut normalise.
- helpdeskai/corpus/io_utils.py: ecriture JSONL, SHA256, verification checksums.
- helpdeskai/corpus/config.py: constantes de dataset et configuration typed.

Ce decoupage permet:

- la reutilisation depuis d'autres scripts ou flows
- des tests unitaires plus ciblables par couche
- un script CLI stable meme si la logique interne evolue

## Focus: architecture ingestion

L'ingestion suit maintenant la meme regle que corpus:

- scripts/run_ingestion.py: script executable avec toute la CLI ingestion.
- helpdeskai/ingestion/workflow.py: orchestration du pipeline et taches.
- helpdeskai/ingestion/normalize.py: normalisation texte et schemas cibles.
- helpdeskai/ingestion/chunking.py: strategies de chunking + benchmark.
- helpdeskai/ingestion/quality.py: calcul qualite + rapport HTML.
- helpdeskai/ingestion/golden.py: construction du golden dataset.
- helpdeskai/ingestion/io_utils.py: read/write JSONL et JSON.
- helpdeskai/ingestion/config.py: constantes et configuration typed.
- helpdeskai/ingestion/pipeline.py: facade de compatibilite pour les imports historiques.

Regle explicite: aucun parsing CLI ingestion dans les sous-modules helpdeskai/ingestion.

## Flux de donnees

1. scripts/download_corpus.py lance le workflow corpus.
2. helpdeskai/corpus recupere TechQA, Bitext et MSDialog puis produit data/raw/*.jsonl.
3. helpdeskai/corpus ecrit data/raw/checksums.sha256.json pour garantir l'integrite.
4. scripts/run_ingestion.py lance l'ingestion.
5. helpdeskai/ingestion lit data/raw, normalise et cree data/processed.
6. Les modules retrieval/rag/agents consomment ensuite les artefacts normalises.

## Exigences non fonctionnelles

- Reproductibilite: seed fixe, formats JSONL deterministes, checksums de sortie.
- Tracabilite: logging script et manifests de generation.
- Maintenabilite: separation claire entre code executable et logique metier.
- Securite: validation stricte des inputs cote serveurs MCP (a renforcer au fur et a mesure).
- FinOps: suivi des couts de run et d'inference via observability.

## Decisions d'architecture actives

1. Tous les scripts executables vivent dans scripts/.
2. Toute logique metier reutilisable vit dans helpdeskai/.
3. Les datasets bruts restent dans data/raw et ne sont jamais modifies in-place par ingestion.
4. Chaque etape produit des artefacts explicites et verifiables.

