_Projet fil rouge — Cursus AI Engineer Neosoft_

# HelpDeskAI

_Concevoir et industrialiser un assistant de support N1 augmenté par RAG et agents_

équipes de 3 à 4 personnes

✦

## Contexte fictif

Vous êtes consultants AI Engineers, mandatés par **NovaCloud**, un éditeur SaaS B2B (gestion documentaire, 12 000 entreprises clientes, 80 000 utilisateurs finaux). Le service support reçoit **environ 1 800 tickets par semaine**, dont **60 % sont des questions répétitives** déjà documentées dans la base de connaissances publique du produit (FAQ, guides utilisateurs, release notes, articles d'aide).

Le directeur du support a fait un POC avec ChatGPT en copier-collant des FAQ dans le prompt système : ça marche en démo, ça plante en production (hallucinations sur les versions de produit, citations inventées, pas de logs, pas de coûts maîtrisés). La direction veut maintenant un **vrai système d'entreprise** : déployable en interne, traçable, évaluable, capable d'agir sur le SI (créer un ticket, vérifier le statut d'un compte, escalader vers un agent humain).

Votre mission : concevoir, développer, et démontrer un POC défendable d'**assistant de support N1** intégrant RAG sur la base documentaire, agent LangGraph capable d'utiliser les outils du SI via MCP, observabilité complète (MLflow + Langfuse), évaluation rigoureuse (Ragas), et chiffrage FinOps.

## Ce que vous ne faites pas

_Vous ne construisez pas une interface utilisateur finale (un Streamlit ou Gradio basique suffit pour la démo). Vous ne faites pas de fine-tuning de modèle. Vous n'industrialisez pas un déploiement multi-régions ni un cluster Kubernetes — un docker-compose local démontrable suffit. Le périmètre est le système IA lui-même : chaîne RAG, agent, intégrations, observabilité, évaluation._

✦

## Données

Le projet exploite **trois corpus publics complémentaires**, tous téléchargeables sans authentification.

### Corpus 1 — Base documentaire (RAG)

**TechQA dataset** (IBM Research, licence CDLA-Sharing 1.0) : 600 questions techniques annotées + corpus de **~800 000 documents techniques IBM** (technotes, support pages, troubleshooting guides). Disponible sur Hugging Face (`rojagtap/tech-qa`) et github.com/IBM/techqa.

> Caractéristiques pour le projet : documents non-structurés (HTML/texte), métadonnées riches (produit, version, date, catégorie), questions/réponses or-de-vérité pour l'évaluation Ragas. Volume volontairement réduit à un sous-ensemble (5 000 documents environ) pour les contraintes du POC.

### Corpus 2 — Tickets support (test conversationnel)

**Bitext Customer Support LLM Chatbot Training Dataset** (Hugging Face : `bitext/Bitext-customer-support-llm-chatbot-training-dataset`, licence CDLA-Sharing 1.0) : **27 000 paires** question utilisateur / réponse de référence, annotées par intent (27 intents) et par catégorie (11 catégories : ACCOUNT, ORDER, PAYMENT, REFUND, etc.).

> Usage dans le projet : construction du golden dataset pour évaluation RAG, simulation de tickets entrants pour la démo, génération du jeu de test du module évaluation (M5).

### Corpus 3 — Conversations multi-tours (agent)

**MSDialog-Intent** (Microsoft Research, licence MIT) : **35 000 conversations** support technique multi-tours, annotées (clarification, élaboration, suggestion, etc.). Téléchargeable depuis [github.com/qhjqhj00/MSDialog](https://raw.githubusercontent.com/SCU-ChenYue/MSDialog_RL/main/test_MSDialog.jsonl) ou la page officielle Microsoft Research.

> Usage : tester les patterns agentiques (clarification utilisateur, plan-and-execute), construire des scénarios de test pour le module orchestration (M6/M7).

### Pipeline de téléchargement

Le repo de départ contient un script `scripts/download_corpus.py` qui télécharge les trois corpus depuis Hugging Face Hub avec vérification de checksum, et produit un sous-ensemble exploitable :

| Corpus           | Volume retenu     | Usage                       |
| ---------------- | ----------------- | --------------------------- |
| TechQA documents | 5 000 docs        | Indexation vectorielle (M4) |
| TechQA Q/A       | 600 paires        | Golden dataset Ragas (M5)   |
| Bitext           | 2 000 paires      | Tests d'intent et démo      |
| MSDialog         | 500 conversations | Tests agent multi-tours     |

✦

## Repo de départ

Vous recevez un repository minimal, fonctionnel mais non industrialisé. Le squelette est prêt, **rien n'est implémenté**.

### Contenu fourni

```
helpdeskai/
├── pyproject.toml              # uv, dépendances déclarées non installées
├── scripts/
│   └── download_corpus.py      # téléchargement des 3 corpus avec checksum
├── helpdeskai/
│   ├── __init__.py
│   ├── ingestion/              # vide, à implémenter (M2)
│   ├── retrieval/              # vide, à implémenter (M4)
│   ├── rag/                    # vide, à implémenter (M5)
│   ├── agents/                 # vide, à implémenter (M6/M7)
│   ├── mcp_servers/            # vide, à implémenter (M8)
│   └── observability/          # vide, à implémenter (M9)
├── tests/                      # vide
├── docker-compose.yml          # squelette : qdrant, mlflow, langfuse, api
├── .github/workflows/ci.yml    # squelette CI
├── README.md                   # mission, sans solution
└── docs/
    └── ARCHITECTURE.md         # template à remplir au M3
```

### Ce que le repo de départ ne contient pas

- Aucun code métier (extraction, indexation, RAG, agent).
- Aucune configuration Qdrant, MLflow, Langfuse fonctionnelle.
- Aucun serveur MCP.
- Aucun test.
- Aucune intégration LLM.

✦

## Phases du projet (calées sur le cursus)

### Phase 1 — Cadrage (Module 3, 1 jour)

**Livrables**

- Document de cadrage : contexte, parties prenantes, exigences fonctionnelles et non-fonctionnelles, critères de succès chiffrés (taux de résolution autonome cible, latence p95, coût par ticket).
- Schéma d'architecture cible V1 (composants, flux de données, intégrations SI simulées).
- Backlog priorisé MoSCoW.
- Constitution des équipes (3 à 4 personnes), répartition des rôles.

**Critères d'évaluation**

- Cohérence du périmètre avec les contraintes (13 jours, équipe).
- Argumentation des choix d'architecture face au formateur.

---

### Phase 2 — Préparation du corpus (Module 2, 2 jours)

**Objectifs techniques**

- Télécharger les trois corpus via le script fourni.
- Implémenter `helpdeskai.ingestion` : extraction texte des documents TechQA (HTML → texte propre), normalisation (encodage, ponctuation, déduplication), enrichissement métadonnées (produit, version, date, catégorie).
- Mettre en place une stratégie de chunking documentée et testée : comparer fixed-size, recursive, semantic. Justifier le choix retenu sur un échantillon de 50 documents.
- Pipeline reproductible avec Prefect ou simple `make ingest`.
- Rapport qualité automatisé avec Evidently ou Great Expectations sur le corpus indexé (complétude métadonnées, distribution des longueurs, détection de doublons).
- Préparer un golden dataset de 100 questions à partir de TechQA + Bitext pour les évaluations futures.

**Livrables**

- Corpus normalisé prêt à indexation (parquet ou jsonl avec métadonnées).
- Notebook ou script de comparaison des stratégies de chunking.
- Rapport qualité Evidently exporté en HTML.
- Golden dataset commité dans `tests/golden/`.

---

### Phase 3 — Recherche vectorielle et hybride (Module 4, 2 jours)

**Objectifs techniques**

- Déployer **Qdrant** via le `docker-compose.yml` du repo, avec volume persistant.
- Implémenter `helpdeskai.retrieval.indexer` : indexation du corpus avec embeddings (au choix : `BAAI/bge-m3`, `intfloat/multilingual-e5-large`, ou OpenAI `text-embedding-3-small` — justifier le choix avec un benchmark MTEB).
- Indexer en parallèle dans **PostgreSQL avec pgvector** (le second moteur sert de comparaison).
- Implémenter la recherche dense, sparse (BM25 via Qdrant ou Elasticsearch), et hybride avec Reciprocal Rank Fusion.
- Filtrage par métadonnées (produit, version, date) — démontrer un cas multi-tenant simulé (3 produits NovaCloud distincts).
- Benchmark : recall@5, recall@10, MRR, latence p95 sur les 100 questions du golden dataset.

**Livrables**

- Qdrant peuplé, requêtable, persistant.
- Module `retrieval` exposant `search(query, top_k, filters, mode)` avec `mode in {dense, sparse, hybrid}`.
- Rapport de benchmark comparant les trois modes (tableau de métriques + analyse).
- Tests pytest sur le retrieval (au moins 5 tests).

---

### Phase 4 — RAG avancé et évaluation (Module 5, 1 jour)

**Objectifs techniques**

- Implémenter `helpdeskai.rag.pipeline` : chaîne RAG avec **query rewriting** (réécriture de la question utilisateur), **re-ranking** par cross-encoder (`BAAI/bge-reranker-v2-m3` ou Cohere Rerank), **contextual compression** des chunks retenus, génération avec LLM (Claude, GPT-4o, ou Mistral au choix — justifier).
- Construire la boucle d'évaluation **Ragas** : faithfulness, answer relevancy, context precision, context recall.
- Versionner trois variantes de prompt système et comparer leurs scores Ragas.
- Intégrer l'évaluation dans la CI : un PR qui dégrade faithfulness de plus de 5 points doit faire échouer le job.

**Livrables**

- Pipeline RAG fonctionnel, paramétrable (mode retrieval, modèle, prompt).
- Trois variantes de prompts versionnées avec leurs scores Ragas comparés.
- Job CI `evaluate` qui exécute Ragas sur un sous-ensemble du golden dataset.

---

### Phase 5 — Agent et orchestration (Modules 6 et 7, 2 jours)

**Objectifs techniques**

_Module 6 — Patterns agentiques avec LangGraph_

- Implémenter `helpdeskai.agents.support_agent` : agent ReAct LangGraph qui combine recherche RAG, classification d'intent, et génération.
- Modéliser l'agent en graphe d'états : nœuds (`classify_intent`, `retrieve`, `generate`, `escalate`), edges conditionnels selon la confiance.
- Ajouter un nœud de **clarification** : si la question est ambigüe, l'agent pose une question de clarification au lieu de répondre.

_Module 7 — State management et human-in-the-loop_

- Ajouter le **checkpointing** (SQLite ou PostgreSQL) pour reprendre une conversation interrompue.
- Implémenter un point **human-in-the-loop** : avant toute action sensible (ex : "créer un ticket de remboursement"), l'agent demande validation explicite.
- Gérer les budgets d'exécution : max 5 itérations, max 10 000 tokens par session, fallback gracieux si dépassement.
- Comparer brièvement avec une version CrewAI multi-agents (rôle "support" + rôle "escalation officer") — analyse écrite des compromis.

**Livrables**

- Agent LangGraph fonctionnel avec visualisation du graphe (export PNG ou Mermaid).
- Démo de la conversation reprise après interruption (checkpointing).
- Démo human-in-the-loop sur action sensible.
- Note d'analyse LangGraph vs CrewAI (1 page).

---

### Phase 6 — MCP et intégration SI (Module 8, 1 jour)

**Objectifs techniques**

- Implémenter `helpdeskai.mcp_servers.crm` : serveur MCP exposant un mock CRM (FastMCP) avec tools : `get_customer(customer_id)`, `get_subscription_status(customer_id)`, `list_recent_tickets(customer_id)`, `create_ticket(customer_id, subject, body, priority)`.
- Implémenter `helpdeskai.mcp_servers.knowledge` : serveur MCP exposant la recherche RAG comme tool MCP réutilisable.
- Sécuriser : authentification par token, validation Pydantic stricte sur les entrées, rate limiting basique.
- Intégrer ces deux serveurs MCP dans l'agent LangGraph de la phase 5 — l'agent doit pouvoir vérifier le statut d'abonnement d'un client avant de répondre à une question facturation.

**Livrables**

- Deux serveurs MCP fonctionnels, démarrables via `docker-compose`.
- Agent capable d'orchestrer les appels MCP en chaîne.
- Documentation OpenAPI-like des tools MCP exposés.

---

### Phase 7 — LLMOps, observabilité et FinOps (Module 9, 1 jour)

**Objectifs techniques**

- **MLflow Tracking** : logger chaque run d'évaluation Ragas (paramètres : modèle, température, prompt version, retrieval mode ; métriques : faithfulness, relevancy, latence, coût ; artefacts : golden dataset utilisé).
- **MLflow Prompt Registry** : versionner les trois prompts système avec promotion dev → staging → prod.
- **MLflow Model Registry** : enregistrer la chaîne RAG complète comme modèle pyfunc, promouvoir une version "production".
- **Langfuse** (ou Phoenix) : instrumenter l'agent LangGraph pour le tracing temps réel — chaque conversation produit une trace explorable (chaîne d'appels LLM, latences, coûts par requête).
- **Dashboard FinOps** : extraction des coûts par cas d'usage, alerte si dépassement de budget mensuel simulé.
- **Évaluation continue** : pipeline qui sample 5 % des conversations production simulées, les ré-évalue avec un LLM-as-a-judge, et ré-injecte les scores dans MLflow.

**Livrables**

- Stack MLflow + Langfuse opérationnelle dans `docker-compose`.
- Captures du Prompt Registry avec trois versions promues.
- Trace Langfuse d'une conversation agent complète.
- Note FinOps : coût par requête mesuré, projection mensuelle, leviers d'optimisation identifiés.

---

### Phase 8 — POC final et soutenance (Module 10, 2 jours)

**Jour 1 — Finalisation**

- Intégration finale de toutes les briques.
- Streamlit ou Gradio minimal pour la démo (chat avec l'agent + visualisation de la trace Langfuse en parallèle).
- Dossier d'architecture final : schéma C4 niveau 2 minimum, décisions documentées (ADR), flux de données.
- Chiffrage FinOps détaillé : coût d'infrastructure (Qdrant, MLflow, Langfuse, agent), coût d'inférence (par modèle, par cas d'usage), projection à 1 800 tickets/semaine.
- Analyse de risques : techniques (hallucinations, drift, panne LLM provider), sécurité (prompt injection, fuite de données), conformité (RGPD, données clients), dépendances.

**Jour 2 — Soutenance**

- Présentation devant jury (formateurs Neosoft + un évaluateur externe si possible) : 25 minutes présentation + démo + 15 minutes Q/R.
- Démonstration en direct : conversation avec l'agent, déclenchement d'un MCP CRM, observation Langfuse en temps réel, explication d'une trace.
- Feedback individuel et collectif.

**Livrables finaux**

- Repo Git complet, CI verte, badge dans le README.
- `docker-compose up --build` démarre toute la stack en moins de 5 minutes.
- POC démontrable de bout en bout (ingestion → RAG → agent → MCP → observabilité).
- Dossier d'architecture, chiffrage FinOps, analyse de risques.
- Support de soutenance (slides + démo).
