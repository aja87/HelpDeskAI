# Architecture HelpDeskAI

## Perimetre

HelpDeskAI est un POC local d'assistant support N1. Le flux couvert est :

```text
ingestion -> retrieval -> RAG -> agent -> MCP -> observabilite
```

Le systeme s'execute avec `uv` pour les scripts Python et Docker Compose pour
Qdrant, pgvector, MLflow, Langfuse et la demo Streamlit.

## Vue Technique

```mermaid
flowchart LR
    User[Utilisateur support] --> Streamlit[Streamlit demo]
    User --> AgentCLI[run_agent.py]
    User --> RagCLI[run_rag.py]
    User --> IndexCLI[index_retrieval.py]

    Streamlit --> Agent[SupportAgent LangGraph]
    AgentCLI --> Agent
    RagCLI --> RAG[AdvancedRagPipeline]

    Agent --> Classifier[classify_intent]
    Classifier --> Planner[plan_mcp_calls]
    Planner --> Approval[request_human_approval]
    Planner --> Executor[execute_mcp_calls]
    Approval --> Executor
    Executor --> McpClient[StdioMcpClient]
    Executor --> Answer[generate_answer]
    Answer --> Quality[quality_check]

    McpClient --> CRM[MCP CRM]
    McpClient --> Knowledge[MCP Knowledge]
    CRM --> CRMData[(CRM simule)]
    Knowledge --> Search[SearchEngine]
    RAG --> Search

    Search --> Qdrant[(Qdrant)]
    Search --> BM25[BM25 local]
    Search --> Chunks[(TechQA chunks)]
    IndexCLI --> Qdrant
    IndexCLI --> PGVector[(pgvector)]

    RAG --> Claude[Claude]
    Agent --> Claude

    Eval[evaluate_rag.py] --> MLflow[(MLflow)]
    ModelReg[register_rag_model.py] --> MLflow
    Prompts[register_prompts.py] --> MLflow
    Trace[run_agent_langfuse.py] --> Langfuse[(Langfuse)]
    FinOps[finops_dashboard.py] --> Reports[(reports/finops)]
```

## Graphe Agent

Ce diagramme correspond aux noeuds et routes declares dans
`helpdeskai/agents/support_agent.py`.

```mermaid
flowchart TD
    START([START]) --> classify_intent

    classify_intent -->|ambiguous ou faible confiance| ask_clarification
    classify_intent -->|chitchat ou out_of_scope| direct_answer
    classify_intent -->|budget depasse| escalate_to_human
    classify_intent -->|besoin outils| plan_mcp_calls

    plan_mcp_calls -->|info manquante| ask_clarification
    plan_mcp_calls -->|action sensible| request_human_approval
    plan_mcp_calls -->|plan pret| execute_mcp_calls

    request_human_approval --> execute_mcp_calls
    execute_mcp_calls --> generate_answer
    generate_answer --> quality_check

    quality_check -->|fiable| END([END])
    quality_check -->|non fiable| escalate_to_human

    ask_clarification --> END
    direct_answer --> END
    escalate_to_human --> END
```

## Composants

| Composant | Role | Implementation |
| --- | --- | --- |
| Ingestion | Prepare les documents TechQA indexables. | `helpdeskai.ingestion`, `scripts/prepare_corpus.py` |
| Retrieval | Recherche dense Qdrant, sparse BM25 et hybride par fusion. pgvector est alimente par l'indexation mais n'est pas le chemin de recherche public actuel. | `helpdeskai.retrieval`, Qdrant, BM25, pgvector |
| RAG | Rewrite, retrieve, rerank, compress, generate. | `helpdeskai.rag`, `scripts/run_rag.py` |
| Agent | Orchestre classification, outils MCP, HITL et qualite. | `helpdeskai.agents.support_agent` |
| MCP CRM | Donnees client, abonnement et creation de ticket. | `helpdeskai.mcp_servers.crm` |
| MCP Knowledge | Outil `search_knowledge` branche sur retrieval. | `helpdeskai.mcp_servers.knowledge` |
| Demo | Chat local et validation d'action sensible. | `scripts/demo_streamlit.py` |
| Observabilite | Evaluations, registry, traces et FinOps. | MLflow, Langfuse, `helpdeskai.observability` |

## Flux De Donnees

```text
scripts/download_corpus.py
    -> data/raw/

scripts/prepare_corpus.py
    -> data/processed/techqa/documents.jsonl
    -> data/processed/techqa/chunks.jsonl
    -> data/processed/techqa/manifest.json

scripts/index_retrieval.py
    -> Qdrant collection helpdeskai_techqa_chunks
    -> pgvector table retrieval_chunks

MCP Knowledge search_knowledge
    -> helpdeskai.retrieval.search.search(...)
    -> Qdrant en dense, BM25 en sparse, Qdrant + BM25 en hybrid

pgvector
    -> alimente par scripts/index_retrieval.py
    -> conserve comme stockage vectoriel comparatif

scripts/evaluate_rag.py
    -> reports/rag/
    -> MLflow si tracking URI configure

scripts/register_rag_model.py
    -> MLflow pyfunc model helpdeskai-rag-chain
    -> alias production par defaut
```
