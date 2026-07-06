# Exemples De Logs Et Traces

Ce document complete le deck `helpdeskai_soutenance.md`. Il donne des exemples
montrables pendant la soutenance sans inventer de fausses captures.

## 1. Chemin Agent

Visible dans la demo Streamlit, section "Details agent".

Question RAG technique :

```text
classify_intent -> plan_mcp_calls -> execute_mcp_calls -> generate_answer -> quality_check
```

Question CRM :

```text
classify_intent -> plan_mcp_calls -> execute_mcp_calls -> generate_answer -> quality_check
```

Action sensible avec validation humaine :

```text
classify_intent -> plan_mcp_calls -> request_human_approval
```

Apres approbation :

```text
request_human_approval -> execute_mcp_calls -> generate_answer -> quality_check
```

## 2. Audit MCP CRM

Source : `data/audit/mcp-crm.jsonl`

Exemple succes abonnement :

```json
{
  "event": "tool_call",
  "tool": "get_subscription_status",
  "actor_id": "streamlit_demo",
  "trace_id": "14e6a152-f37c-4e78-b087-b69500529972",
  "args": {"customer_id": "cust_acme"},
  "result": "success",
  "duration_ms": 2
}
```

Exemple creation de ticket apres validation humaine :

```json
{
  "event": "tool_call",
  "tool": "create_ticket",
  "actor_id": "streamlit_demo",
  "trace_id": "8fa50c87-287a-4a57-9159-7404374ac431",
  "args": {
    "customer_id": "cust_acme",
    "subject": "Validation action support sensible",
    "priority": "high"
  },
  "result": "success",
  "duration_ms": 5
}
```

Ce que le log prouve :

- l'outil appele ;
- l'acteur (`streamlit_demo`, `agent_default`, `test`) ;
- l'identifiant de trace technique ;
- les arguments transmis ;
- le resultat et la latence.

## 3. Audit MCP Knowledge

Source : `data/audit/mcp-knowledge.jsonl`

Exemple recherche documentaire :

```json
{
  "event": "tool_call",
  "tool": "search_knowledge",
  "actor_id": "streamlit_demo",
  "trace_id": "eeadcc81-250e-4f32-9b36-b87466ba984a",
  "args": {
    "query": "How do I configure queue watcher parameters in IBM Sterling B2B Integrator?",
    "top_k": 5,
    "product": null,
    "version": null,
    "tenant": null
  },
  "result": "success",
  "duration_ms": 23145
}
```

Ce que le log prouve :

- le RAG est appele via MCP Knowledge ;
- la question utilisateur est auditable ;
- la latence outil est mesurable ;
- les filtres produit/version/tenant sont explicites.

## 4. Logs De Securite Et Validation MCP

Source : `data/audit/mcp.jsonl`

Exemple validation Pydantic :

```json
{
  "event": "tool_call",
  "tool": "create_ticket",
  "actor_id": "test",
  "args": {"customer_id": "cust_acme", "priority": "critical"},
  "result": "validation_error",
  "errors": [
    {
      "loc": ["priority"],
      "msg": "Input should be 'low', 'medium', 'high' or 'urgent'"
    }
  ]
}
```

Exemple authentification MCP :

```json
{
  "event": "tool_call",
  "tool": "get_customer",
  "actor_id": "test",
  "args": {"customer_id": "cust_acme"},
  "result": "auth_error",
  "error_msg": "invalid MCP token"
}
```

Ces exemples montrent que le serveur MCP ne se contente pas d'executer les
outils : il valide les entrees et journalise les refus.

## 5. Trace Langfuse

La trace Langfuse n'est pas stockee comme fichier dans le repo. Elle est visible
dans l'interface Langfuse quand les cles sont configurees.

Commande :

```bash
uv run python scripts/run_agent_langfuse.py
```

URL :

```text
http://localhost:3000 -> Tracing -> Traces
```

Ce que la trace doit montrer :

- une conversation rattachee a `langfuse_session_id` ;
- un `langfuse_user_id` ;
- les appels LangGraph ;
- les appels LLM ;
- les latences ;
- les erreurs eventuelles ;
- le lien entre question utilisateur, decision agent et reponse.

Dans Streamlit, si Langfuse est configure, la sidebar affiche aussi :

- le statut `Langfuse tracing: actif` ;
- le bouton `Derniere trace` ;
- le `trace_id` de la derniere conversation.

## 6. MLflow Tracking

Commande d'evaluation :

```bash
uv run python scripts/evaluate_rag.py --prompt strict --limit 2 --mlflow-tracking-uri http://127.0.0.1:5000
```

Champs logges par `helpdeskai.observability.mlflow_tracking` :

- parametres : modele, temperature, prompt version, retrieval mode ;
- metriques : faithfulness, relevancy, latency average, latency p95 ;
- couts : `cost_total_usd`, `cost_per_query_usd` ;
- artefacts : rapports et golden dataset si disponible.

URL :

```text
http://127.0.0.1:5000
```

## 7. Prompt Registry MLflow

Commande :

```bash
uv run python scripts/register_prompts.py --tracking-uri http://127.0.0.1:5000
```

Promotions attendues :

```text
dev        -> concise
staging    -> pedagogical
production -> strict
```

Capture attendue dans `docs/presentation/assets/mlflow-prompt-registry.png`.

## 8. Commandes Utiles Pendant La Demo

Afficher les derniers logs CRM :

```powershell
Get-Content data\audit\mcp-crm.jsonl -Tail 5
```

Afficher les derniers logs Knowledge :

```powershell
Get-Content data\audit\mcp-knowledge.jsonl -Tail 5
```

Afficher les logs Docker de la demo :

```bash
docker compose logs demo --tail 100
```

Afficher les logs MCP :

```bash
docker compose logs mcp-crm --tail 100
docker compose logs mcp-knowledge --tail 100
```

Afficher le statut de la stack :

```bash
docker compose ps
```
