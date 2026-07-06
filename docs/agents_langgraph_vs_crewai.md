# Agent LangGraph vs CrewAI

## Choix implemente

HelpDeskAI utilise LangGraph pour l'agent support N1. Le graphe explicite les
noeuds de politique et d'outillage : `classify_intent`, `plan_mcp_calls`,
`request_human_approval`, `execute_mcp_calls`, `generate_answer`,
`quality_check`, `ask_clarification` et `escalate_to_human`.

Le noeud `classify_intent` appelle un LLM, puis valide une intention metier
(`technical_question`, `account_question`, `account_plus_knowledge_question`,
`sensitive_action`, `out_of_scope`, `chitchat`, `ambiguous`). Les intentions
terminales (`ambiguous`, `chitchat`, `out_of_scope`) sortent avant toute
planification d'outil. Les intentions qui exigent des donnees ou des sources
passent par `plan_mcp_calls`, qui produit une liste ordonnee d'outils MCP
(`search_knowledge`, `get_customer`, `get_subscription_status`,
`create_ticket`).

LangGraph est retenu pour ce POC car il donne un controle fin sur le routage, la
planification deterministe des appels MCP, le checkpointing SQLite par
`thread_id`, les interruptions human-in-the-loop avant une action sensible et les
budgets d'execution. Ces points sont critiques pour un assistant support
tracable et auditable.

## Human-in-the-loop

Les demandes sensibles, par exemple une creation de ticket d'escalade support,
preparent une action `create_ticket` dans `plan_mcp_calls`, passent par
`request_human_approval`, puis suspendent le graphe juste apres ce noeud. Un
humain peut approuver ou rejeter en modifiant l'etat checkpointe, puis
l'execution reprend sur le meme `thread_id` vers `execute_mcp_calls`.
