# Phase 5 - Agent LangGraph vs CrewAI

## Choix implemente

HelpDeskAI utilise LangGraph pour l'agent support N1. Le graphe explicite les
noeuds demandes : `classify_intent`, `retrieve`, `generate`, `clarification` et
`escalate`. Le noeud `classify_intent` appelle un LLM, puis valide une des cinq
intentions metier exactes (`nova_question`, `account_question`, `out_of_scope`,
`chitchat`, `ambiguous`). Cette intention
est ensuite mappee vers une route interne (`answer_with_rag`, `account_support`,
`out_of_scope`, `chitchat`, `clarification`, `sensitive_action`) pour le
routage conditionnel.

LangGraph est retenu pour ce POC car il donne un controle fin sur le routage, le
checkpointing SQLite par `thread_id`, les interruptions human-in-the-loop avant
une action sensible et les budgets d'execution. Ces points sont critiques pour
un assistant support tracable et auditable.

## Human-in-the-loop

Les demandes sensibles, par exemple une creation de ticket d'escalade support,
preparent une action `create_ticket` puis suspendent le graphe avant `escalate`.
Un humain peut approuver ou rejeter en modifiant l'etat checkpointe, puis
l'execution reprend sur le meme `thread_id`.


