# FinOps HelpDeskAI

## Hypotheses

- Volume cible : 1 800 tickets/semaine, soit environ 7 800 tickets/mois.
- Scenario de reference disponible : 10 000 requetes/mois dans
  `reports/finops/scenarios.csv`.
- Requete RAG moyenne : 4 000 tokens entrants et 300 tokens sortants.
- POC actuel : compression du contexte RAG active, sans routage modele, prompt
  caching ou cache semantique mesures.
- Les montants sont des ordres de grandeur POC, pas un devis fournisseur.

## Estimation

| Variante | Cout/mois pour 10k requetes | Cout/requete | Projection 7 800 tickets/mois |
| --- | ---: | ---: | ---: |
| Sans optimisations | $247.00 | $0.02470 | $192.66 |
| POC actuel estime | $199.00 | $0.01990 | $155.22 |
| Cible optimisee | $141.16 | $0.01412 | $110.14 |

Le cout aujourd'hui du POC est estime a **$0.01990 par requete**, soit
**$155.22/mois** au volume cible de 1 800 tickets/semaine.

La cible optimisee reduit l'estimation mensuelle d'environ **$82.52** par
rapport au scenario sans optimisations, et d'environ **$45.08** par rapport au
POC actuel estime, au volume cible.

## Cout Par Requete

- Sans optimisations : **$0.02470** par requete.
- POC actuel estime : **$0.01990** par requete.
- Cible optimisee : **$0.01412** par requete.
- Gain POC actuel : **$0.00480** par requete, soit environ **19 %**.
- Gain cible optimisee : **$0.01058** par requete, soit environ **43 %**.

Ces valeurs viennent de `reports/finops/scenarios.csv`, colonne
`cost_per_query_usd`, sur le scenario `Small scale (10k)`.

## Ce Qui Coute

- LLM : principal poste de cout, surtout les tokens entrants du contexte RAG.
- Infrastructure : Qdrant, pgvector, MLflow, Langfuse et runtime agent.
- Embeddings : faible cout mensuel dans ce POC, plus visible pendant
  l'indexation que pendant l'usage courant.

## Leviers

- Router les cas simples vers un modele moins couteux.
- Compresser le contexte avant generation.
- Activer le prompt caching pour les prompts et schemas stables.
- Ajouter un cache semantique sur les questions repetitives.
- Evaluer seulement un echantillon de conversations avec LLM-as-a-judge.
