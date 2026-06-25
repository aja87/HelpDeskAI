# Architecture HelpDeskAI

## Architecture fonctionnelle cible V1

Le schéma ci-dessous présente la vision fonctionnelle de HelpDeskAI.
Il ne détaille pas les composants techniques, mais illustre les grands rôles du système : comprendre la demande, rechercher dans la base documentaire, consulter les outils internes simulés si nécessaire, répondre avec sources, demander une clarification ou escalader vers un humain.

```mermaid
flowchart LR
    U[Utilisateur / Agent support N1] --> A[Assistant IA HelpDeskAI]

    A --> B[Compréhension de la demande]
    B --> C{Type de demande}

    C -->|Question documentée| D[Recherche dans la base de connaissances]
    D --> E[Réponse sourcée]

    C -->|Demande ambiguë| F[Question de clarification]

    C -->|Donnée client nécessaire| G[Consultation des outils internes simulés]
    G --> E

    C -->|Cas sensible ou incertain| H[Escalade vers un agent humain]

    E --> U
    F --> U
    H --> U

    A --> I[Traçabilité et suivi qualité]
    I --> J[Évaluation des réponses]
    I --> K[Suivi des coûts et de la performance]
```

## Statut

Template a completer pendant la phase de cadrage.

## Contexte

Decrire le contexte metier, les utilisateurs et les contraintes du systeme.

## Architecture cible

Ajouter un diagramme C4 de niveau 2 presentant les composants et leurs flux.

## Composants

| Composant | Responsabilité | Technologie / statut |
| --- | --- | --- |
| Ingestion | Pipeline modulaire `extract → normalize → document dedup → enrich → chunk recursive → chunk dedup → persist → quality` appliqué uniquement aux 5 000 documents TechQA. Les questions/réponses TechQA restent hors pipeline et alimentent le golden dataset via un script indépendant. Le téléchargement et la comparaison des stratégies restent aussi indépendants. | Python, BeautifulSoup, tokenizer BGE-M3, Prefect, Evidently — implémenté dans `helpdeskai.ingestion`, `helpdeskai.corpus` et `scripts/`. |
| Retrieval | A definir | A definir |
| RAG | A definir | A definir |
| Agent | A definir | A definir |
| Serveurs MCP | A definir | A definir |
| Observabilite | A definir | A definir |

## Flux de donnees

A documenter.

## Exigences non fonctionnelles

- Securite : a definir.
- Performance : a definir.
- Disponibilite : a definir.
- Observabilite : a definir.
- FinOps : a definir.

## Decisions d'architecture

Consigner les decisions importantes dans des ADR dedies.

