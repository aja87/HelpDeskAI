"""Versioned prompt variants for RAG."""

from __future__ import annotations

from collections.abc import Callable

REWRITE_PROMPT = """Reformule cette question utilisateur en UNE requete de recherche claire,
incluant les termes techniques pertinents pour retrouver une documentation support.
Reponds uniquement avec la question reformulee.

Question : {question}

Reformulation :"""


def strict_prompt(question: str, context: str) -> str:
    """Strict support prompt: grounded answer, mandatory citations, refusal if missing."""
    return f"""Tu es un assistant support technique NovaCloud.

Tu reponds UNIQUEMENT avec les informations presentes dans le contexte.
Pour chaque affirmation factuelle, cite au moins une source entre crochets : [chunk_id].
Si le contexte ne contient pas l'information, reponds explicitement :
"Information non disponible dans les documents fournis."

Contexte :
{context}

Question :
{question}

Reponse avec citations obligatoires :"""


def pedagogical_prompt(question: str, context: str) -> str:
    """Pedagogical support prompt: simple explanation with citations."""
    return f"""Tu es un assistant support NovaCloud qui explique clairement a un utilisateur final.

Utilise uniquement le contexte ci-dessous.
Explique en termes simples, structure la reponse en etapes si utile, et cite les sources
entre crochets : [chunk_id].
Si l'information manque, dis-le sans inventer.

Contexte :
{context}

Question :
{question}

Reponse pedagogique avec citations :"""


def concise_prompt(question: str, context: str) -> str:
    """Concise support prompt: short operational answer with citations."""
    return f"""Tu es un assistant support NovaCloud.

Reponds de facon concise et operationnelle, uniquement a partir du contexte.
Ajoute les citations utiles entre crochets : [chunk_id].
Si la reponse n'est pas dans le contexte, indique que l'information est indisponible.

Contexte :
{context}

Question :
{question}

Reponse concise :"""


PROMPT_VARIANTS: dict[str, Callable[[str, str], str]] = {
    "strict": strict_prompt,
    "pedagogical": pedagogical_prompt,
    "concise": concise_prompt,
}


def get_prompt_variant(name: str) -> Callable[[str, str], str]:
    """Return a prompt function by version name."""
    try:
        return PROMPT_VARIANTS[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(PROMPT_VARIANTS))
        raise ValueError(f"unknown prompt version '{name}'. Expected one of: {allowed}") from exc
