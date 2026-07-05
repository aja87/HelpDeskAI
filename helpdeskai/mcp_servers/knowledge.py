"""Knowledge MCP server exposing source-aware retrieval."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ValidationError

from helpdeskai.mcp_servers.security import audited_tool
from helpdeskai.retrieval.models import SearchFilters, SearchMode, SearchResult
from helpdeskai.retrieval.search import search


class SearchKnowledgeInput(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    product: str | None = Field(default=None, min_length=1, max_length=80)
    version: str | None = Field(default=None, min_length=1, max_length=40)
    tenant: str | None = Field(default=None, min_length=1, max_length=80)


SearchBackend = Callable[[str, int, SearchFilters | None], list[SearchResult]]


def _default_search_backend(
    query: str,
    top_k: int,
    filters: SearchFilters | None,
) -> list[SearchResult]:
    return search(query, top_k=top_k, filters=filters, mode=SearchMode.HYBRID)


def search_knowledge_business(
    *,
    actor_id: str,
    query: str,
    top_k: int = 5,
    product: str | None = None,
    version: str | None = None,
    tenant: str | None = None,
    backend: SearchBackend = _default_search_backend,
) -> dict[str, Any]:
    try:
        args = SearchKnowledgeInput(
            query=query,
            top_k=top_k,
            product=product,
            version=version,
            tenant=tenant,
        )
    except ValidationError as exc:
        return {"error": "validation_error", "details": exc.errors()}
    filters = SearchFilters(product=args.product, version=args.version, tenant=args.tenant)
    results = backend(args.query, args.top_k, filters)
    return {
        "query": args.query,
        "top_k": args.top_k,
        "filters": {
            "product": args.product,
            "version": args.version,
            "tenant": args.tenant,
        },
        "results": [
            {
                "source_id": result.chunk_id,
                "document_id": result.document_id,
                "snippet": result.content[:500],
                "score": result.score,
                "metadata": result.metadata,
                "source_scores": result.source_scores,
            }
            for result in results
        ],
    }


@audited_tool("search_knowledge")
def _search_knowledge_audited(
    *,
    actor_id: str,
    query: str,
    top_k: int = 5,
    product: str | None = None,
    version: str | None = None,
    tenant: str | None = None,
) -> dict[str, Any]:
    return search_knowledge_business(
        actor_id=actor_id,
        query=query,
        top_k=top_k,
        product=product,
        version=version,
        tenant=tenant,
    )


mcp = FastMCP("helpdeskai-knowledge")


@mcp.tool()
def search_knowledge(
    query: str,
    token: str,
    top_k: int = 5,
    product: str | None = None,
    version: str | None = None,
    tenant: str | None = None,
    actor_id: str = "agent_default",
) -> dict[str, Any]:
    """Search the NovaCloud knowledge base with source-aware results."""
    return _search_knowledge_audited(
        actor_id=actor_id,
        token=token,
        query=query,
        top_k=top_k,
        product=product,
        version=version,
        tenant=tenant,
    )


if __name__ == "__main__":
    if "--transport" in sys.argv:
        transport = sys.argv[sys.argv.index("--transport") + 1]
        mcp.run(transport=transport)
    else:
        mcp.run()
