from __future__ import annotations

from typing import Any, Dict, Optional

from ...services.embeddings import embed_text
from ...services.rag import ensure_index_ready
from ...db import pgvector_store
from ...settings import settings


async def retrieve_knowledge(query: str, top_k: int | None = None) -> Dict[str, Any]:
    """Embed the query and vector-search via pgvector. Filters by similarity threshold."""
    await ensure_index_ready()
    k = top_k or settings.rag_top_k

    vec = embed_text(query)
    hits = await pgvector_store.query(vec.tolist(), top_k=k)

    results = []
    for hit in hits:
        results.append({
            "score": hit["similarity"],
            "name": hit["item_name"],
            "type": hit.get("category"),
            "service": None,
            "qty": None,
            "price": hit["price"],
            "in_stock": hit["in_stock"],
        })

    return {
        "tool": "retrieve_knowledge",
        "query": query,
        "results": results,
        "count": len(results),
    }
