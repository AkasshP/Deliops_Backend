from __future__ import annotations
from fastapi import APIRouter

from ..services.rag import build_index

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/index-status")
async def index_status():
    """Check pgvector index status by running a simple count query."""
    from ..db import get_pool
    try:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT count(*) AS cnt FROM item_embeddings")
            count = row["cnt"] if row else 0
        return {"ready": count > 0, "count": count, "backend": "pgvector"}
    except Exception as e:
        return {"ready": False, "count": 0, "backend": "pgvector", "error": str(e)}


@router.post("/reindex")
async def reindex():
    """Manually rebuild the pgvector index from Postgres items."""
    result = await build_index()
    return {"ok": True, **result}
