"""Postgres + pgvector backed vector store for item embeddings."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from . import get_pool
from ..settings import settings


async def upsert_items(
    items: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """
    Insert or update item embeddings in Postgres.

    Each item dict should have: id, name, category, description, price, in_stock.
    Returns the number of rows upserted.
    """
    if not items or not embeddings:
        return 0

    pool = await get_pool()
    now = datetime.now(timezone.utc)

    # pgvector accepts a string like '[0.1, 0.2, ...]' for vector columns
    rows = []
    for item, emb in zip(items, embeddings):
        vec_str = "[" + ",".join(str(float(v)) for v in emb) + "]"
        rows.append((
            item["id"],
            item.get("name", ""),
            item.get("category"),
            item.get("description", ""),
            item.get("price"),
            item.get("in_stock", True),
            now,
            vec_str,
        ))

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO item_embeddings
                (item_id, item_name, category, description, price, in_stock, updated_at, embedding)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7, $8::vector)
            ON CONFLICT (item_id) DO UPDATE SET
                item_name   = EXCLUDED.item_name,
                category    = EXCLUDED.category,
                description = EXCLUDED.description,
                price       = EXCLUDED.price,
                in_stock    = EXCLUDED.in_stock,
                updated_at  = EXCLUDED.updated_at,
                embedding   = EXCLUDED.embedding
            """,
            rows,
        )

    return len(rows)


async def query(
    query_embedding: List[float],
    top_k: int = 4,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Find the top_k most similar items by cosine distance.

    Cosine distance in pgvector: `embedding <=> query` returns a value in [0, 2].
    We convert to cosine similarity: similarity = 1 - distance (range [-1, 1],
    but for normalised embeddings this is [0, 1]).

    Optional filters:
        category  (str)  – exact match on category column
        in_stock  (bool) – filter to only in-stock items

    Returns list of dicts with: item_id, item_name, category, description,
    price, in_stock, similarity.
    Results below RAG_SIMILARITY_THRESHOLD are excluded.
    """
    threshold = settings.rag_similarity_threshold
    vec_str = "[" + ",".join(str(float(v)) for v in query_embedding) + "]"

    where_clauses = ["1=1"]
    params: list[Any] = [vec_str, top_k]

    if filters:
        if filters.get("category"):
            where_clauses.append(f"category = ${len(params) + 1}")
            params.append(filters["category"])
        if filters.get("in_stock") is not None:
            where_clauses.append(f"in_stock = ${len(params) + 1}")
            params.append(bool(filters["in_stock"]))

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT
            item_id,
            item_name,
            category,
            description,
            price,
            in_stock,
            1 - (embedding <=> $1::vector) AS similarity
        FROM item_embeddings
        WHERE {where_sql}
        ORDER BY embedding <=> $1::vector
        LIMIT $2
    """

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    results = []
    for row in rows:
        sim = float(row["similarity"])
        if sim < threshold:
            continue
        results.append({
            "item_id": row["item_id"],
            "item_name": row["item_name"],
            "category": row["category"],
            "description": row["description"],
            "price": float(row["price"]) if row["price"] is not None else None,
            "in_stock": row["in_stock"],
            "similarity": round(sim, 4),
        })

    return results


async def delete_missing(active_item_ids: Set[str]) -> int:
    """
    Remove embeddings for items that are no longer active.
    Returns the number of rows deleted.
    """
    if not active_item_ids:
        return 0

    pool = await get_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            DELETE FROM item_embeddings
            WHERE item_id != ALL($1::text[])
            """,
            list(active_item_ids),
        )
    # result looks like "DELETE 3"
    return int(result.split()[-1])
