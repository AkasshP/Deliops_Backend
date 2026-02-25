"""Shared fixtures: patch heavy dependencies so tests run without credentials."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, AsyncMock
import os

# Set dummy env vars BEFORE any app imports
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")

# Stub stripe
sys.modules.setdefault("stripe", MagicMock())

# Stub openai
_openai_mod = MagicMock()
sys.modules.setdefault("openai", _openai_mod)

# Stub asyncpg so we never touch a real database
_asyncpg_mod = MagicMock()
sys.modules.setdefault("asyncpg", _asyncpg_mod)

import pytest
import numpy as np


# ---------- Fake pgvector store ----------

_fake_items_db: list[dict] = []


async def _fake_upsert(items, embeddings):
    _fake_items_db.clear()
    for item, emb in zip(items, embeddings):
        _fake_items_db.append({**item, "_embedding": emb})
    return len(items)


async def _fake_query(query_embedding, top_k=4, filters=None):
    """Return items from our fake DB, scored by a simple dot product."""
    q = np.array(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)

    scored = []
    for row in _fake_items_db:
        emb = np.array(row["_embedding"], dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        sim = float(np.dot(q, emb))
        scored.append((sim, row))

    scored.sort(key=lambda x: -x[0])

    results = []
    for sim, row in scored[:top_k]:
        if sim < 0.0:  # very permissive threshold for tests
            continue
        results.append({
            "item_id": row.get("id"),
            "item_name": row.get("name"),
            "category": row.get("category"),
            "description": row.get("description", ""),
            "price": row.get("price"),
            "in_stock": row.get("in_stock", True),
            "similarity": round(sim, 4),
        })
    return results


async def _fake_delete_missing(active_ids):
    return 0


# ---------- Fixtures ----------

@pytest.fixture(autouse=True)
def _patch_embeddings(monkeypatch):
    """Patch embed_text / embed_texts to return deterministic vectors."""
    def _fake_embed_texts(texts):
        rng = np.random.RandomState(42)
        return rng.randn(len(texts), 1536).astype(np.float32)

    def _fake_embed_text(text):
        return _fake_embed_texts([text])[0]

    monkeypatch.setattr("app.services.embeddings.embed_texts", _fake_embed_texts)
    monkeypatch.setattr("app.services.embeddings.embed_text", _fake_embed_text)


@pytest.fixture(autouse=True)
def _patch_pgvector(monkeypatch):
    """Replace pgvector_store functions with in-memory fakes."""
    monkeypatch.setattr("app.db.pgvector_store.upsert_items", _fake_upsert)
    monkeypatch.setattr("app.db.pgvector_store.query", _fake_query)
    monkeypatch.setattr("app.db.pgvector_store.delete_missing", _fake_delete_missing)


@pytest.fixture(autouse=True)
def _patch_rag_globals(monkeypatch):
    """Seed the RAG module's _name_map with fake data and populate fake pgvector DB."""
    from app.services import rag

    fake_metas = [
        {"id": "item1", "name": "Turkey", "type": "prepared", "service": "cold",
         "qty": 5, "price": 8.99, "category": "prepared", "in_stock": True},
        {"id": "item2", "name": "Mac & Cheese", "type": "prepared", "service": "hot",
         "qty": 3, "price": 5.99, "category": "prepared", "in_stock": True},
        {"id": "item3", "name": "Bagel", "type": "prepared", "service": "cold",
         "qty": 10, "price": 2.49, "category": "prepared", "in_stock": True},
    ]

    # Build name map
    name_map = {m["name"].strip().lower(): m for m in fake_metas}
    monkeypatch.setattr(rag, "_name_map", name_map)

    # Populate fake pgvector DB with embeddings
    rng = np.random.RandomState(99)
    vecs = rng.randn(len(fake_metas), 1536).astype(np.float32)
    _fake_items_db.clear()
    for item, emb in zip(fake_metas, vecs):
        _fake_items_db.append({**item, "_embedding": emb.tolist()})

    # Patch _get_fresh_item_data to just return the meta as-is (async)
    async def _fake_fresh(meta):
        return meta
    monkeypatch.setattr(rag, "_get_fresh_item_data", _fake_fresh)

    # Patch _format_item_response to sync-compatible version for tests
    async def _fake_format_item_response(meta, include_price=True, include_qty=True):
        name = meta.get("name", "This item")
        qty = meta.get("qty")
        price = meta.get("price")
        parts = []
        if include_qty and isinstance(qty, int):
            if qty > 0:
                parts.append(f"{name} is available with {qty} in stock.")
            else:
                parts.append(f"{name} is currently out of stock.")
        else:
            parts.append(f"{name} is available.")
        if include_price and price is not None:
            parts.append(f"It costs ${float(price):.2f} plus tax.")
        return " ".join(parts)
    monkeypatch.setattr(rag, "_format_item_response", _fake_format_item_response)

    # Patch ensure_index_ready to no-op (async)
    async def _noop(**kw):
        pass
    monkeypatch.setattr(rag, "ensure_index_ready", _noop)


@pytest.fixture()
def client():
    """FastAPI TestClient (sync)."""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
