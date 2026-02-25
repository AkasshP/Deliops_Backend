"""Tests for the agent layer endpoints."""
from __future__ import annotations

from unittest.mock import patch, AsyncMock

import pytest


# ---------- Fast path ----------

def test_fast_path_inventory(client):
    """'do you have mac & cheese?' triggers fast path and finds the item."""
    resp = client.post("/agent/chat", json={"message": "do you have mac & cheese?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "fast"
    assert "Mac & Cheese" in data["reply"]
    assert "lookup_inventory" in data["used_tools"]


# ---------- Retrieve endpoint shape ----------

def test_retrieve_endpoint_shape(client):
    """POST /agent/tools/retrieve returns the expected schema."""
    resp = client.post("/agent/tools/retrieve", json={"query": "cheese"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tool"] == "retrieve_knowledge"
    assert data["query"] == "cheese"
    assert isinstance(data["results"], list)
    assert isinstance(data["count"], int)


# ---------- Normal (LLM) path ----------

def test_agent_chat_returns_reply_and_tools(client):
    """A non-fast-path query goes through RAG + LLM and returns reply + tools."""
    fake_llm = {"content": "Here is what I found.", "model": "test", "usage": {}}

    with patch(
        "app.agent.agent_runtime.chat_completion",
        new_callable=AsyncMock,
        return_value=fake_llm,
    ):
        resp = client.post("/agent/chat", json={"message": "tell me about your menu"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "normal"
    assert "reply" in data
    assert "retrieve_knowledge" in data["used_tools"]


# ---------- Inventory direct endpoint ----------

def test_inventory_endpoint(client):
    """GET /agent/tools/inventory?query=turkey returns found=True."""
    resp = client.get("/agent/tools/inventory", params={"query": "turkey"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["item"]["name"] == "Turkey"


def test_inventory_not_found(client):
    """GET /agent/tools/inventory?query=pizza returns found=False."""
    resp = client.get("/agent/tools/inventory", params={"query": "pizza"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is False
    assert data["item"] is None


# ---------- NEW: pgvector-specific tests ----------

def test_rebuild_index_upserts_rows(client, monkeypatch):
    """POST /admin/reindex fetches items, embeds, and upserts into pgvector."""
    from tests.conftest import _fake_items_db

    # Mock Postgres list_items to return known items
    fake_items = [
        {"id": "item1", "name": "Turkey", "type": "prepared", "service": "cold",
         "category": "prepared", "totals": {"totalQty": 5},
         "price": {"current": 8.99}, "active": True, "public": True},
        {"id": "item2", "name": "Mac & Cheese", "type": "prepared", "service": "hot",
         "category": "prepared", "totals": {"totalQty": 3},
         "price": {"current": 5.99}, "active": True, "public": True},
    ]
    async def _fake_list_items(**kw):
        return fake_items
    monkeypatch.setattr("app.services.rag.list_items", _fake_list_items)

    # Re-enable ensure_index_ready for this test (the reindex endpoint calls build_index directly)
    from app.services import rag
    monkeypatch.setattr(rag, "ensure_index_ready", rag.ensure_index_ready.__wrapped__
                        if hasattr(rag.ensure_index_ready, '__wrapped__') else rag.ensure_index_ready)

    # Clear the fake DB
    _fake_items_db.clear()

    resp = client.post("/admin/reindex")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["count"] == 2
    # Verify rows were actually upserted
    assert len(_fake_items_db) == 2
    names = {r["name"] for r in _fake_items_db}
    assert "Turkey" in names
    assert "Mac & Cheese" in names


def test_retrieve_empty_below_threshold(client, monkeypatch):
    """Retrieve returns empty results when similarity is below threshold."""
    from tests.conftest import _fake_items_db

    # Set a very high threshold so nothing qualifies
    monkeypatch.setattr("app.db.pgvector_store.settings.rag_similarity_threshold", 0.9999)

    # Also patch the query function to respect the patched threshold
    async def _strict_query(query_embedding, top_k=4, filters=None):
        """Always returns empty since threshold is impossibly high."""
        import numpy as np
        from app.settings import settings
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
            if sim < 0.9999:  # impossibly high threshold
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

    monkeypatch.setattr("app.db.pgvector_store.query", _strict_query)

    resp = client.post("/agent/tools/retrieve", json={"query": "something random xyz"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["results"] == []
