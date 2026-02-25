from __future__ import annotations

from typing import Dict, Any

from ...services.rag import (
    _exact_or_contains_lookup,
    _get_fresh_item_data,
    ensure_index_ready,
)


async def lookup_inventory(query: str) -> Dict[str, Any]:
    """Look up an item by name. Returns found=True with item data, or found=False."""
    await ensure_index_ready()

    meta = _exact_or_contains_lookup(query)
    if meta is None:
        return {"tool": "lookup_inventory", "query": query, "found": False, "item": None}

    fresh = await _get_fresh_item_data(meta)
    return {
        "tool": "lookup_inventory",
        "query": query,
        "found": True,
        "item": {
            "id": fresh.get("id"),
            "name": fresh.get("name"),
            "type": fresh.get("type"),
            "service": fresh.get("service"),
            "qty": fresh.get("qty"),
            "price": fresh.get("price"),
        },
    }
