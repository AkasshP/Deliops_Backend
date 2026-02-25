# app/services/items.py
from __future__ import annotations

from typing import List, Dict, Any, Optional

from ..db import get_pool


# --- Row → nested dict helper ------------------------------------------------

def _row_to_dict(row) -> Dict[str, Any]:
    """Convert a flat Postgres row into the nested shape routes/Pydantic expect."""
    return {
        "id": row["id"],
        "name": row["name"],
        "type": row["type"],
        "service": row["service"],
        "uom": row["uom"],
        "category": row["category"],
        "public": row["public"],
        "active": row["active"],
        "price": {"current": float(row["price_current"])} if row["price_current"] is not None else None,
        "totals": {
            "floorQty": row["floor_qty"],
            "backQty": row["back_qty"],
            "totalQty": row["total_qty"],
        },
        "imageUrl": row["image_url"],
    }


# --- READ HELPERS -------------------------------------------------------------

async def list_public_items() -> List[Dict[str, Any]]:
    """Public + active items for the guest chat/dashboard."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM items WHERE public = TRUE AND active = TRUE"
        )
    return [_row_to_dict(r) for r in rows]


async def list_items(
    public: Optional[bool] = None,
    active: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Admin list: optional filters for public/active."""
    clauses = ["1=1"]
    params: list[Any] = []
    if public is not None:
        params.append(bool(public))
        clauses.append(f"public = ${len(params)}")
    if active is not None:
        params.append(bool(active))
        clauses.append(f"active = ${len(params)}")

    sql = f"SELECT * FROM items WHERE {' AND '.join(clauses)}"
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
    return [_row_to_dict(r) for r in rows]


async def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    if not item_id:
        return None
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM items WHERE id = $1", item_id)
    if row is None:
        return None
    return _row_to_dict(row)


async def create_item(payload: Dict[str, Any]) -> Dict[str, Any]:
    name: str = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")

    item_id = payload.get("id") or _slug(name)
    price_obj = payload.get("price") or {}
    totals_obj = payload.get("totals") or {}

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO items (id, name, type, service, uom, category,
                               public, active, price_current,
                               floor_qty, back_qty, total_qty, image_url)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
            ON CONFLICT (id) DO UPDATE SET
                name=EXCLUDED.name, type=EXCLUDED.type, service=EXCLUDED.service,
                uom=EXCLUDED.uom, category=EXCLUDED.category, public=EXCLUDED.public,
                active=EXCLUDED.active, price_current=EXCLUDED.price_current,
                floor_qty=EXCLUDED.floor_qty, back_qty=EXCLUDED.back_qty,
                total_qty=EXCLUDED.total_qty, image_url=EXCLUDED.image_url,
                updated_at=NOW()
            RETURNING *
            """,
            item_id,
            name,
            payload.get("type", "ingredient"),
            payload.get("service", "none"),
            payload.get("uom", "ea"),
            payload.get("category"),
            bool(payload.get("public", False)),
            bool(payload.get("active", True)),
            float(price_obj.get("current")) if price_obj.get("current") is not None else None,
            int(totals_obj.get("floorQty") or 0),
            int(totals_obj.get("backQty") or 0),
            int(totals_obj.get("totalQty") or 0),
            payload.get("imageUrl"),
        )
    return _row_to_dict(row)


async def update_item(item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not item_id:
        raise ValueError("item_id required")

    # Build SET clause dynamically from provided fields
    col_map = {
        "name": "name",
        "type": "type",
        "service": "service",
        "uom": "uom",
        "category": "category",
        "public": "public",
        "active": "active",
        "imageUrl": "image_url",
    }
    sets: list[str] = []
    params: list[Any] = [item_id]  # $1 = item_id

    for key, col in col_map.items():
        if key in payload:
            params.append(payload[key])
            sets.append(f"{col} = ${len(params)}")

    # Handle nested price
    if "price" in payload and payload["price"]:
        price_obj = payload["price"]
        if isinstance(price_obj, dict) and "current" in price_obj:
            params.append(float(price_obj["current"]))
            sets.append(f"price_current = ${len(params)}")

    # Handle nested totals
    if "totals" in payload and payload["totals"]:
        totals_obj = payload["totals"]
        if isinstance(totals_obj, dict):
            for fk, col in [("floorQty", "floor_qty"), ("backQty", "back_qty"), ("totalQty", "total_qty")]:
                if fk in totals_obj:
                    params.append(int(totals_obj[fk]))
                    sets.append(f"{col} = ${len(params)}")

    if not sets:
        # Nothing to update — just return current
        item = await get_item(item_id)
        if not item:
            raise ValueError("item not found")
        return item

    sets.append("updated_at = NOW()")
    sql = f"UPDATE items SET {', '.join(sets)} WHERE id = $1 RETURNING *"

    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *params)
    if row is None:
        raise ValueError("item not found")
    return _row_to_dict(row)


async def delete_item(item_id: str) -> None:
    if not item_id:
        raise ValueError("item_id required")
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM items WHERE id = $1", item_id)


# --- UTILS --------------------------------------------------------------------

def _slug(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("_", " ")
        .encode("ascii", "ignore").decode("ascii")
        .replace("'", "")
        .replace(".", " ")
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "-")
    )
