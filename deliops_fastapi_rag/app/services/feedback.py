# app/services/feedback.py
from __future__ import annotations

import uuid
from typing import Dict, Any, List
from datetime import datetime, timezone

from ..db import get_pool


async def create_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a feedback row:
      { name?, email?, message, rating?, createdAt }
    """
    name = (payload.get("name") or "").strip() or None
    email = (payload.get("email") or "").strip() or None
    message = (payload.get("message") or "").strip()
    rating = int(payload.get("rating") or 0)

    if not message:
        raise ValueError("message is required")
    if rating < 0 or rating > 5:
        raise ValueError("rating must be between 0 and 5")

    fb_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO feedback (id, name, email, message, rating, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            fb_id, name, email, message, rating, now,
        )

    return {
        "id": fb_id,
        "name": name,
        "email": email,
        "message": message,
        "rating": rating,
        "createdAt": now.isoformat(),
    }


async def list_feedback(limit: int = 100) -> List[Dict[str, Any]]:
    """Return latest feedback (newest first)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT $1", limit
        )

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r["id"],
            "name": r["name"],
            "email": r["email"],
            "message": r["message"],
            "rating": r["rating"],
            "createdAt": r["created_at"].isoformat() if r["created_at"] else None,
        })
    return out
