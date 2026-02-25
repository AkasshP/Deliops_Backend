# app/services/orders.py
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List

import stripe
from fastapi import HTTPException

from ..db import get_pool
from ..settings import settings

if not settings.stripe_secret_key:
    raise RuntimeError("STRIPE_SECRET_KEY is not set in environment")
stripe.api_key = settings.stripe_secret_key

TAX_RATE = float(os.environ.get("TAX_RATE", "0.0"))  # e.g., 0.0625


def _now():
    return datetime.now(timezone.utc)


def _oid():
    return uuid.uuid4().hex[:24]


def _cents(usd: float) -> int:
    return int(round(usd * 100))


def _order_row_to_dict(row) -> Dict[str, Any]:
    """Convert a flat Postgres order row into the nested shape routes expect."""
    return {
        "id": row["id"],
        "status": row["status"],
        "customer": {"name": row["customer_name"], "email": row["customer_email"]},
        "lines": json.loads(row["lines"]) if isinstance(row["lines"], str) else row["lines"],
        "amounts": {
            "subtotal": float(row["subtotal"]),
            "tax": float(row["tax"]),
            "total": float(row["total"]),
            "currency": row["currency"],
        },
        "payment": {
            "provider": row["payment_provider"],
            "intentId": row["payment_intent_id"],
        },
        "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
        "updatedAt": row["updated_at"].isoformat() if row["updated_at"] else None,
    }


async def _load_items_map(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Load items by IDs from Postgres."""
    if not ids:
        return {}
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM items WHERE id = ANY($1::text[])", ids
        )
    from .items import _row_to_dict
    return {r["id"]: _row_to_dict(r) for r in rows}


async def _price_lines(lines_in: List[Dict[str, Any]]):
    items = await _load_items_map([l["itemId"] for l in lines_in])
    priced, subtotal = [], 0.0
    for l in lines_in:
        it = items.get(l["itemId"])
        if not it or not it.get("active", True):
            raise ValueError("item unavailable")
        unit = float((it.get("price") or {}).get("current") or 0)
        if unit <= 0:
            raise ValueError("item has no price")
        qty = int(l["qty"])
        line_total = round(unit * qty, 2)
        priced.append({
            "itemId": it["id"], "name": it["name"],
            "unitPrice": unit, "qty": qty, "lineTotal": line_total,
        })
        subtotal += line_total
    tax = round(subtotal * TAX_RATE, 2)
    total = round(subtotal + tax, 2)
    return priced, {"subtotal": subtotal, "tax": tax, "total": total, "currency": "USD"}


async def create_order_with_intent(body) -> Dict[str, Any]:
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=503, detail="Payments not configured")

    lines, amounts = await _price_lines([l.model_dump() for l in body.lines])

    pool = await get_pool()
    order_id = _oid()
    now = _now()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO orders (id, status, customer_name, customer_email, lines,
                                subtotal, tax, total, currency, created_at, updated_at)
            VALUES ($1, 'draft', $2, $3, $4::jsonb, $5, $6, $7, $8, $9, $9)
            """,
            order_id,
            getattr(body, "customerName", None),
            getattr(body, "customerEmail", None),
            json.dumps(lines),
            amounts["subtotal"],
            amounts["tax"],
            amounts["total"],
            amounts["currency"],
            now,
        )

    intent = stripe.PaymentIntent.create(
        amount=_cents(amounts["total"]),
        currency=amounts["currency"].lower(),
        metadata={"orderId": order_id},
        description=f"Huskies order {order_id[-6:]}",
        automatic_payment_methods={"enabled": True},
    )

    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE orders
            SET payment_provider = 'stripe',
                payment_intent_id = $2,
                status = 'pending_payment',
                updated_at = $3
            WHERE id = $1
            """,
            order_id,
            intent["id"],
            _now(),
        )

    return {
        "orderId": order_id,
        "clientSecret": intent["client_secret"],
        "total": amounts["total"],
    }


async def finalize_paid_and_decrement(order_id: str, payment_intent_id: str):
    pi = stripe.PaymentIntent.retrieve(payment_intent_id)
    if (pi.metadata or {}).get("orderId") != order_id:
        raise ValueError("PI/order mismatch")
    if pi.status != "succeeded":
        raise ValueError(f"PaymentIntent not succeeded: {pi.status}")

    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Lock the order row
            order_row = await conn.fetchrow(
                "SELECT * FROM orders WHERE id = $1 FOR UPDATE", order_id
            )
            if not order_row:
                raise ValueError("order missing")
            if order_row["status"] == "paid":
                return {"ok": True, "orderId": order_id}  # idempotent

            lines = json.loads(order_row["lines"]) if isinstance(order_row["lines"], str) else order_row["lines"]

            # Sort item IDs to prevent deadlocks when locking items
            item_ids = sorted({li["itemId"] for li in lines})

            # Lock item rows in consistent order
            for iid in item_ids:
                item_row = await conn.fetchrow(
                    "SELECT id, total_qty, name FROM items WHERE id = $1 FOR UPDATE", iid
                )
                if not item_row:
                    raise ValueError(f"item missing: {iid}")

            # Check stock and decrement
            for li in lines:
                item_row = await conn.fetchrow(
                    "SELECT total_qty FROM items WHERE id = $1", li["itemId"]
                )
                cur = item_row["total_qty"]
                if cur < li["qty"]:
                    raise ValueError(f"insufficient stock: {li['name']}")

                await conn.execute(
                    "UPDATE items SET total_qty = total_qty - $2, updated_at = NOW() WHERE id = $1",
                    li["itemId"],
                    li["qty"],
                )

            # Mark order as paid
            await conn.execute(
                "UPDATE orders SET status = 'paid', updated_at = $2 WHERE id = $1",
                order_id,
                _now(),
            )

    return {"ok": True, "orderId": order_id}


async def get_order(order_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM orders WHERE id = $1", order_id)
    if not row:
        return None
    return _order_row_to_dict(row)


async def list_orders():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM orders ORDER BY created_at DESC LIMIT 200"
        )
    return [_order_row_to_dict(r) for r in rows]
