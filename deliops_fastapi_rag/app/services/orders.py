from __future__ import annotations
import os
import time
import uuid
from typing import Dict, Any, List
import stripe
from google.cloud import firestore 
from google.cloud.firestore_v1 import Increment
from google.cloud.firestore import Transaction
from .firebase import ensure_firestore
from ..settings import settings
from fastapi import HTTPException

if not settings.stripe_secret_key:
    raise RuntimeError("STRIPE_SECRET_KEY is not set in environment")
stripe.api_key = settings.stripe_secret_key

TAX_RATE = float(os.environ.get("TAX_RATE", "0.0"))  # e.g., 0.0625

def _now():
    return time.time()

def _oid():
    return uuid.uuid4().hex[:24]

def _cents(usd: float) -> int:
    return int(round(usd * 100))

def _load_items_map(ids: List[str]) -> Dict[str, Dict[str, Any]]:
    db = ensure_firestore()
    snaps = [db.collection("items").document(i).get() for i in ids]
    return {s.id: {**(s.to_dict() or {}), "id": s.id} for s in snaps if s.exists}

def _price_lines(lines_in: List[Dict[str, Any]]):
    items = _load_items_map([l["itemId"] for l in lines_in])
    priced, subtotal = [], 0.0
    for l in lines_in:
        it = items.get(l["itemId"])
        if not it or not it.get("active", True):
            raise ValueError("item unavailable")
        unit = float((it.get("price") or {}).get("current") or 0)
        if unit <= 0: raise ValueError("item has no price")
        qty = int(l["qty"])
        line_total = round(unit * qty, 2)
        priced.append({"itemId": it["id"], "name": it["name"], "unitPrice": unit, "qty": qty, "lineTotal": line_total})
        subtotal += line_total
    tax = round(subtotal * TAX_RATE, 2)
    total = round(subtotal + tax, 2)
    return priced, {"subtotal": subtotal, "tax": tax, "total": total, "currency": "USD"}

def create_order_with_intent(body) -> Dict[str, Any]:
    if not settings.stripe_secret_key:
        # nicer than the raw AuthenticationError
        raise HTTPException(status_code=503, detail="Payments not configured")

    lines, amounts = _price_lines([l.model_dump() for l in body.lines])

    db = ensure_firestore()
    order_id = _oid()
    doc = {
        "status": "draft",
        "customer": {"name": body.customerName, "email": body.customerEmail},
        "lines": lines,
        "amounts": amounts,
        "payment": {},
        "createdAt": _now(),
        "updatedAt": _now(),
    }
    db.collection("orders").document(order_id).set(doc)

    intent = stripe.PaymentIntent.create(
        amount=_cents(amounts["total"]),
        currency=amounts["currency"].lower(),
        metadata={"orderId": order_id},
        description=f"Huskies order {order_id[-6:]}",
        automatic_payment_methods={"enabled": True},
    )

    db.collection("orders").document(order_id).set(
        {"payment": {"provider": "stripe", "intentId": intent["id"]}, "status": "pending_payment", "updatedAt": _now()},
        merge=True,
    )
    return {"orderId": order_id, "clientSecret": intent["client_secret"], "total": amounts["total"]}

@firestore.transactional
def _validate_and_decrement(transaction: Transaction, db, order_id: str):
    oref = db.collection("orders").document(order_id)
    osnap = oref.get(transaction=transaction)
    if not osnap.exists:
        raise ValueError("order missing")
    order = osnap.to_dict() or {}
    if order.get("status") == "paid":
        return  # idempotent

    # Check stock first
    for li in order["lines"]:
        iref = db.collection("items").document(li["itemId"])
        isnap = iref.get(transaction=transaction)
        if not isnap.exists:
            raise ValueError("item missing")
        cur = int(((isnap.to_dict() or {}).get("totals") or {}).get("totalQty") or 0)
        if cur < li["qty"]:
            raise ValueError(f"insufficient stock: {li['name']}")

    # Decrement
    for li in order["lines"]:
        iref = db.collection("items").document(li["itemId"])
        transaction.update(iref, {"totals.totalQty": Increment(-li["qty"])})

    transaction.update(oref, {"status": "paid", "updatedAt": _now()})

def finalize_paid_and_decrement(order_id: str, payment_intent_id: str):
    pi = stripe.PaymentIntent.retrieve(payment_intent_id)
    if (pi.metadata or {}).get("orderId") != order_id:
        raise ValueError("PI/order mismatch")
    if pi.status != "succeeded":
        raise ValueError(f"PaymentIntent not succeeded: {pi.status}")

    db = ensure_firestore()
    transaction = db.transaction()
    _validate_and_decrement(transaction, db, order_id)   # 👈 now runs inside a real transaction
    return {"ok": True, "orderId": order_id}

def get_order(order_id: str):
    db = ensure_firestore()
    s = db.collection("orders").document(order_id).get()
    if not s.exists:
        return None
    d = s.to_dict() or {}
    d["id"] = s.id
    return d

def list_orders():
    db = ensure_firestore()
    snaps = db.collection("orders").order_by("createdAt", direction="DESCENDING").limit(200).stream()
    out = []
    for s in snaps:
        d = s.to_dict() or {}
        d["id"] = s.id
        out.append(d)
    return out
