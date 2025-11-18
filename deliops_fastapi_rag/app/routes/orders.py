# app/routes/orders.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..services.orders import create_order_with_intent, finalize_paid_and_decrement, list_orders


router = APIRouter(prefix="/orders", tags=["orders"])


class OrderLineIn(BaseModel):
    itemId: str
    qty: int


class StartOrderBody(BaseModel):
    customerName: str | None = None
    customerEmail: str | None = None
    lines: list[OrderLineIn]


class FinalizeBody(BaseModel):
    # we keep camelCase to match the frontend JSON exactly
    paymentIntentId: str


@router.post("/intent")
def start_order_intent(body: StartOrderBody):
    return create_order_with_intent(body)


@router.post("/{order_id}/finalize")
def finalize_order(order_id: str, body: FinalizeBody):
    try:
        return finalize_paid_and_decrement(order_id, body.paymentIntentId)
    except Exception as e:
        # Turn opaque 500s into readable 400s so you can see the message
        raise HTTPException(status_code=400, detail=str(e))
    

@router.get("")
def list_orders_endpoint():
    """
    Return recent orders for the admin UI.
    We just forward the dictionaries from services.orders.list_orders().
    """
    return list_orders()