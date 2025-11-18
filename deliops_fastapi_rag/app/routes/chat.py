# app/routes/chat.py
from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter
from pydantic import BaseModel

from ..services.nlu import parse_query
from ..services.rag import (
    answer_from_items,
    ensure_index_ready,
    _get_store,                    # ok even though it's "private"
    extract_order_lines_with_gpt,
)
from ..services.orders import create_order_with_intent

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------- Chat Schemas ----------

class ChatIn(BaseModel):
    message: str


class ChatOut(BaseModel):
    # "chat"    = normal text reply
    # "payment" = created order + PaymentIntent; frontend should render Stripe
    mode: str = "chat"
    message: str

    # present only when mode == "payment"
    orderId: Optional[str] = None
    clientSecret: Optional[str] = None
    total: Optional[float] = None


# ---------- Lightweight order schemas (local to chat) ----------

class OrderLineIn(BaseModel):
    itemId: str
    qty: int


class OrderStartIn(BaseModel):
    customerName: Optional[str] = None
    customerEmail: Optional[str] = None
    lines: List[OrderLineIn]


# ---------- Helpers ----------

def _order_lines_from_gpt(user_text: str) -> List[OrderLineIn]:
    """
    Use GPT to interpret the user's sentence as an order and
    map item names to itemIds from the vectorstore metas.
    Returns [] if nothing could be parsed confidently.
    """
    ensure_index_ready()
    store = _get_store()
    metas = store.metas or []

    parsed = extract_order_lines_with_gpt(user_text, metas)
    if not parsed:
        return []

    lines: List[OrderLineIn] = []
    for ln in parsed:
        name = (ln["name"] or "").strip().lower()
        qty = int(ln["qty"])
        meta = next(
            (m for m in metas if (m.get("name") or "").strip().lower() == name),
            None,
        )
        if not meta or not meta.get("id"):
            continue

        lines.append(OrderLineIn(itemId=meta["id"], qty=qty))

    return lines


# ---------- Route ----------

@router.post("", response_model=ChatOut)
def chat_endpoint(body: ChatIn) -> ChatOut:
    """
    Main chat endpoint used by the guest UI.

    Behaviours:
    - If the user is clearly asking to place an order ("can you place 2 turkey sandwich")
      we try to parse the sentence directly with GPT, create an order + PaymentIntent,
      and return mode="payment" with clientSecret.
    - Otherwise we fall back to normal RAG answer_from_items.
    """
    q = parse_query(body.message)

    # 1) Treat clear "order / confirm / place" requests as order intents
    if q.is_order_request:
        order_lines = _order_lines_from_gpt(body.message)

        if order_lines:
            order_in = OrderStartIn(
                customerName="Guest",
                customerEmail=None,
                lines=order_lines,
            )
            intent = create_order_with_intent(order_in)

            return ChatOut(
                mode="payment",
                message=(
                    "I’ve placed your order. Please complete the payment below to confirm."
                ),
                orderId=intent["orderId"],
                clientSecret=intent["clientSecret"],
                total=float(intent["total"]),
            )

        # GPT couldn’t confidently parse; gentle fallback prompt
        return ChatOut(
            mode="chat",
            message=(
                "I can place the order—could you list it in one line like "
                "“2 chicken tenders, 1 mac & cheese”?"
            ),
        )

    # 2) Normal RAG answer
    reply = answer_from_items(body.message)
    return ChatOut(mode="chat", message=reply)
