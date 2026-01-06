# app/routes/chat.py
from __future__ import annotations

import uuid
from typing import Optional, List, Dict
from collections import OrderedDict
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ..services.nlu import parse_query
from ..services.rag import (
    answer_from_items,
    ensure_index_ready,
    _get_store,
    extract_order_lines_with_gpt,
)
from ..services.orders import create_order_with_intent

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------- Simple In-Memory Session Store ----------
class SessionStore:
    """Simple in-memory store for conversation history with TTL."""

    def __init__(self, max_sessions: int = 1000, ttl_seconds: int = 3600):
        self._store: Dict[str, dict] = OrderedDict()
        self._max_sessions = max_sessions
        self._ttl = ttl_seconds

    def get_or_create(self, session_id: Optional[str]) -> tuple[str, List[dict]]:
        """Get existing session or create new one. Returns (session_id, history)."""
        self._cleanup_expired()

        if session_id and session_id in self._store:
            self._store[session_id]["last_access"] = time.time()
            return session_id, self._store[session_id]["history"]

        # Create new session
        new_id = session_id or str(uuid.uuid4())
        self._store[new_id] = {"history": [], "last_access": time.time()}

        # Evict oldest if too many
        while len(self._store) > self._max_sessions:
            self._store.popitem(last=False)

        return new_id, []

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to session history."""
        if session_id in self._store:
            self._store[session_id]["history"].append({"role": role, "content": content})
            # Keep only last 20 messages
            self._store[session_id]["history"] = self._store[session_id]["history"][-20:]
            self._store[session_id]["last_access"] = time.time()

    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now - v["last_access"] > self._ttl]
        for k in expired:
            del self._store[k]


# Global session store
_sessions = SessionStore()


# ---------- Chat Schemas ----------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatIn(BaseModel):
    message: str
    session_id: Optional[str] = None  # Session ID for conversation continuity
    history: Optional[List[ChatMessage]] = None  # Optional explicit history (overrides session)


class ChatOut(BaseModel):
    # "chat"    = normal text reply
    # "payment" = created order + PaymentIntent; frontend should render Stripe
    mode: str = "chat"
    message: str
    session_id: Optional[str] = None  # Return session ID for frontend to reuse

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

def _order_lines_from_gpt(user_text: str, history: Optional[List[ChatMessage]] = None) -> List[OrderLineIn]:
    """
    Use GPT to interpret the user's sentence as an order and
    map item names to itemIds from the vectorstore metas.
    Returns [] if nothing could be parsed confidently.
    """
    ensure_index_ready()
    store = _get_store()
    metas = store.metas or []

    # Build context from history if available
    context = ""
    if history:
        context = "\n".join([f"{m.role}: {m.content}" for m in history[-6:]])  # Last 6 messages

    parsed = extract_order_lines_with_gpt(user_text, metas, context)
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
    - Maintains conversation history via session_id for context
    - If the user is clearly asking to place an order ("can you place 2 turkey sandwich")
      we try to parse the sentence directly with GPT, create an order + PaymentIntent,
      and return mode="payment" with clientSecret.
    - Otherwise we fall back to normal RAG answer_from_items.
    """
    # Get or create session for conversation continuity
    session_id, session_history = _sessions.get_or_create(body.session_id)

    # Use explicit history if provided, otherwise use session history
    history = body.history
    if not history and session_history:
        history = [ChatMessage(role=m["role"], content=m["content"]) for m in session_history]

    # Add current user message to session
    _sessions.add_message(session_id, "user", body.message)

    q = parse_query(body.message)

    # 1) Treat clear "order / confirm / place" requests as order intents
    if q.is_order_request:
        order_lines = _order_lines_from_gpt(body.message, history)

        if order_lines:
            order_in = OrderStartIn(
                customerName="Guest",
                customerEmail=None,
                lines=order_lines,
            )
            intent = create_order_with_intent(order_in)

            response_msg = "I've placed your order. Please complete the payment below to confirm."
            _sessions.add_message(session_id, "assistant", response_msg)

            return ChatOut(
                mode="payment",
                message=response_msg,
                session_id=session_id,
                orderId=intent["orderId"],
                clientSecret=intent["clientSecret"],
                total=float(intent["total"]),
            )

        # GPT couldn't confidently parse; gentle fallback prompt
        response_msg = "I can place the order—could you tell me what item and quantity? For example: '2 honey chicken' or '1 mac & cheese'"
        _sessions.add_message(session_id, "assistant", response_msg)
        return ChatOut(
            mode="chat",
            message=response_msg,
            session_id=session_id,
        )

    # 2) Normal RAG answer
    reply = answer_from_items(body.message)
    _sessions.add_message(session_id, "assistant", reply)
    return ChatOut(mode="chat", message=reply, session_id=session_id)
