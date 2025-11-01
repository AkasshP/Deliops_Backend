# app/routes/chat.py
from __future__ import annotations

from typing import List, Literal, Optional, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.rag import answer_from_items

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatBody(BaseModel):
    # Accept either a single 'message' or an array of 'messages'
    message: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str


def _coerce_history(body: ChatBody) -> List[Dict[str, str]]:
    """
    Normalize payload into a list of {role, content}.
    Supports:
      { "messages": [{role, content}, ...] }  OR  { "message": "hi" }
    """
    hist: List[Dict[str, str]] = []

    if body.messages:
        for m in body.messages:
            content = (m.content or "").strip()
            if not content:
                continue
            role = "user" if m.role == "user" else "assistant"
            hist.append({"role": role, "content": content})

    if body.message:
        msg = body.message.strip()
        if msg:
            hist.append({"role": "user", "content": msg})

    # If both paths produced the same trailing message, drop the duplicate
    if len(hist) >= 2 and hist[-1] == hist[-2]:
        hist.pop()

    # Optional: cap history length to last 20 turns
    if len(hist) > 40:
        hist = hist[-40:]

    return hist


@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatBody):
    history = _coerce_history(body)
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)

    if not last_user:
        raise HTTPException(status_code=422, detail="No user message provided.")

    try:
        answer = answer_from_items(question=last_user, history=history)
        return ChatResponse(answer=answer)
    except Exception as e:
        # You can add a server log here if desired
        raise HTTPException(status_code=422, detail=f"chat failed: {e}")
