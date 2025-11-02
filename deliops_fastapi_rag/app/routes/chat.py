from __future__ import annotations
from typing import List, Literal, Optional, Dict
import logging, traceback

from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel

from ..services.rag import answer_from_items

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger("uvicorn.error")

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
    if len(hist) >= 2 and hist[-1] == hist[-2]:
        hist.pop()
    return hist[-40:] if len(hist) > 40 else hist

@router.post("", response_model=ChatResponse)
async def chat_endpoint(body: ChatBody = Body(...)):
    history = _coerce_history(body)
    last_user = next((m["content"] for m in reversed(history) if m["role"] == "user"), None)
    if not last_user:
        raise HTTPException(status_code=422, detail="No user message provided.")
    try:
        answer = answer_from_items(question=last_user, history=history)
        return ChatResponse(answer=answer)
    except HTTPException:
        raise
    except Exception:
        logger.exception("chat failed")
        raise HTTPException(status_code=500, detail="chat failed")
