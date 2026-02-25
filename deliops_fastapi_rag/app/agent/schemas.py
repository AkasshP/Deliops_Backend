from __future__ import annotations

from typing import Optional, List, Dict, Any

from pydantic import BaseModel


class AgentChatIn(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = None


class AgentChatOut(BaseModel):
    reply: str
    used_tools: List[str]
    path: str


class RetrieveIn(BaseModel):
    query: str
    top_k: Optional[int] = None


class RetrieveOut(BaseModel):
    tool: str
    query: str
    results: List[Dict[str, Any]]
    count: int


class InventoryItemOut(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    service: Optional[str] = None
    qty: Optional[int] = None
    price: Optional[float] = None


class InventoryOut(BaseModel):
    tool: str
    query: str
    found: bool
    item: Optional[InventoryItemOut] = None
