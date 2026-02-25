from __future__ import annotations

from fastapi import APIRouter, Query

from .agent_runtime import run_agent
from .schemas import (
    AgentChatIn, AgentChatOut,
    RetrieveIn, RetrieveOut,
    InventoryOut,
)
from .tools.lookup_inventory import lookup_inventory
from .tools.retrieve_knowledge import retrieve_knowledge

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat", response_model=AgentChatOut)
async def agent_chat(body: AgentChatIn) -> AgentChatOut:
    """Two-path agent: fast inventory lookup or RAG + LLM."""
    result = await run_agent(body.message, body.history)
    return AgentChatOut(**result)


@router.post("/tools/retrieve", response_model=RetrieveOut)
async def retrieve_endpoint(body: RetrieveIn) -> RetrieveOut:
    """Direct access to the vector-search tool."""
    result = await retrieve_knowledge(body.query, body.top_k)
    return RetrieveOut(**result)


@router.get("/tools/inventory", response_model=InventoryOut)
async def inventory_endpoint(query: str = Query(..., description="Item name to look up")) -> InventoryOut:
    """Direct access to the inventory lookup tool."""
    result = await lookup_inventory(query)
    return InventoryOut(**result)
