from __future__ import annotations

import re
from typing import Dict, Any, List, Optional

from .tools.lookup_inventory import lookup_inventory
from .tools.retrieve_knowledge import retrieve_knowledge
from ..llm.openrouter_client import chat_completion
from ..services.rag import STORE_RULES

_FAST_PATTERNS = [
    re.compile(r"\b(?:do you (?:have|carry|sell|stock))\b", re.I),
    re.compile(r"\b(?:is there|are there|got any)\b", re.I),
    re.compile(r"\b(?:how much (?:is|does|for))\b", re.I),
    re.compile(r"\b(?:what(?:'s| is) the price)\b", re.I),
    re.compile(r"\b(?:in stock|available)\b", re.I),
    re.compile(r"\b(?:price of|cost of)\b", re.I),
]

_hrs = STORE_RULES["hours"]
SYSTEM_PROMPT = (
    "You are a friendly deli assistant for Huskies Deli.\n"
    "Use only facts from CONTEXT. Never invent items, prices, or hours.\n"
    f"Store hours: {_hrs['open']} to {_hrs['close']}. "
    f"Hot sandwiches until {STORE_RULES['hot_sandwich_cutoff']}.\n"
    f"Late-night deals start at {STORE_RULES['late_deals_start']}.\n"
    "Keep answers short (1-3 sentences), clear, and polite.\n"
    "If you cannot find the answer in CONTEXT, say so honestly."
)


def _matches_fast_pattern(message: str) -> bool:
    return any(p.search(message) for p in _FAST_PATTERNS)


def _format_inventory_reply(result: Dict[str, Any]) -> str:
    """Build a human-friendly reply from a lookup_inventory result."""
    item = result["item"]
    name = item.get("name", "that item")
    qty = item.get("qty")
    price = item.get("price")

    parts = []
    if isinstance(qty, int):
        if qty > 0:
            parts.append(f"Yes, we have {name}! We have {qty} in stock.")
        else:
            parts.append(f"We carry {name}, but it's currently sold out.")
    else:
        parts.append(f"Yes, we have {name}.")

    if price is not None:
        parts.append(f"It costs ${float(price):.2f} plus tax.")

    return " ".join(parts)


async def run_agent(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Main agent entry point.
    Returns {"reply": str, "used_tools": list, "path": "fast"|"normal"}.
    """
    used_tools: List[str] = []

    # --- FAST PATH ---
    if _matches_fast_pattern(message):
        inv = await lookup_inventory(message)
        used_tools.append("lookup_inventory")

        if inv["found"]:
            return {
                "reply": _format_inventory_reply(inv),
                "used_tools": used_tools,
                "path": "fast",
            }
        # Not found → fall through to normal path

    # --- NORMAL PATH ---
    # 1. Retrieve relevant knowledge
    knowledge = await retrieve_knowledge(message)
    used_tools.append("retrieve_knowledge")

    # 2. Build context from results
    context_lines = []
    for r in knowledge["results"]:
        line = r.get("name", "")
        if r.get("qty") is not None:
            line += f" | In stock: {r['qty']}"
        if r.get("price") is not None:
            line += f" | Price: ${float(r['price']):.2f}"
        context_lines.append(line)

    context = "\n".join(context_lines) if context_lines else "No matching items found."

    # 3. Build messages for LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{message}",
    })

    # 4. Call OpenRouter LLM
    llm_result = await chat_completion(messages)
    used_tools.append("chat_completion")

    return {
        "reply": llm_result["content"],
        "used_tools": used_tools,
        "path": "normal",
    }
