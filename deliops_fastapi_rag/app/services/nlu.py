# app/services/nlu.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from ..settings import settings


@dataclass
class ParsedQuery:
    text: str

    # small-talk
    is_greeting: bool = False
    is_thanks: bool = False
    is_goodbye: bool = False

    # intents
    ask_hours: bool = False         # store hours
    ask_deals: bool = False         # late-night deals / promos
    ask_price: bool = False         # price of an item
    ask_count: bool = False         # how many in stock
    ask_hotcold: bool = False       # hot vs cold sandwiches
    ask_payment: bool = False       # payment methods

    is_order_request: bool = False   # "can you place 2 turkey sandwich"
    is_order_confirm: bool = False   # "yes, confirm the order"

    # free-text item name if we can guess it
    item: Optional[str] = None


# Lazy-loaded OpenAI client
_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    global _client
    if _client is None and settings.openai_api_key:
        _client = OpenAI(api_key=settings.openai_api_key)
    return _client


def parse_query(text: str) -> ParsedQuery:
    """
    Use OpenAI to classify user intent for the deli bot.
    Falls back to empty ParsedQuery if OpenAI is not available.
    """
    t = (text or "").strip()
    pq = ParsedQuery(text=t)

    client = _get_client()
    if not client or not t:
        return pq

    system_prompt = """You are an intent classifier for a deli restaurant chatbot.
Analyze the user message and return a JSON object with these boolean fields:
- is_greeting: true if user is saying hi/hello/hey
- is_thanks: true if user is thanking
- is_goodbye: true if user is saying bye
- ask_hours: true if asking about store hours/opening/closing times
- ask_deals: true if asking about deals/discounts/promotions
- ask_price: true if asking about price of a specific item
- ask_count: true if asking about stock/availability/quantity
- ask_hotcold: true if asking about hot vs cold sandwiches
- ask_payment: true if asking about payment methods/how to pay
- is_order_request: true if user wants to place/make an order
- is_order_confirm: true if user is confirming an order
- item: the item name if user is asking about a specific menu item, else null

Return ONLY valid JSON, no explanation."""

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model or "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": t},
            ],
            temperature=0,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or "{}"

        # Clean up response (remove markdown code blocks if present)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)

        pq.is_greeting = bool(data.get("is_greeting"))
        pq.is_thanks = bool(data.get("is_thanks"))
        pq.is_goodbye = bool(data.get("is_goodbye"))
        pq.ask_hours = bool(data.get("ask_hours"))
        pq.ask_deals = bool(data.get("ask_deals"))
        pq.ask_price = bool(data.get("ask_price"))
        pq.ask_count = bool(data.get("ask_count"))
        pq.ask_hotcold = bool(data.get("ask_hotcold"))
        pq.ask_payment = bool(data.get("ask_payment"))
        pq.is_order_request = bool(data.get("is_order_request"))
        pq.is_order_confirm = bool(data.get("is_order_confirm"))
        pq.item = data.get("item") if data.get("item") else None

    except Exception:
        # If OpenAI fails, return empty ParsedQuery (RAG will handle it)
        pass

    return pq
