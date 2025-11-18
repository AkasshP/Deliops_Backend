# app/services/nlu.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

    is_order_request: bool = False   # “can you place 2 turkey sandwich”
    is_order_confirm: bool = False   # “yes, confirm the order”

    # free-text item name if we can guess it
    item: Optional[str] = None


def parse_query(text: str) -> ParsedQuery:
    """
    Very simple keyword-based NLU that is 'good enough' for the deli bot.
    The RAG layer still controls the actual answer.
    """
    t = (text or "").strip()
    tl = t.lower()

    pq = ParsedQuery(text=t)

    # --- small talk ---
    if any(w in tl for w in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]):
        pq.is_greeting = True
    if "thank" in tl or "thx" in tl:
        pq.is_thanks = True
    if any(w in tl for w in ["bye", "goodbye", "see you", "see ya", "later"]):
        pq.is_goodbye = True

    # --- store info / rules ---
    if any(w in tl for w in ["hour", "open", "close", "closing time", "closing-time"]):
        pq.ask_hours = True
    if any(w in tl for w in ["deal", "offer", "promo", "discount", "late night", "late-night"]):
        pq.ask_deals = True
    if any(w in tl for w in ["hot sandwich", "cold sandwich", "hot or cold", "hot vs cold"]):
        pq.ask_hotcold = True

    # --- item-centric questions ---
    if any(w in tl for w in ["price", "cost", "how much", "$"]):
        pq.ask_price = True
    if any(w in tl for w in ["how many", "quantity", "qty", "in stock", "left", "available"]):
        pq.ask_count = True

    if any(w in tl for w in ["place my order", "place the order", "order this", "order now",
                             "i want to order", "can you place", "can you order", "add to my order"]):
        pq.is_order_request = True

    if any(w in tl for w in ["confirm my order", "confirm the order",
                             "yes go ahead", "yes place it", "looks good, place it"]):
        pq.is_order_confirm = True

    # crude guess for an "item name" substring
    # everything before '?' or 'price' / 'cost' etc. tends to be the item.
    # This is optional; RAG still does semantic search.
    possible = t
    for sep in ["?", " price", " cost", " how much"]:
        idx = possible.lower().find(sep)
        if idx > 0:
            possible = possible[:idx]
            break

    possible = possible.strip()
    if possible and len(possible.split()) <= 6:
        pq.item = possible

    return pq
