# app/services/nlu.py
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Dict

# Simple canonicalization for menu entities
CANON = {
    "chicken tenders": ["tenders", "chicken tender", "chicken tenders", "tender"],
    "mac & cheese": ["mac and cheese", "mac & cheese", "mac n cheese", "mac", "mac&cheese"],
    "steak & cheese": ["steak & cheese", "stake & cheese", "steak and cheese", "steak"],
    "hashbrowns": ["hash browns", "hashbrown", "hashbrowns"],
    "tko chicken": ["tko", "tko chicken"],
    # add more aliases as needed
}

def canon_item(q: str) -> Optional[str]:
    s = q.lower()
    for canon, aliases in CANON.items():
        for a in aliases:
            if a in s:
                return canon
    # bare nouns
    if re.search(r"\btender(s)?\b", s):
        return "chicken tenders"
    return None

# Quick intent heuristics — extend as needed
def is_greeting(q: str) -> bool:
    return bool(re.search(r"\b(hi|hello|hey|hola)\b", q, re.I))

def is_thanks(q: str) -> bool:
    return bool(re.search(r"\b(thanks|thank you|appreciate)\b", q, re.I))

def is_goodbye(q: str) -> bool:
    return bool(re.search(r"\b(bye|goodbye|see ya|see you)\b", q, re.I))

def asks_hours(q: str) -> bool:
    return bool(re.search(r"\b(hour|open|close|closing|opening)\b", q, re.I))

def asks_hot_vs_cold(q: str) -> bool:
    return bool(re.search(r"\b(hot|cold)\b.*\b(sandwich|item|menu)", q, re.I)) or \
           bool(re.search(r"\b(hot sandwich|cold sandwich)\b", q, re.I))

def asks_deals(q: str) -> bool:
    return bool(re.search(r"\b(deal|discount|sale|offer|special)\b", q, re.I))

def asks_count(q: str) -> bool:
    return bool(re.search(r"\b(how many|how much|left|available|in stock)\b", q, re.I))

def asks_price(q: str) -> bool:
    return bool(re.search(r"\b(price|cost|how much)\b", q, re.I))

@dataclass
class ParsedQuery:
    raw: str
    item: Optional[str]
    is_greeting: bool
    is_thanks: bool
    is_goodbye: bool
    ask_hours: bool
    ask_hotcold: bool
    ask_deals: bool
    ask_count: bool
    ask_price: bool

def parse_query(q: str) -> ParsedQuery:
    item = canon_item(q)
    return ParsedQuery(
        raw=q,
        item=item,
        is_greeting=is_greeting(q),
        is_thanks=is_thanks(q),
        is_goodbye=is_goodbye(q),
        ask_hours=asks_hours(q),
        ask_hotcold=asks_hot_vs_cold(q),
        ask_deals=asks_deals(q),
        ask_count=asks_count(q),
        ask_price=asks_price(q),
    )
