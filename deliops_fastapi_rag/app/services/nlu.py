from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Dict, Set, Tuple
import re
import time

from .items import list_items


@dataclass
class ParsedQuery:
    # small intents
    is_greeting: bool = False
    is_thanks: bool = False
    is_goodbye: bool = False

    # question intents
    ask_hours: bool = False
    ask_deals: bool = False
    ask_price: bool = False
    ask_count: bool = False
    ask_hotcold: bool = False

    # target (canonical item name if detected)
    item: Optional[str] = None


# -------------------------------
# Inventory name cache (lazy)
# -------------------------------

# Cache item names for ~60s to avoid hitting Firestore every turn when under load
_NAME_CACHE: Dict[str, object] = {
    "names": set(),   # type: ignore[assignment]
    "stamp": 0.0,
}

_CACHE_TTL = 60.0  # seconds


def _load_inventory_names() -> Set[str]:
    now = time.time()
    names: Set[str] = _NAME_CACHE.get("names", set())  # type: ignore[assignment]
    stamp: float = _NAME_CACHE.get("stamp", 0.0)        # type: ignore[assignment]

    if names and now - stamp < _CACHE_TTL:
        return names

    try:
        docs = list_items(public=True, active=True)
        fresh: Set[str] = set()
        for d in docs:
            n = (d.get("name") or "").strip()
            if n:
                fresh.add(n)
        if fresh:
            _NAME_CACHE["names"] = fresh
            _NAME_CACHE["stamp"] = now
            return fresh
    except Exception:
        pass

    return names or set()


# -------------------------------
# Simple helpers
# -------------------------------

_PUNCT_RE = re.compile(r"[^\w\s&'\-]+", re.UNICODE)
_WS_RE = re.compile(r"\s+", re.UNICODE)


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def _wboundary_contains(text: str, needle: str) -> bool:
    """
    Word-boundary-ish contains (works for multi-word names).
    Example: text="do you have turkey?" needle="turkey" -> True
    """
    # escape regex special chars in needle words
    pattern = r"\b" + re.escape(needle) + r"\b"
    return re.search(pattern, text) is not None


def _best_item_match(msg: str, inv_names: Set[str]) -> Optional[str]:
    """
    Choose a canonical inventory name from `inv_names` that best matches `msg`.
    Strategy:
      1) exact word-boundary match (multi-word OK)
      2) fallback: containment either way for robustness ("tenders" ~ "Chicken tenders")
      3) tie-break by longest name
    """
    if not msg or not inv_names:
        return None

    msg_n = _normalize(msg)

    # Prepare normalized mapping
    norm_to_orig: Dict[str, str] = {}
    for n in inv_names:
        nn = _normalize(n)
        if nn:
            norm_to_orig[nn] = n

    candidates: List[Tuple[int, str, str]] = []  # (score, norm_name, orig_name)

    for nn, orig in norm_to_orig.items():
        # 1) word-boundary exact match
        if _wboundary_contains(msg_n, nn):
            # higher score = better; longer wins
            candidates.append((3, nn, orig))
            continue

        # 2) soft containment (either direction), but avoid super-short overlaps
        if (len(nn) >= 3 and (nn in msg_n or msg_n in nn)):
            candidates.append((2, nn, orig))
            continue

        # 3) very soft: overlapping token sets
        msg_tokens = set(msg_n.split())
        name_tokens = set(nn.split())
        if msg_tokens & name_tokens:
            candidates.append((1, nn, orig))

    if not candidates:
        return None

    # sort by (score desc, length desc) then pick top
    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return candidates[0][2]


# -------------------------------
# Public parse function
# -------------------------------

def parse_query(text: str) -> ParsedQuery:
    pq = ParsedQuery()

    if not text:
        return pq

    t = text.strip().lower()

    # --- small intents ---
    if re.search(r"\b(hi|hello|hey|yo|good\s+morning|good\s+afternoon|good\s+evening)\b", t):
        pq.is_greeting = True
    if re.search(r"\b(thanks|thank\s+you|ty|appreciate\s+it)\b", t):
        pq.is_thanks = True
    if re.search(r"\b(bye|goodbye|see\s+ya|see\s+you|later)\b", t):
        pq.is_goodbye = True

    # --- question intents ---
    if re.search(r"\b(hours?|open|close|opening|closing|when.*(open|close))\b", t):
        pq.ask_hours = True
    if re.search(r"\b(deals?|specials?|offers?|discounts?)\b", t):
        pq.ask_deals = True
    if re.search(r"(\$|\b(price|cost|how\s+much|rate|charges?)\b)", t):
        pq.ask_price = True
    if re.search(r"\b(how\s+many|in\s+stock|available|qty|quantity|left)\b", t):
        pq.ask_count = True
    if re.search(r"\b(hot|cold)\b", t) and not pq.ask_hours:  # avoid 'hot hours' false positives
        pq.ask_hotcold = True

    # --- item detection ---
    # Try best match against current public item names
    inv = _load_inventory_names()
    item = _best_item_match(text, inv)
    if item:
        pq.item = item

    return pq
