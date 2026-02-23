# app/services/rag.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import os
import json
import traceback
from textwrap import dedent
import numpy as np
import re

from openai import OpenAI

from ..settings import settings
from .embeddings import embed_texts, embed_text
from ..vectorstore.simple_store import SimpleStore
from .items import list_items, get_item
from .nlu import parse_query, ParsedQuery

# Where the vector index lives on disk
INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index")
)

# ---------- Globals ----------
_store: Optional[SimpleStore] = None
_name_map: Dict[str, Dict[str, Any]] = {}  # canonical-name -> meta
_llm_client: Optional[OpenAI] = None


def _get_store() -> SimpleStore:
    global _store
    if _store is None:
        _store = SimpleStore(INDEX_DIR)
        _store.load()  # fine if empty; we will build on demand
    return _store


def _rebuild_name_map(metas: List[Dict[str, Any]]) -> None:
    global _name_map
    _name_map = {}
    for m in metas:
        nm = (m.get("name") or "").strip().lower()
        if nm:
            _name_map[nm] = m


def _get_llm_client() -> Optional[OpenAI]:
    """
    Lazily construct an OpenAI client, or return None if not configured.
    """
    global _llm_client
    if _llm_client is not None:
        return _llm_client

    if not settings.openai_api_key:
        # LLM polish is optional; if no key, we just skip it.
        return None

    _llm_client = OpenAI(api_key=settings.openai_api_key)
    return _llm_client


# ---------- Store rules / direct answers ----------
STORE_RULES = {
    "hours": {"open": "6:00 AM", "close": "12:00 AM"},
    "hot_sandwich_cutoff": "11:00 PM",
    "late_deals_start": "10:00 PM",
    "late_deals_note": "Some items go on sale after 10 PM.",
}


def _rules_answer(q: ParsedQuery) -> Optional[str]:
    # Friendly small talk
    if q.is_greeting:
        return "Hi there! How can I help you today? I can check availability, prices, or late-night deals."

    if q.is_thanks and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "You are very welcome. If you need anything else, I am right here."

    if q.is_goodbye and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "Thanks for stopping by. Have a great day."

    # House rules
    if q.ask_hours:
        return (
            f"We are open from {STORE_RULES['hours']['open']} to {STORE_RULES['hours']['close']}. "
            f"Hot sandwiches are served until {STORE_RULES['hot_sandwich_cutoff']}. After that, only cold sandwiches are available."
        )
    if q.ask_hotcold:
        return (
            f"Hot sandwiches are served until {STORE_RULES['hot_sandwich_cutoff']}. "
            "After that time we offer cold sandwiches."
        )
    if q.ask_deals:
        return (
            f"Late-night deals start at {STORE_RULES['late_deals_start']}. "
            f"{STORE_RULES['late_deals_note']}"
        )
    if q.ask_payment:
        return "We accept cash and all major credit/debit cards including Visa, Mastercard, and American Express."
    return None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert Firestore-specific types to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    # Handle datetime-like objects (Firestore DatetimeWithNanoseconds, etc.)
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    # Handle GeoPoint and other Firestore types
    if hasattr(obj, '__dict__'):
        return str(obj)
    return str(obj)


# ---------- Build / refresh index ----------
def build_index() -> dict:
    """Build the vector index from the full item list (prepared + ingredients)."""
    items = list_items(public=None, active=None)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("name", "")
        typ = it.get("type") or "item"         # prepared / ingredient / packaged ...
        svc = it.get("service") or "none"      # hot / cold / drink / none
        totals = it.get("totals") or {}
        qty = totals.get("totalQty")
        price = (it.get("price") or {}).get("current")

        desc = f"{name} | Type: {typ} | Service: {svc}"
        if isinstance(qty, int):
            desc += f" | In stock: {qty}"
        if price is not None:
            desc += f" | Price: ${float(price):.2f}"

        texts.append(desc)
        metas.append(
            {
                "id": it.get("id"),
                "name": name,
                "type": typ,
                "service": svc,
                "qty": qty,
                "price": price,
                "raw": _sanitize_for_json(it),
            }
        )

    vecs = embed_texts(texts) if texts else np.zeros((0, 384), np.float32)
    store = _get_store()
    store.build(vecs, metas)
    store.save()
    _rebuild_name_map(metas)
    return {"ok": True, "count": len(metas)}


def ensure_index_ready(startup: bool = False) -> SimpleStore:
    """
    Ensure the vector index is loaded into memory.
    - If it already exists: load it.
    - If it's missing: build a fresh one.
    - If something unexpected happens:
        * during startup: re-raise (fail fast)
        * during lazy calls: log and re-raise.
    """
    global _store
    if _store is not None:
        return _store

    try:
        store = _get_store()
        # Check if store has data loaded
        if store.emb is not None and len(store.metas) > 0:
            print(f"[embeddings] Loaded existing index from {INDEX_DIR}")
            _rebuild_name_map(store.metas)
            return store

        # No existing index or empty, build a new one
        print(f"[embeddings] No index found or empty at {INDEX_DIR}, building new one...")
        os.makedirs(INDEX_DIR, exist_ok=True)
        result = build_index()
        print(f"[embeddings] Built index with {result.get('count', 0)} items")
        return _get_store()

    except Exception:
        traceback.print_exc()
        if startup:
            print("[startup] WARNING: Could not build vector index (Firebase may be unavailable). "
                  "The app will start but /chat and /items may fail until Firebase is reachable.")
            return None
        raise

# ---------- LLM “polish” using OpenAI ----------
def _rewrite_with_llm(context: str, user: str, draft: str) -> Optional[str]:
    """
    Ask the configured OpenAI chat model to lightly polish the draft.
    Keeps it grounded in CONTEXT and store rules.
    """
    client = _get_llm_client()
    if client is None:
        return None

    system = dedent(
        """\
        You are a friendly deli assistant for Huskies Deli.
        Use only facts from CONTEXT and from the store rules in the prompt.
        Never invent items, prices, or hours. If something is missing, say so.
        Keep answers short (one to three sentences), clear, and polite.
        Avoid em dashes and semicolons.
        """
    )

    messages = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": dedent(
                f"""\
                CONTEXT:
                {context}

                USER QUESTION:
                {user}

                DRAFT ANSWER:
                {draft}

                Please rewrite the DRAFT ANSWER so it is natural, clear English,
                grounded only in the CONTEXT. If draft is already good, keep it almost the same.
                """
            ),
        },
    ]

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        out = resp.choices[0].message.content or ""
        return out.strip()
    except Exception:
        # If anything goes wrong, just fall back to the original draft.
        return None


# ---------- Helpers: natural sentences ----------
def _format_item_sentence(meta: Dict[str, Any]) -> str:
    """
    Return a clean, friendly sentence about a single item.
    Examples:
      - "Beef is available. We have 4 in stock."
      - "Pork is currently sold out."
      - "Mac & Cheese is available. We have 5 in stock. It costs $5.99 plus tax."
    """
    name = meta.get("name", "This item")
    qty = meta.get("qty")
    price = meta.get("price")

    parts: List[str] = []

    if isinstance(qty, int):
        if qty > 0:
            parts.append(f"{name} is available.")
            parts.append(f"We have {qty} in stock.")
        else:
            parts.append(f"{name} is currently sold out.")
    else:
        parts.append(f"{name}.")

    if price is not None:
        parts.append(f"It costs ${float(price):.2f} plus tax.")

    return " ".join(parts)


def _exact_or_contains_lookup(user_text: str) -> Optional[Dict[str, Any]]:
    """
    Try exact-name match first, then a conservative 'contains' match on word boundaries.
    Handles short queries like "turkey?" or "beef".
    """
    if not _name_map:
        return None
    q = user_text.strip().lower()

    # exact
    if q in _name_map:
        return _name_map[q]

    # word-boundary contains
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-\& ]+", q)
    cand = " ".join(tokens).strip()
    for nm, meta in _name_map.items():
        if re.search(rf"\b{re.escape(nm)}\b", cand):
            return meta
    return None


def _get_fresh_item_data(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch fresh item data from Firestore to get real-time qty and price.
    Returns updated meta dict with current values.
    """
    item_id = meta.get("id")
    if not item_id:
        return meta

    try:
        fresh = get_item(item_id)
        if fresh:
            totals = fresh.get("totals") or {}
            price_obj = fresh.get("price") or {}
            return {
                **meta,
                "qty": totals.get("totalQty"),
                "price": price_obj.get("current"),
                "raw": fresh,
            }
    except Exception:
        pass

    return meta


def _format_item_response(meta: Dict[str, Any], include_price: bool = True, include_qty: bool = True) -> str:
    """
    Format a response for an item with fresh data from Firestore.
    """
    # Get fresh data
    fresh_meta = _get_fresh_item_data(meta)
    name = fresh_meta.get("name", "This item")
    qty = fresh_meta.get("qty")
    price = fresh_meta.get("price")

    parts: List[str] = []

    # Stock info
    if include_qty and isinstance(qty, int):
        if qty > 0:
            parts.append(f"{name} is available with {qty} in stock.")
        else:
            parts.append(f"{name} is currently out of stock.")
    else:
        parts.append(f"{name} is available.")

    # Price info
    if include_price and price is not None:
        parts.append(f"It costs ${float(price):.2f} plus tax.")

    return " ".join(parts)


# ---------- Main QA ----------
def answer_from_items(
    question: str, history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None
) -> str:
    ensure_index_ready(startup=False)
    s = _get_store()
    if s.emb is None or len(s.metas) == 0:
        return "I do not see any items yet."

    # 0) quick rules and small talk
    q = parse_query(question)
    rule = _rules_answer(q)
    if rule:
        return rule

    # 1) exact / contains name lookup first (fast and reliable)
    meta = _exact_or_contains_lookup(question)
    if meta:
        # Always get fresh data from Firestore for accurate stock/price
        response = _format_item_response(meta)
        return response

    # 2) semantic search on the full question
    vec = embed_text(question)
    hits = s.search(vec, top_k=int(top_k or settings.rag_top_k))
    if hits:
        top = hits[0][1]
        # Always get fresh data from Firestore for accurate stock/price
        response = _format_item_response(top)
        return response

    # 3) generic availability fallback (short, friendly list)
    metas = s.metas
    in_stock = [m for m in metas if isinstance(m.get("qty"), int) and m["qty"] > 0]
    show = in_stock[:6] if in_stock else metas[:6]
    if not show:
        return "Right now I do not see any items in stock."

    lines = [_format_item_sentence(m) for m in show]
    draft = "Here is what I can serve right now:\n- " + "\n- ".join(lines)
    better = _rewrite_with_llm("\n".join(lines), question, draft)
    return better or draft

def extract_order_lines_with_gpt(
    user_text: str,
    known_items: list[dict[str, Any]],
    conversation_context: str = "",
) -> list[dict[str, Any]]:
    """
    Use GPT-3.5 to turn free-text into a list of {name, qty} lines.
    Returns [] if it can't confidently detect an order.
    Uses conversation context to understand references like "i want 3" after discussing an item.
    """
    # keep only item names so the model snaps to your real menu
    menu_names = [it.get("name", "") for it in known_items if it.get("name")]

    system = (
        "You are an ordering assistant for a deli.\n"
        "User text may be messy (typos, extra words).\n"
        "Your job is ONLY to extract what they are trying to order.\n"
        "Use the MENU list to match item names. If nothing is being ordered, "
        "return an empty list.\n"
        "Use the CONVERSATION HISTORY to understand context (e.g., if user says 'i want 3' "
        "after asking about 'honey chicken', they mean 3 honey chicken).\n"
        "Always respond with pure JSON: {\"lines\":[{\"name\":...,\"qty\":...}, ...]}."
    )

    menu_str = ", ".join(menu_names)

    # Include conversation context if available
    context_section = ""
    if conversation_context:
        context_section = f"CONVERSATION HISTORY:\n{conversation_context}\n\n"

    prompt = (
        f"MENU ITEMS: {menu_str}\n\n"
        f"{context_section}"
        f"CURRENT USER MESSAGE: {user_text}\n\n"
        "Return JSON only."
    )

    client = _get_llm_client()
    if client is None:
        return []

    try:
        resp = client.chat.completions.create(
            model=settings.openai_model or "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception:
        return []

    try:
        data = json.loads(raw)
    except Exception:
        return []

    lines = data.get("lines") or []
    out: list[dict[str, Any]] = []
    for ln in lines:
        name = (ln.get("name") or "").strip()
        try:
            qty = int(ln.get("qty") or 0)
        except Exception:
            qty = 0
        if name and qty > 0:
            out.append({"name": name, "qty": qty})
    return out
