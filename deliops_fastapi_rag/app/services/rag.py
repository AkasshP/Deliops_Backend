# app/services/rag.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os
from textwrap import dedent
import numpy as np

from ..settings import settings
from .embeddings import embed_texts, embed_text
from ..vectorstore.simple_store import SimpleStore
from .items import list_items
from .nlu import parse_query, ParsedQuery

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None  # optional

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index"))

# ----- Store lifecycle -----
_store: Optional[SimpleStore] = None

def _get_store() -> SimpleStore:
    global _store
    if _store is None:
        _store = SimpleStore(INDEX_DIR)
        _store.load()  # ok if empty on first boot
    return _store

# ----- Business rules (hours & policies) -----
STORE_RULES = {
    "hours": {
        "open": "6:00 AM",
        "close": "12:00 AM",
    },
    "hot_sandwich_cutoff": "11:00 PM",
    "late_deals_start": "10:00 PM",
    "late_deals_note": "Some items go on sale after 10 PM.",
}

def _rules_answer(q: ParsedQuery) -> Optional[str]:
    if q.is_greeting and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "Hi there! How can I help—availability, prices, or late-night deals?"

    if q.is_thanks and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "You’re very welcome! If you need anything else, I’m right here."

    if q.is_goodbye and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "Thanks for stopping by—have a great day!"

    if q.ask_hours:
        return (
            f"We’re open from {STORE_RULES['hours']['open']} to {STORE_RULES['hours']['close']}. "
            f"Hot sandwiches stop at {STORE_RULES['hot_sandwich_cutoff']}; after that, only cold sandwiches are available."
        )

    if q.ask_hotcold:
        return (
            f"Hot sandwiches are served until {STORE_RULES['hot_sandwich_cutoff']}. "
            "After that time (and late at night), we offer cold sandwiches."
        )

    if q.ask_deals:
        return (
            f"Late-night deals start at {STORE_RULES['late_deals_start']}. "
            f"{STORE_RULES['late_deals_note']}"
        )
    return None

# ----- Build/refresh index from Firestore -----
def build_index() -> dict:
    items = list_items(public=True, active=True)
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("name", "")
        cat  = it.get("category") or it.get("type") or "item"
        totals = it.get("totals") or {}
        qty  = totals.get("totalQty")
        price = (it.get("price") or {}).get("current")
        desc = f"{name} | Category: {cat}"
        if isinstance(qty, int):   desc += f" | In stock: {qty}"
        if price is not None:      desc += f" | Price: ${float(price):.2f}"
        texts.append(desc)
        metas.append({
            "id": it.get("id"),
            "name": name,
            "category": cat,
            "qty": qty,
            "price": price,
            "raw": it,
        })

    vecs = embed_texts(texts) if texts else np.zeros((0, 384), np.float32)
    store = _get_store()
    store.build(vecs, metas)
    store.save()
    return {"ok": True, "count": len(metas)}

def _ensure_index_ready():
    s = _get_store()
    if s.emb is None or len(s.metas) == 0:
        build_index()

# ----- HF rewriter (optional) -----
def _hf_client() -> Optional[InferenceClient]:
    if InferenceClient is None:  # package missing
        return None
    api_key = settings.huggingface_api_key
    if not api_key:
        return None
    try:
        return InferenceClient(api_key=api_key, model=settings.hf_model_id, timeout=25)
    except Exception:
        return None

def _rewrite_with_hf(context: str, user: str, draft: str) -> Optional[str]:
    client = _hf_client()
    if client is None:
        return None

    system = dedent("""\
        You are a friendly deli assistant. You must ONLY use the facts in CONTEXT.
        Never invent items, prices, or hours. If the info is missing in CONTEXT, say you don't have it.
        Keep answers short (1–3 sentences), polite, and specific to the user's question.
    """)

    prompt = dedent(f"""\
        [SYSTEM]
        {system}

        [CONTEXT]
        {context}

        [USER QUESTION]
        {user}

        [DRAFT ANSWER]
        {draft}

        [INSTRUCTIONS]
        Improve the DRAFT ANSWER so it reads like natural, friendly English while staying 100% faithful to CONTEXT.
        If the DRAFT already looks good, keep it concise and return as-is.

        [ASSISTANT]
    """)

    try:
        text = client.text_generation(
            prompt,
            max_new_tokens=160,
            temperature=0.2,
            do_sample=False,
            return_full_text=False,
        )
        return (text or "").strip()
    except Exception:
        return None

# ----- Retrieval helpers -----
def _search_items_by_name(name: str, top_k: int) -> List[Tuple[float, Dict[str, Any]]]:
    # semantic search via embeddings + exact-ish boost
    s = _get_store()
    qv = embed_text(name)
    hits = s.search(qv, top_k=top_k)
    # light boost if exact canonical name matches
    out: List[Tuple[float, Dict[str, Any]]] = []
    for score, meta in hits:
        boost = 0.15 if meta.get("name","").lower() == name.lower() else 0.0
        out.append((score + boost, meta))
    out.sort(key=lambda x: x[0], reverse=True)
    return out

def _format_item_line(meta: Dict[str, Any]) -> str:
    name  = meta.get("name", "This item")
    qty   = meta.get("qty")
    price = meta.get("price")
    parts = [name]
    if isinstance(qty, int):
        parts.append(f"{qty} in stock" if qty > 0 else "out of stock")
    if price is not None:
        parts.append(f"${float(price):.2f} plus tax")
    return " — ".join(parts)

# ----- Main QA entry point -----
def answer_from_items(question: str, history: Optional[List[Dict[str,str]]] = None, top_k: Optional[int] = None) -> str:
    top_k = int(top_k or settings.rag_top_k)
    q = parse_query(question)

    # 1) quick rule answers (hours/greetings/deals)
    rule = _rules_answer(q)
    if rule:
        return rule

    # 2) ensure index (public, active) is ready
    _ensure_index_ready()
    store = _get_store()
    if store.emb is None or len(store.metas) == 0:
        return "I don’t see any public items right now."

    # 3) if a specific item is referenced, answer with that; else general “what’s available” list
    hits: List[Tuple[float, Dict[str, Any]]] = []
    if q.item:
        hits = _search_items_by_name(q.item, top_k=min(3, top_k))
        if not hits:
            return f"I couldn’t find “{q.item}” on the menu right now."
        top = hits[0][1]
        line = _format_item_line(top)

        # tailor to price vs count questions
        if q.ask_count and isinstance(top.get("qty"), int):
            return f"{top.get('name')} — we have {top['qty']} in stock."
        if q.ask_price and top.get("price") is not None:
            return f"{top.get('name')} is {float(top['price']):.2f} dollars plus tax."

        # default single-item answer with small suggestion set
        alts = [m["name"] for _, m in hits[1:]]
        draft = line + (f". You might also like: {', '.join(alts)}." if alts else ".")
        # build context for safe rewrite
        ctx = "\n".join([_format_item_line(m) for _, m in hits])
        better = _rewrite_with_hf(ctx, question, draft)
        return better or draft

    # 4) generic availability (“what’s available now?”)
    # sort in-stock first, then show a few top items
    metas = store.metas
    in_stock = [m for m in metas if isinstance(m.get("qty"), int) and m["qty"] > 0]
    in_stock = in_stock[: min(top_k, 6)] if in_stock else metas[: min(top_k, 6)]

    if not in_stock:
        return "Right now I don’t see any items in stock."

    lines = [_format_item_line(m) for m in in_stock]
    draft = "Here’s what’s available:\n- " + "\n- ".join(lines)
    better = _rewrite_with_hf("\n".join(lines), question, draft)
    return better or draft
