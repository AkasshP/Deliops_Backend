# app/services/rag.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os
from textwrap import dedent
import numpy as np
import re

from ..settings import settings
from .embeddings import embed_texts, embed_text
from ..vectorstore.simple_store import SimpleStore
from .items import list_items
from .nlu import parse_query, ParsedQuery

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index"))

# ---------- Store & name map ----------
_store: Optional[SimpleStore] = None
_name_map: Dict[str, Dict[str, Any]] = {}     # canonical-name -> meta

def _get_store() -> SimpleStore:
    global _store
    if _store is None:
        _store = SimpleStore(INDEX_DIR)
        _store.load()  # ok if empty; we’ll build when needed
    return _store

def _rebuild_name_map(metas: List[Dict[str, Any]]) -> None:
    global _name_map
    _name_map = {}
    for m in metas:
        nm = (m.get("name") or "").strip().lower()
        if nm:
            _name_map[nm] = m

# ---------- Rules ----------
STORE_RULES = {
    "hours": {"open": "6:00 AM", "close": "12:00 AM"},
    "hot_sandwich_cutoff": "11:00 PM",
    "late_deals_start": "10:00 PM",
    "late_deals_note": "Some items go on sale after 10 PM.",
}

def _rules_answer(q: ParsedQuery) -> Optional[str]:
    # Be lenient with greetings
    if q.is_greeting:
        return "Hi there! How can I help—availability, prices, or late-night deals?"

    if q.is_thanks and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "You’re very welcome! If you need anything else, I’m right here."

    if q.is_goodbye and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "Thanks for stopping by—have a great day!"

    if q.ask_hours:
        return (f"We’re open from {STORE_RULES['hours']['open']} to {STORE_RULES['hours']['close']}. "
                f"Hot sandwiches stop at {STORE_RULES['hot_sandwich_cutoff']}; after that, only cold sandwiches are available.")
    if q.ask_hotcold:
        return (f"Hot sandwiches are served until {STORE_RULES['hot_sandwich_cutoff']}. "
                "After that time, we offer cold sandwiches.")
    if q.ask_deals:
        return (f"Late-night deals start at {STORE_RULES['late_deals_start']}. "
                f"{STORE_RULES['late_deals_note']}")
    return None

# ---------- Build/refresh index ----------
def build_index() -> dict:
    # IMPORTANT: include *everything* so ingredients like “turkey” are searchable
    items = list_items(public=None, active=None)

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("name", "")
        typ  = it.get("type") or "item"        # prepared / ingredient / packaged...
        svc  = it.get("service") or "none"     # hot / cold / drink / none
        totals = it.get("totals") or {}
        qty  = totals.get("totalQty")
        price = (it.get("price") or {}).get("current")

        desc = f"{name} | Type: {typ} | Service: {svc}"
        if isinstance(qty, int):   desc += f" | In stock: {qty}"
        if price is not None:      desc += f" | Price: ${float(price):.2f}"

        texts.append(desc)
        metas.append({
            "id": it.get("id"),
            "name": name,
            "type": typ,
            "service": svc,
            "qty": qty,
            "price": price,
            "raw": it,
        })

    vecs = embed_texts(texts) if texts else np.zeros((0, 384), np.float32)
    store = _get_store()
    store.build(vecs, metas)
    store.save()
    _rebuild_name_map(metas)
    return {"ok": True, "count": len(metas)}

def ensure_index_ready() -> None:
    s = _get_store()
    if s.emb is None or len(s.metas) == 0:
        build_index()
    elif not _name_map:
        _rebuild_name_map(s.metas)

# ---------- HF polish (optional) ----------
def _hf_client() -> Optional[InferenceClient]:
    if InferenceClient is None:
        return None
    ak = settings.huggingface_api_key
    if not ak:
        return None
    try:
        return InferenceClient(api_key=ak, model=settings.hf_model_id, timeout=25)
    except Exception:
        return None

def _rewrite_with_hf(context: str, user: str, draft: str) -> Optional[str]:
    client = _hf_client()
    if client is None:
        return None

    system = dedent("""\
        You are a friendly deli assistant. ONLY use facts found in CONTEXT.
        Never invent items, prices, or hours. If missing, say you don't have it.
        Keep answers short (1–3 sentences), polite, and specific.
    """)
    prompt = dedent(f"""\
        [SYSTEM]
        {system}

        [CONTEXT]
        {context}

        [USER]
        {user}

        [DRAFT]
        {draft}

        [ASSISTANT]
    """)
    try:
        out = client.text_generation(prompt, max_new_tokens=160, temperature=0.2,
                                     do_sample=False, return_full_text=False)
        return (out or "").strip()
    except Exception:
        return None

# ---------- Helpers ----------
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

def _exact_or_contains_lookup(user_text: str) -> Optional[Dict[str, Any]]:
    """
    First try exact-name match; then a conservative 'contains' match on word boundaries.
    """
    if not _name_map:
        return None
    q = user_text.strip().lower()

    # exact
    if q in _name_map:
        return _name_map[q]

    # word-boundary contains (e.g., "do you have turkey?" -> "turkey")
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-\& ]+", q)
    cand = " ".join(tokens).strip()
    for nm, meta in _name_map.items():
        if re.search(rf"\b{re.escape(nm)}\b", cand):
            return meta
    return None

# ---------- Main QA ----------
def answer_from_items(question: str, history: Optional[List[Dict[str,str]]] = None,
                      top_k: Optional[int] = None) -> str:
    ensure_index_ready()
    s = _get_store()
    if s.emb is None or len(s.metas) == 0:
        return "I don’t see any items yet."

    # 0) quick rules/greetings
    q = parse_query(question)
    rule = _rules_answer(q)
    if rule:
        return rule

    # 1) very fast exact lookup by name (handles “turkey?”, “beef?”, etc.)
    meta = _exact_or_contains_lookup(question)
    if meta:
        if q.ask_count and isinstance(meta.get("qty"), int):
            return f"{meta['name']} — we have {meta['qty']} in stock."
        if q.ask_price and meta.get("price") is not None:
            return f"{meta['name']} is {float(meta['price']):.2f} dollars plus tax."
        # default single-item answer, with a couple of similar suggestions via semantic
        qv = embed_text(meta["name"])
        hits = s.search(qv, top_k=3)
        alts = [m["name"] for _, m in hits if m is not meta][:2]
        draft = _format_item_line(meta) + (f". You might also like: {', '.join(alts)}." if alts else ".")
        ctx = "\n".join([_format_item_line(m) for _, m in hits])
        better = _rewrite_with_hf(ctx, question, draft)
        return better or draft

    # 2) semantic search by the full question
    vec = embed_text(question)
    hits = s.search(vec, top_k=int(top_k or settings.rag_top_k))
    if hits:
        top = hits[0][1]
        # tailor
        if q.ask_count and isinstance(top.get("qty"), int):
            return f"{top['name']} — we have {top['qty']} in stock."
        if q.ask_price and top.get("price") is not None:
            return f"{top['name']} is {float(top['price']):.2f} dollars plus tax."
        # default
        alts = [m["name"] for _, m in hits[1:3]]
        draft = _format_item_line(top) + (f". You might also like: {', '.join(alts)}." if alts else ".")
        ctx = "\n".join([_format_item_line(m) for _, m in hits])
        better = _rewrite_with_hf(ctx, question, draft)
        return better or draft

    # 3) generic availability fallback
    metas = s.metas
    in_stock = [m for m in metas if isinstance(m.get("qty"), int) and m["qty"] > 0]
    show = in_stock[:6] if in_stock else metas[:6]
    if not show:
        return "Right now I don’t see any items in stock."
    lines = [_format_item_line(m) for m in show]
    draft = "Here’s what’s available:\n- " + "\n- ".join(lines)
    better = _rewrite_with_hf("\n".join(lines), question, draft)
    return better or draft
