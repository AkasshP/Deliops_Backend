# app/services/rag.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
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

# Where the vector index lives on disk
INDEX_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vectorstore", "index")
)

# ---------- Globals ----------
_store: Optional[SimpleStore] = None
_name_map: Dict[str, Dict[str, Any]] = {}  # canonical-name -> meta


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
    return None


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
                "raw": it,
            }
        )

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


# ---------- Optional HF polish ----------
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
    """Ask the configured HF instruct model to lightly polish the draft."""
    client = _hf_client()
    if client is None:
        return None

    system = dedent(
        """\
        You are a friendly deli assistant. Use only facts from CONTEXT.
        Never invent items, prices, or hours. If something is missing, say so.
        Keep answers short (one to three sentences), clear, and polite.
        Avoid em dashes and semicolons.
        """
    )
    prompt = dedent(
        f"""\
        [SYSTEM]
        {system}

        [CONTEXT]
        {context}

        [USER]
        {user}

        [DRAFT]
        {draft}

        [ASSISTANT]
        """
    )
    try:
        out = client.text_generation(
            prompt, max_new_tokens=160, temperature=0.2, do_sample=False, return_full_text=False
        )
        return (out or "").strip()
    except Exception:
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


# ---------- Main QA ----------
def answer_from_items(
    question: str, history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None
) -> str:
    ensure_index_ready()
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
        if q.ask_count and isinstance(meta.get("qty"), int):
            return (
                f"{meta['name']} is available. We have {meta['qty']} in stock."
                if meta["qty"] > 0
                else f"{meta['name']} is currently sold out."
            )
        if q.ask_price and meta.get("price") is not None:
            return f"{meta['name']} costs ${float(meta['price']):.2f} plus tax."

        qv = embed_text(meta["name"])
        hits = s.search(qv, top_k=3)
        alts = [m["name"] for _, m in hits if m is not meta][:2]

        draft = _format_item_sentence(meta)
        if alts:
            draft += f" You might also like: {', '.join(alts)}."

        ctx = "\n".join([_format_item_sentence(m) for _, m in hits])
        better = _rewrite_with_hf(ctx, question, draft)
        return better or draft

    # 2) semantic search on the full question
    vec = embed_text(question)
    hits = s.search(vec, top_k=int(top_k or settings.rag_top_k))
    if hits:
        top = hits[0][1]

        if q.ask_count and isinstance(top.get("qty"), int):
            return (
                f"{top['name']} is available. We have {top['qty']} in stock."
                if top["qty"] > 0
                else f"{top['name']} is currently sold out."
            )
        if q.ask_price and top.get("price") is not None:
            return f"{top['name']} costs ${float(top['price']):.2f} plus tax."

        alts = [m["name"] for _, m in hits[1:3]]
        draft = _format_item_sentence(top)
        if alts:
            draft += f" You might also like: {', '.join(alts)}."

        ctx = "\n".join([_format_item_sentence(m) for _, m in hits])
        better = _rewrite_with_hf(ctx, question, draft)
        return better or draft

    # 3) generic availability fallback (short, friendly list)
    metas = s.metas
    in_stock = [m for m in metas if isinstance(m.get("qty"), int) and m["qty"] > 0]
    show = in_stock[:6] if in_stock else metas[:6]
    if not show:
        return "Right now I do not see any items in stock."

    lines = [_format_item_sentence(m) for m in show]
    draft = "Here is what I can serve right now:\n- " + "\n- ".join(lines)
    better = _rewrite_with_hf("\n".join(lines), question, draft)
    return better or draft
