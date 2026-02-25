# app/services/rag.py
from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import re
import traceback
from textwrap import dedent

from openai import OpenAI

from ..settings import settings
from .embeddings import embed_texts, embed_text
from .items import list_items, get_item
from .nlu import parse_query, ParsedQuery
from ..db import pgvector_store

# ---------- Globals ----------
_name_map: Dict[str, Dict[str, Any]] = {}  # canonical-name -> meta
_llm_client: Optional[OpenAI] = None


def _rebuild_name_map(metas: List[Dict[str, Any]]) -> None:
    global _name_map
    _name_map = {}
    for m in metas:
        nm = (m.get("name") or "").strip().lower()
        if nm:
            _name_map[nm] = m


def _get_llm_client() -> Optional[OpenAI]:
    global _llm_client
    if _llm_client is not None:
        return _llm_client
    if not settings.openrouter_api_key:
        return None
    _llm_client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    return _llm_client


# ---------- Store rules / direct answers ----------
STORE_RULES = {
    "hours": {"open": "6:00 AM", "close": "12:00 AM"},
    "hot_sandwich_cutoff": "11:00 PM",
    "late_deals_start": "10:00 PM",
    "late_deals_note": "Some items go on sale after 10 PM.",
}


def _rules_answer(q: ParsedQuery) -> Optional[str]:
    if q.is_greeting:
        return "Hi there! How can I help you today? I can check availability, prices, or late-night deals."
    if q.is_thanks and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "You are very welcome. If you need anything else, I am right here."
    if q.is_goodbye and not (q.ask_hours or q.ask_deals or q.ask_price or q.ask_count or q.item):
        return "Thanks for stopping by. Have a great day."
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
    """Recursively convert non-JSON-serializable types to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, 'isoformat'):
        return obj.isoformat()
    if hasattr(obj, '__dict__'):
        return str(obj)
    return str(obj)


# ---------- Build / refresh index (Postgres pgvector) ----------
async def build_index() -> dict:
    """Build the pgvector index from the full item list (Postgres items → pgvector embeddings)."""
    items = await list_items(public=None, active=None)

    texts: List[str] = []
    rows: List[Dict[str, Any]] = []

    for it in items:
        name = it.get("name", "")
        typ = it.get("type") or "item"
        svc = it.get("service") or "none"
        category = it.get("category") or typ
        totals = it.get("totals") or {}
        qty = totals.get("totalQty")
        price = (it.get("price") or {}).get("current")
        in_stock = isinstance(qty, int) and qty > 0

        desc = f"{name} | Type: {typ} | Service: {svc}"
        if isinstance(qty, int):
            desc += f" | In stock: {qty}"
        if price is not None:
            desc += f" | Price: ${float(price):.2f}"

        texts.append(desc)
        rows.append({
            "id": it.get("id"),
            "name": name,
            "category": category,
            "description": desc,
            "price": float(price) if price is not None else None,
            "in_stock": in_stock,
            # Keep full meta for name_map
            "type": typ,
            "service": svc,
            "qty": qty,
            "raw": _sanitize_for_json(it),
        })

    # Generate embeddings
    if texts:
        vecs = embed_texts(texts)
        embeddings = [v.tolist() for v in vecs]
    else:
        embeddings = []

    # Upsert into Postgres pgvector
    count = await pgvector_store.upsert_items(rows, embeddings)

    # Clean up embeddings for items no longer active
    active_ids = {r["id"] for r in rows if r["id"]}
    await pgvector_store.delete_missing(active_ids)

    # Rebuild in-memory name map for fast-path lookups
    _rebuild_name_map(rows)

    return {"ok": True, "count": count}


async def ensure_index_ready(startup: bool = False) -> None:
    """
    Ensure the name_map is populated for fast-path lookups.
    On startup, attempt to build the full index.
    On lazy calls, just populate the name_map from Postgres items.
    """
    global _name_map

    if _name_map:
        return

    try:
        if startup:
            result = await build_index()
            print(f"[startup] pgvector index built with {result.get('count', 0)} items")
        else:
            # Just populate the name_map from Postgres for fast-path lookups
            items = await list_items(public=None, active=None)
            metas = []
            for it in items:
                totals = it.get("totals") or {}
                price = (it.get("price") or {}).get("current")
                metas.append({
                    "id": it.get("id"),
                    "name": it.get("name", ""),
                    "type": it.get("type") or "item",
                    "service": it.get("service") or "none",
                    "qty": totals.get("totalQty"),
                    "price": price,
                })
            _rebuild_name_map(metas)
    except Exception:
        traceback.print_exc()
        if startup:
            print("[startup] WARNING: Could not build pgvector index. "
                  "The app will start but vector search may fail.")
        else:
            raise


# ---------- LLM "polish" using OpenAI ----------
def _rewrite_with_llm(context: str, user: str, draft: str) -> Optional[str]:
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
        {"role": "system", "content": system},
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
            model=settings.openrouter_model,
            messages=messages,
            temperature=0.3,
            max_tokens=200,
        )
        out = resp.choices[0].message.content or ""
        return out.strip()
    except Exception:
        return None


# ---------- Helpers: natural sentences ----------
def _format_item_sentence(meta: Dict[str, Any]) -> str:
    name = meta.get("name") or meta.get("item_name", "This item")
    qty = meta.get("qty")
    price = meta.get("price")

    parts: List[str] = []
    if isinstance(qty, int):
        if qty > 0:
            parts.append(f"{name} is available.")
            parts.append(f"We have {qty} in stock.")
        else:
            parts.append(f"{name} is currently sold out.")
    elif meta.get("in_stock") is not None:
        if meta["in_stock"]:
            parts.append(f"{name} is available.")
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
    """
    if not _name_map:
        return None
    q = user_text.strip().lower()

    if q in _name_map:
        return _name_map[q]

    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-\& ]+", q)
    cand = " ".join(tokens).strip()
    for nm, meta in _name_map.items():
        if re.search(rf"\b{re.escape(nm)}\b", cand):
            return meta
    return None


async def _get_fresh_item_data(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch fresh item data from Postgres for real-time qty and price."""
    item_id = meta.get("id")
    if not item_id:
        return meta
    try:
        fresh = await get_item(item_id)
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


async def _format_item_response(meta: Dict[str, Any], include_price: bool = True, include_qty: bool = True) -> str:
    fresh_meta = await _get_fresh_item_data(meta)
    name = fresh_meta.get("name", "This item")
    qty = fresh_meta.get("qty")
    price = fresh_meta.get("price")

    parts: List[str] = []
    if include_qty and isinstance(qty, int):
        if qty > 0:
            parts.append(f"{name} is available with {qty} in stock.")
        else:
            parts.append(f"{name} is currently out of stock.")
    else:
        parts.append(f"{name} is available.")

    if include_price and price is not None:
        parts.append(f"It costs ${float(price):.2f} plus tax.")

    return " ".join(parts)


# ---------- Main QA ----------
async def answer_from_items(
    question: str, history: Optional[List[Dict[str, str]]] = None, top_k: Optional[int] = None
) -> str:
    await ensure_index_ready()

    # 0) quick rules and small talk
    q = parse_query(question)
    rule = _rules_answer(q)
    if rule:
        return rule

    # 1) exact / contains name lookup first (fast and reliable)
    meta = _exact_or_contains_lookup(question)
    if meta:
        response = await _format_item_response(meta)
        return response

    # 2) semantic search via pgvector
    vec = embed_text(question)
    k = int(top_k or settings.rag_top_k)
    hits = await pgvector_store.query(vec.tolist(), top_k=k)

    if hits:
        top = hits[0]
        # Map pgvector result to the format _format_item_response expects
        meta_from_pg = {
            "id": top["item_id"],
            "name": top["item_name"],
            "price": top["price"],
            "in_stock": top["in_stock"],
        }
        response = await _format_item_response(meta_from_pg)
        return response

    # 3) generic availability fallback
    metas = list(_name_map.values())
    in_stock = [m for m in metas if isinstance(m.get("qty"), int) and m["qty"] > 0]
    show = in_stock[:6] if in_stock else metas[:6]
    if not show:
        return "Right now I do not see any items in stock."

    lines = [_format_item_sentence(m) for m in show]
    draft = "Here is what I can serve right now:\n- " + "\n- ".join(lines)
    better = _rewrite_with_llm("\n".join(lines), question, draft)
    return better or draft


async def extract_order_lines_with_gpt(
    user_text: str,
    known_items: list[dict[str, Any]],
    conversation_context: str = "",
) -> list[dict[str, Any]]:
    """Use GPT to turn free-text into a list of {name, qty} lines."""
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
            model=settings.openrouter_model,
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
