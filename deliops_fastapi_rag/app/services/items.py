# app/services/items.py

# app/services/items.py
from __future__ import annotations
from typing import List, Dict, Any, Optional

from .firebase import ensure_firestore 

# --- READ HELPERS -------------------------------------------------------------
def list_public_items() -> List[Dict[str, Any]]:
    """
    Public + active items for the guest chat/dashboard.
    """
    db = ensure_firestore()
    col = (
        db.collection("items")
        .where("public", "==", True)
        .where("active", "==", True)
    )
    docs = col.stream()
    out: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        data["id"] = d.id
        out.append(data)
    return out


def list_items(public: Optional[bool] = None,
               active: Optional[bool] = None) -> List[Dict[str, Any]]:
    """
    Admin list: optional filters for public/active.
    """
    db = ensure_firestore()
    col = db.collection("items")
    if public is not None:
        col = col.where("public", "==", bool(public))
    if active is not None:
        col = col.where("active", "==", bool(active))

    docs = col.stream()
    out: List[Dict[str, Any]] = []
    for d in docs:
        data = d.to_dict() or {}
        data["id"] = d.id
        out.append(data)
    return out


def get_item(item_id: str) -> Optional[Dict[str, Any]]:
    if not item_id:
        return None
    db = ensure_firestore()
    snap = db.collection("items").document(item_id).get()
    if not snap.exists:
        return None
    data = snap.to_dict() or {}
    data["id"] = snap.id
    return data


def create_item(payload: Dict[str, Any]) -> Dict[str, Any]:
    db = ensure_firestore()
    name: str = (payload.get("name") or "").strip()
    if not name:
        raise ValueError("name is required")

    # use local _slug helper
    item_id = payload.get("id") or _slug(name)

    ref = db.collection("items").document(item_id)

    base = {
        "name": name,
        "type": payload.get("type", "ingredient"),
        "service": payload.get("service", "none"),
        "uom": payload.get("uom", "ea"),
        "category": payload.get("category"),
        "public": bool(payload.get("public", False)),
        "active": bool(payload.get("active", True)),
    }
    if "price" in payload:
        base["price"] = payload["price"]
    if "totals" in payload:
        base["totals"] = payload["totals"]

    ref.set(base, merge=True)
    out = base.copy()
    out["id"] = item_id
    return out


def update_item(item_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if not item_id:
        raise ValueError("item_id required")
    db = ensure_firestore()
    ref = db.collection("items").document(item_id)
    ref.set(payload, merge=True)
    snap = ref.get()
    data = snap.to_dict() or {}
    data["id"] = item_id
    return data


def delete_item(item_id: str) -> None:
    if not item_id:
        raise ValueError("item_id required")
    db = ensure_firestore()
    db.collection("items").document(item_id).delete()


# --- UTILS --------------------------------------------------------------------
def _slug(s: str) -> str:
    return (
        s.strip()
        .lower()
        .replace("&", " and ")
        .replace("/", " ")
        .replace("_", " ")
        .encode("ascii", "ignore").decode("ascii")
        .replace("'", "")
        .replace(".", " ")
        .replace(",", " ")
        .replace("  ", " ")
        .strip()
        .replace(" ", "-")
    )