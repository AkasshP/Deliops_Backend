from __future__ import annotations
from typing import Dict, Any, List
from datetime import datetime, timezone

from firebase_admin import firestore
from .firebase import ensure_firestore

COLLECTION = "feedback"

def create_feedback(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a feedback document:
      { name?, email?, message, rating?, createdAt }
    """
    db = ensure_firestore()

    name = (payload.get("name") or "").strip() or None
    email = (payload.get("email") or "").strip() or None
    message = (payload.get("message") or "").strip()
    rating = int(payload.get("rating") or 0)

    if not message:
        raise ValueError("message is required")
    if rating < 0 or rating > 5:
        raise ValueError("rating must be between 0 and 5")

    data: Dict[str, Any] = {
        "name": name,
        "email": email,
        "message": message,
        "rating": rating,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }

    ref = db.collection(COLLECTION).document()
    ref.set(data)
    data["id"] = ref.id
    return data

def list_feedback(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Return latest feedback (newest first).
    """
    db = ensure_firestore()
    q = (
        db.collection(COLLECTION)
        .order_by("createdAt", direction=firestore.Query.DESCENDING)
        .limit(limit)
    )
    out: List[Dict[str, Any]] = []
    for snap in q.stream():
        row = snap.to_dict() or {}
        row["id"] = snap.id
        out.append(row)
    return out
