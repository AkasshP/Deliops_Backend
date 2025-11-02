from __future__ import annotations
from fastapi import APIRouter

from ..services.rag import INDEX_DIR, ensure_index_ready, build_index
from ..services.rag import _get_store  # local import for status details

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/index-status")
def index_status():
    """
    Helpful for debugging: shows whether the vector index is loaded and how many items it has.
    (No auth here; add your auth dependency if you want this locked down.)
    """
    ensure_index_ready()
    s = _get_store()
    ready = (s.emb is not None) and (len(s.metas) > 0)
    return {"ready": ready, "count": len(s.metas), "dir": str(INDEX_DIR)}

@router.post("/reindex")
def reindex():
    """
    Manually rebuild the index from Firestore. Use from your admin UI.
    """
    result = build_index()
    return {"ok": True, **result}
