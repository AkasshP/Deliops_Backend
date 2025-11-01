# app/routes/admin.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from ..services.rag import build_index

router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/reindex")
def reindex_endpoint():
    try:
        result = build_index()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reindex failed: {e}")
