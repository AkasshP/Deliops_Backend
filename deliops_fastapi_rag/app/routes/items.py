# app/routes/items.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..services.items import (
    list_items,
    get_item,
    create_item,
    update_item,
    delete_item,
)

router = APIRouter(prefix="/items", tags=["items"])

# ---------- Models ----------
class Price(BaseModel):
    current: float = Field(..., description="Current price")

class Totals(BaseModel):
    floorQty: int = 0
    backQty: int = 0
    totalQty: int = 0

class ItemIn(BaseModel):
    id: Optional[str] = None
    name: str
    type: Optional[str] = "ingredient"       # prepared | ingredient
    service: Optional[str] = "none"          # hot | cold | none
    uom: Optional[str] = "ea"
    category: Optional[str] = None
    public: Optional[bool] = False
    active: Optional[bool] = True
    price: Optional[Price] = None
    totals: Optional[Totals] = None
    imageUrl: Optional[str] = None

class ItemOut(ItemIn):
    id: str

# PATCH-specific schema: every field optional (including nested)
class PricePatch(BaseModel):
    current: Optional[float] = None

class TotalsPatch(BaseModel):
    floorQty: Optional[int] = None
    backQty: Optional[int] = None
    totalQty: Optional[int] = None

class ItemPatch(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = None
    service: Optional[str] = None
    uom: Optional[str] = None
    category: Optional[str] = None
    public: Optional[bool] = None
    active: Optional[bool] = None
    price: Optional[PricePatch] = None
    totals: Optional[TotalsPatch] = None
    imageUrl: Optional[str] = None

# ---------- Routes ----------

@router.get("", response_model=List[ItemOut])
@router.get("/", response_model=List[ItemOut])
async def list_items_endpoint(public: Optional[bool] = Query(None),
                              active: Optional[bool] = Query(None)):
    return [ItemOut(**i) for i in await list_items(public=public, active=active)]

@router.get("/{item_id}", response_model=ItemOut)
async def get_item_endpoint(item_id: str):
    item = await get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="item not found")
    return ItemOut(**item)

# IMPORTANT: allow both "" and "/" for POST (your file had two identical "/" decorators)
@router.post("", response_model=ItemOut)
@router.post("/", response_model=ItemOut)
async def create_item_endpoint(payload: ItemIn):
    try:
        created = await create_item(payload.model_dump(exclude_unset=True))
        return ItemOut(**created)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.patch("/{item_id}", response_model=ItemOut)
async def update_item_endpoint(item_id: str, payload: ItemPatch):
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    updated = await update_item(item_id, data)
    return ItemOut(**updated)

@router.delete("/{item_id}")
async def delete_item_endpoint(item_id: str):
    await delete_item(item_id)
    return {"ok": True}
