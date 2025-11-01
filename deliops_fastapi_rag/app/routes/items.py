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


# ---- Pydantic models ---------------------------------------------------------
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


class ItemOut(ItemIn):
    id: str


# ---- Helpers -----------------------------------------------------------------
def _coerce_out(items: List[Dict[str, Any]]) -> List[ItemOut]:
    return [ItemOut(**{**i, "id": i["id"]}) for i in items if "id" in i]


# ---- Routes ------------------------------------------------------------------
# GET /items   and   GET /items/
@router.get("", response_model=List[ItemOut])
def list_items_endpoint(
    public: Optional[bool] = Query(None),
    active: Optional[bool] = Query(None),
):
    return _coerce_out(list_items(public=public, active=active))

# GET /items/{id}
@router.get("/{item_id}", response_model=ItemOut)
def get_item_endpoint(item_id: str):
    item = get_item(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="item not found")
    return ItemOut(**item)

# POST /items    and   POST /items/
@router.post("/", response_model=ItemOut)
@router.post("/", response_model=ItemOut)
def create_item_endpoint(payload: ItemIn):
    try:
        created = create_item(payload.model_dump(exclude_unset=True))
        return ItemOut(**created)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# PATCH /items/{id}
@router.patch("/{item_id}", response_model=ItemOut)
def update_item_endpoint(item_id: str, payload: ItemIn):
    updated = update_item(item_id, payload.model_dump(exclude_unset=True))
    return ItemOut(**updated)

# DELETE /items/{id}
@router.delete("/{item_id}")
def delete_item_endpoint(item_id: str):
    delete_item(item_id)
    return {"ok": True}
