# app/schemas/orders.py
from typing import List, Optional
from pydantic import BaseModel

class OrderLineOut(BaseModel):
    itemId: str
    name: str
    qty: int
    unitPrice: float
    lineTotal: float

class AmountsOut(BaseModel):
    subtotal: float
    tax: float
    total: float
    currency: str

class CustomerOut(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None

class OrderOut(BaseModel):
    id: str
    status: str
    customer: Optional[CustomerOut] = None
    lines: List[OrderLineOut]
    amounts: AmountsOut
    createdAt: float
    updatedAt: float
