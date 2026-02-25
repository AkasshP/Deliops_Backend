from __future__ import annotations
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.feedback import create_feedback, list_feedback

router = APIRouter(prefix="/feedback", tags=["feedback"])

class FeedbackIn(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    message: str = Field(..., min_length=3, max_length=2000)
    rating: Optional[int] = Field(0, ge=0, le=5)

class FeedbackOut(FeedbackIn):
    id: str
    createdAt: Optional[str] = None

@router.post("", response_model=FeedbackOut)
async def feedback_create(body: FeedbackIn):
    try:
        return await create_feedback(body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[FeedbackOut])
async def feedback_list():
    return await list_feedback()
