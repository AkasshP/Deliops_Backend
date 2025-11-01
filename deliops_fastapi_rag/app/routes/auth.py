from __future__ import annotations
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

# Env-based credentials (simple, no JWT to keep it minimal)
ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin123")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "ok-admin")  # what frontend stores

class LoginBody(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(body: LoginBody):
    if body.username == ADMIN_USER and body.password == ADMIN_PASS:
        return {"token": ADMIN_TOKEN}
    raise HTTPException(status_code=401, detail="invalid credentials")

@router.get("/me")
def me(token: str):
    if token == ADMIN_TOKEN:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="invalid token")
