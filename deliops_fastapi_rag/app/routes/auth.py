from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..settings import settings

router = APIRouter(prefix="/auth", tags=["auth"])


def _check_auth_configured() -> None:
    """Raise an error if auth credentials are not configured."""
    if not all([settings.admin_username, settings.admin_password, settings.admin_token]):
        raise HTTPException(
            status_code=503,
            detail="Authentication not configured. Set ADMIN_USERNAME, ADMIN_PASSWORD, and ADMIN_TOKEN in .env file."
        )


class LoginBody(BaseModel):
    username: str
    password: str


@router.post("/login")
def login(body: LoginBody):
    """Authenticate admin user and return token."""
    _check_auth_configured()
    if body.username == settings.admin_username and body.password == settings.admin_password:
        return {"token": settings.admin_token}
    raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/me")
def me(token: str = Query(...)):
    """Validate admin token."""
    _check_auth_configured()
    if token == settings.admin_token:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Invalid token")
