# app/services/firestore_client.py
from __future__ import annotations
from typing import Optional
import os
import firebase_admin
from firebase_admin import credentials, firestore

from app.settings import settings

_db: Optional[firestore.Client] = None

def get_db() -> firestore.Client:
    global _db
    if _db is not None:
        return _db

    sa_path = settings.google_application_credentials
    if not sa_path or not os.path.exists(sa_path):
        raise RuntimeError(
            f"Service account file not found at {sa_path!r}. "
            "Set GOOGLE_APPLICATION_CREDENTIALS in your .env."
        )

    # Ensure the process env also has it (handy for libs that look at env)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = sa_path

    try:
        firebase_admin.get_app()
    except ValueError:
        cred = credentials.Certificate(sa_path)
        firebase_admin.initialize_app(cred, {
            "projectId": settings.firebase_project_id
        })

    _db = firestore.client()
    return _db
