from __future__ import annotations
import os
import firebase_admin
from firebase_admin import credentials, firestore

_client: firestore.Client | None = None

def ensure_firestore() -> firestore.Client:
    """
    Lazily initialize and return a singleton Firestore client.
    Uses GOOGLE_APPLICATION_CREDENTIALS if provided, else ADC.
    """
    global _client
    if _client is not None:
        return _client

    if not firebase_admin._apps:
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if sa_path and os.path.isfile(sa_path):
            cred = credentials.Certificate(sa_path)
            firebase_admin.initialize_app(cred)
        else:
            # Falls back to Application Default Credentials if available
            firebase_admin.initialize_app()
    _client = firestore.client()
    return _client
