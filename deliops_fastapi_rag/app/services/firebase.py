# app/services/firebase.py
from __future__ import annotations

import os
from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore


@lru_cache
def ensure_firestore() -> firestore.Client:
    """
    Return a Firestore client, initializing the Firebase app exactly once.

    - Safe to call many times (and from many threads).
    - Uses GOOGLE_APPLICATION_CREDENTIALS if present, or ADC otherwise.
    """

    if not firebase_admin._apps:
        # First-time init
        sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        try:
            if sa_path and os.path.isfile(sa_path):
                cred = credentials.Certificate(sa_path)
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()
        except ValueError:
            # If another request initialized between our check and this call,
            # just ignore the "app already exists" error and continue.
            pass

    return firestore.client()
