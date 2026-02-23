# app/services/firebase.py
from __future__ import annotations

import json
import os
from functools import lru_cache

import firebase_admin
from firebase_admin import credentials, firestore


@lru_cache
def ensure_firestore() -> firestore.Client:
    """
    Return a Firestore client, initializing the Firebase app exactly once.

    - Safe to call many times (and from many threads).
    - Checks (in order):
      1. GOOGLE_CREDENTIALS_JSON env var (raw JSON string — for cloud deploys)
      2. GOOGLE_APPLICATION_CREDENTIALS file path (local dev)
      3. Application Default Credentials (ADC)
    """

    if not firebase_admin._apps:
        try:
            cred_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
            sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

            if cred_json:
                info = json.loads(cred_json)
                cred = credentials.Certificate(info)
                firebase_admin.initialize_app(cred)
            elif sa_path and os.path.isfile(sa_path):
                cred = credentials.Certificate(sa_path)
                firebase_admin.initialize_app(cred)
            else:
                firebase_admin.initialize_app()
        except ValueError:
            # If another request initialized between our check and this call,
            # just ignore the "app already exists" error and continue.
            pass

    return firestore.client()
