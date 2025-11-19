# app/services/embeddings.py  (same path as before)
from __future__ import annotations
from typing import List
import numpy as np

from openai import OpenAI
from ..settings import settings

# choose embedding model – you can set this via EMBED_MODEL env if you like
EMBED_MODEL = settings.embed_model or "text-embedding-3-small"

# text-embedding-3-small has 1536 dims; used only for the empty-case shortcut
EMBED_DIM = 1536

if not settings.openai_api_key:
    raise RuntimeError("OPENAI_API_KEY / openai_api_key is not set in settings")

_client = OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns an array of shape (len(texts), EMBED_DIM) with float32 embeddings.
    Uses OpenAI's embedding API instead of a local SentenceTransformer model.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    resp = _client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    embeddings = [d.embedding for d in resp.data]
    return np.asarray(embeddings, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
