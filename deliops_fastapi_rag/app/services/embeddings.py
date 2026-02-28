# app/services/embeddings.py
from __future__ import annotations
from typing import List
import numpy as np
from huggingface_hub import InferenceClient
from ..settings import settings

EMBED_MODEL = settings.embed_model or "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 outputs 384 dims

if not settings.hf_api_token:
    raise RuntimeError("HF_API_TOKEN is not set — get a free token at https://huggingface.co/settings/tokens")

_client = InferenceClient(token=settings.hf_api_token)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return (len(texts), EMBED_DIM) float32 embeddings via HF Inference API."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    result = _client.feature_extraction(texts, model=EMBED_MODEL)
    return np.asarray(result, dtype=np.float32)


def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
