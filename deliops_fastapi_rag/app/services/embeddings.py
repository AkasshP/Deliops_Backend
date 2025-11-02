from __future__ import annotations
from typing import List
from functools import lru_cache
import numpy as np

from ..settings import settings

@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.embed_model, device="cpu")
    return model

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = _load_model()
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    return vecs.astype(np.float32)

def embed_text(text: str) -> np.ndarray:
    return embed_texts([text])[0]
