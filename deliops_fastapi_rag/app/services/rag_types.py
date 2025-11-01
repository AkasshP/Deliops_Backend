# app/services/rag_types.py
from __future__ import annotations
from typing import Protocol, List
import numpy as np

class EmbeddingsLike(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray: ...
