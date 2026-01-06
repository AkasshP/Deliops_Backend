from __future__ import annotations
from typing import List, Dict, Any, Tuple
from datetime import datetime
import os
import json
import numpy as np


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime objects."""
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, '__str__'):
            return str(obj)
        return super().default(obj)


class SimpleStore:
    """
    Minimal cosine-similarity store persisted to disk.
    Files:
      - embeddings.npy (float32, L2-normalized)
      - metas.json     (list[dict])
    """
    def __init__(self, dirpath: str):
        self.dir = dirpath
        self.emb = None  # np.ndarray [n, d]
        self.metas: List[Dict[str, Any]] = []

    # ---------- persistence ----------
    def load(self) -> bool:
        epath = os.path.join(self.dir, "embeddings.npy")
        mpath = os.path.join(self.dir, "metas.json")
        if not (os.path.exists(epath) and os.path.exists(mpath)):
            return False
        self.emb = np.load(epath).astype(np.float32)
        with open(mpath, "r", encoding="utf-8") as f:
            self.metas = json.load(f)
        return True

    def save(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        epath = os.path.join(self.dir, "embeddings.npy")
        mpath = os.path.join(self.dir, "metas.json")
        np.save(epath, self.emb.astype(np.float32) if self.emb is not None else np.zeros((0, 384), np.float32))
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(self.metas, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)

    # ---------- building ----------
    def build(self, vectors: np.ndarray, metas: List[Dict[str, Any]]) -> None:
        if vectors.shape[0] != len(metas):
            raise ValueError("vectors and metas length mismatch")
        if vectors.size == 0:
            self.emb = np.zeros((0, 384), np.float32)
            self.metas = []
            return
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        self.emb = (vectors / norms).astype(np.float32)
        self.metas = metas

    # ---------- search ----------
    def search(self, q: np.ndarray, top_k: int = 4) -> List[Tuple[float, Dict[str, Any]]]:
        if self.emb is None or len(self.metas) == 0:
            return []
        q = q.astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        sims = (self.emb @ q)  # [n]
        idx = np.argsort(-sims)[:top_k]
        return [(float(sims[i]), self.metas[i]) for i in idx]
