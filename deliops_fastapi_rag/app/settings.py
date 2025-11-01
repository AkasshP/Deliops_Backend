# app/settings.py
from __future__ import annotations
from typing import List, Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
import json
import os

def _parse_cors(v: Optional[str | List[str]]) -> List[str]:
    """
    Accept JSON array (e.g. '["http://localhost:3000"]') or
    comma-separated string ('http://localhost:3000,http://127.0.0.1:3000').
    """
    if v is None:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    if isinstance(v, list):
        return v
    s = v.strip()
    if not s:
        return ["http://localhost:3000", "http://127.0.0.1:3000"]
    # try JSON first
    try:
        parsed = json.loads(s)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except Exception:
        pass
    # fallback: comma separated
    return [p.strip() for p in s.split(",") if p.strip()]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",   # ignore unknown env keys instead of raising
    )

    # --- API ---
    api_host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("API_HOST",))
    api_port: int = Field(default=8000,        validation_alias=AliasChoices("API_PORT",))
    cors_origins_raw: Optional[str | List[str]] = Field(
        default=None, validation_alias=AliasChoices("CORS_ORIGINS",)
    )

    # --- Firebase ---
    firebase_project_id: str = Field(
        default="deliops",
        validation_alias=AliasChoices("FIREBASE_PROJECT_ID",)
    )
    google_application_credentials: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_APPLICATION_CREDENTIALS",)
    )

    # --- Hugging Face / RAG ---
    huggingface_api_key: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("HUGGINGFACE_API_KEY",)
    )
    hf_model_id: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        validation_alias=AliasChoices("HF_MODEL_ID",)
    )
    # accept either EMBED_MODEL or RAG_EMBED_MODEL
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias=AliasChoices("EMBED_MODEL", "RAG_EMBED_MODEL")
    )
    rag_top_k: int = Field(
        default=4, validation_alias=AliasChoices("RAG_TOP_K",)
    )

    @property
    def cors_origins(self) -> List[str]:
        return _parse_cors(self.cors_origins_raw)

# singleton
settings = Settings()

# Make sure GOOGLE_APPLICATION_CREDENTIALS is exported for firebase_admin
if settings.google_application_credentials:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.google_application_credentials
