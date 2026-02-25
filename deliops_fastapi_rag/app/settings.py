from __future__ import annotations
from typing import List, Optional
from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


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
        extra="ignore",
    )

    # --- API ---
    api_host: str = Field(default="127.0.0.1", validation_alias=AliasChoices("API_HOST",))
    api_port: int = Field(default=8000,        validation_alias=AliasChoices("API_PORT",))
    cors_origins_raw: Optional[str | List[str]] = Field(
        default=None, validation_alias=AliasChoices("CORS_ORIGINS",)
    )

    # --- OpenAI / RAG ---
    openai_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-3.5-turbo",
        validation_alias=AliasChoices("OPENAI_MODEL"),
    )
    embed_model: str = Field(
        default="text-embedding-3-small",
        validation_alias=AliasChoices("EMBED_MODEL", "RAG_EMBED_MODEL"),
    )
    rag_top_k: int = Field(
        default=4, validation_alias=AliasChoices("RAG_TOP_K",)
    )
    # --- OpenRouter (Agent LLM) ---
    openrouter_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY",),
    )
    openrouter_model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        validation_alias=AliasChoices("OPENROUTER_MODEL",),
    )

    # --- Postgres / pgvector ---
    database_url: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("DATABASE_URL",),
    )
    rag_similarity_threshold: float = Field(
        default=0.75,
        validation_alias=AliasChoices("RAG_SIMILARITY_THRESHOLD",),
    )

    # --- Stripe ---
    stripe_secret_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("STRIPE_SECRET_KEY", "STRIPE_API_KEY"),
    )
    stripe_publishable_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("STRIPE_PUBLISHABLE_KEY",),
    )

    # --- Admin Authentication ---
    admin_username: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ADMIN_USERNAME",),
    )
    admin_password: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ADMIN_PASSWORD",),
    )
    admin_token: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ADMIN_TOKEN",),
    )

    @property
    def cors_origins(self) -> List[str]:
        return _parse_cors(self.cors_origins_raw)


# singleton
settings = Settings()
