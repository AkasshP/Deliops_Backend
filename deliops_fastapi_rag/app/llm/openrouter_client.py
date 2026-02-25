from __future__ import annotations

from typing import List, Dict, Any

import httpx
from fastapi import HTTPException

from ..settings import settings

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


async def chat_completion(
    messages: List[Dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> Dict[str, Any]:
    """Call OpenRouter chat completions and return a slim dict."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured")

    model = model or settings.openrouter_model

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "DeliOps-Backend-Agent",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(OPENROUTER_URL, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OpenRouter API error {exc.response.status_code}: {exc.response.text[:200]}",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OpenRouter request failed: {exc}",
        )

    choice = data.get("choices", [{}])[0]
    content = (choice.get("message") or {}).get("content", "")
    return {
        "content": content,
        "model": data.get("model", model),
        "usage": data.get("usage", {}),
    }
