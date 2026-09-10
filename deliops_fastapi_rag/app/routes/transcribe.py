from fastapi import APIRouter, HTTPException, Request
import httpx

from ..settings import settings


router = APIRouter(tags=["transcription"])


@router.post("/transcribe")
async def transcribe(request: Request) -> dict[str, str]:
    """Transcribe browser-recorded audio without exposing the Deepgram key."""
    if not settings.deepgram_api_key:
        raise HTTPException(status_code=503, detail="Transcription is not configured")

    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="Audio is required")

    content_type = request.headers.get("content-type", "audio/webm")
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true",
            headers={
                "Authorization": f"Token {settings.deepgram_api_key}",
                "Content-Type": content_type,
            },
            content=audio,
        )

    if not response.is_success:
        raise HTTPException(status_code=502, detail="Transcription provider failed")

    payload = response.json()
    text = payload.get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    return {"text": text}
