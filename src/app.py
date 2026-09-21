from __future__ import annotations

import io
import os
import threading
import wave

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from kokoro import KPipeline
from pydantic import BaseModel, Field


SAMPLE_RATE = 24_000

LANG_CODE = os.getenv("TTS_LANG_CODE", "a")
DEFAULT_VOICE = os.getenv("TTS_DEFAULT_VOICE", "af_heart")


app = FastAPI(
    title="Local Text To Speech",
    version="0.1.0",
)


# Load Kokoro once when the service starts.
pipeline = KPipeline(
    lang_code=LANG_CODE,
)

# For now serialize synthesis requests.
# This avoids multiple requests trying to use the same model simultaneously.
synthesis_lock = threading.Lock()


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1)
    voice: str = DEFAULT_VOICE
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ready",
        "backend": "kokoro",
        "language": LANG_CODE,
        "default_voice": DEFAULT_VOICE,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/voices")
def voices() -> dict[str, object]:
    # Small initial selection.
    # We can expose the complete Kokoro voice catalogue later.
    return {
        "voices": [
            "af_heart",
            "af_bella",
            "af_nicole",
            "af_sarah",
            "am_fenrir",
            "am_michael",
            "am_puck",
            "am_santa",
        ],
        "default": DEFAULT_VOICE,
    }


@app.post("/v1/audio/speech")
def create_speech(request: SpeechRequest) -> Response:
    audio_chunks: list[np.ndarray] = []

    try:
        with synthesis_lock:
            generator = pipeline(
                request.input,
                voice=request.voice,
                speed=request.speed,
            )

            for result in generator:
                audio = result.audio

                if audio is None:
                    continue

                # Kokoro returns a torch tensor.
                chunk = audio.detach().cpu().numpy()

                audio_chunks.append(chunk)

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Speech generation failed: {exc}",
        ) from exc

    if not audio_chunks:
        raise HTTPException(
            status_code=500,
            detail="Kokoro generated no audio.",
        )

    audio = np.concatenate(audio_chunks)

    # Kokoro audio is floating point in approximately [-1, 1].
    audio = np.clip(audio, -1.0, 1.0)

    pcm = (audio * 32767.0).astype(np.int16)

    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())

    return Response(
        content=buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'inline; filename="speech.wav"',
        },
    )