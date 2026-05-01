"""POST /api/tts/{video_id} — TTS with audio-sync endpoint (issue 381)."""

import asyncio
import functools
import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.services.tts_service import TTSService
from foreign_whispers.voice_resolution import resolve_speaker_wav

router = APIRouter(prefix="/api")


def _existing_voice(rel_path: str) -> str | None:
    """Return *rel_path* iff the WAV actually lives under speakers_dir, else None.

    ``resolve_speaker_wav`` returns ``default.wav`` per its test contract even
    when no WAV exists at any tier, so the API filters here to avoid spammy
    'speaker WAV not found' warnings inside the TTS engine.
    """
    if not rel_path:
        return None
    if (settings.speakers_dir / rel_path).exists():
        return rel_path
    return None


def _build_voice_map(translated_segments: list[dict], target_lang: str) -> dict[str, str]:
    """Map every distinct ``speaker`` field in *translated_segments* to a
    reference WAV path that actually exists on disk.  Returns empty dict when
    no segment has a label, or no resolved WAV exists."""
    speakers = sorted({
        s["speaker"] for s in translated_segments if s.get("speaker")
    })
    if not speakers:
        return {}
    out: dict[str, str] = {}
    for sp in speakers:
        rel = resolve_speaker_wav(settings.speakers_dir, target_lang, sp)
        existing = _existing_voice(rel)
        if existing:
            out[sp] = existing
    return out


async def _run_in_threadpool(executor, fn, *args, **kwargs):
    """Run a sync function in the default thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, functools.partial(fn, *args, **kwargs))


@router.post("/tts/{video_id}")
async def tts_endpoint(
    video_id: str,
    request: Request,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
    alignment: bool = Query(False),
    speaker_wav: str | None = Query(None, description="Reference voice WAV path relative to pipeline_data/speakers/ (e.g. 'es/default.wav')"),
    target_language: str = Query("es"),
):
    """Generate TTS audio for a translated transcript.

    *config* is an opaque directory name for caching.
    *alignment* enables temporal alignment (clamped stretch).
    *speaker_wav* is the call-level reference voice; when omitted the language
    default is auto-resolved.  If the translated transcript has per-segment
    ``speaker`` labels (from diarization), a per-speaker voice map is built
    automatically and overrides *speaker_wav* on a per-segment basis.
    """
    trans_dir = settings.translations_dir
    audio_dir = settings.tts_audio_dir / config
    audio_dir.mkdir(parents=True, exist_ok=True)

    svc = TTSService(
        ui_dir=settings.data_dir,
        tts_engine=None,
    )

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    wav_path = audio_dir / f"{title}.wav"

    if wav_path.exists():
        return {
            "video_id": video_id,
            "audio_path": str(wav_path),
            "config": config,
        }

    source_path = str(trans_dir / f"{title}.json")

    # Default reference voice — only set it if the resolved WAV actually exists
    if speaker_wav is None:
        speaker_wav = _existing_voice(resolve_speaker_wav(settings.speakers_dir, target_language))
    else:
        speaker_wav = _existing_voice(speaker_wav)

    # Per-speaker voice map (only if diarization populated speaker labels)
    voice_map: dict[str, str] = {}
    try:
        translated = json.loads(pathlib.Path(source_path).read_text())
        voice_map = _build_voice_map(translated.get("segments", []), target_language)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Translation missing: {source_path}")

    await _run_in_threadpool(
        None,
        svc.text_file_to_speech,
        source_path, str(audio_dir),
        alignment=alignment,
        speaker_wav=speaker_wav,
        voice_map=voice_map or None,
    )

    return {
        "video_id": video_id,
        "audio_path": str(wav_path),
        "config": config,
        "speaker_wav": speaker_wav,
        "voice_map": voice_map,
    }


@router.get("/audio/{video_id}")
async def get_audio(
    video_id: str,
    config: str = Query(..., pattern=r"^c-[0-9a-f]{7}$"),
):
    """Stream the TTS-synthesized WAV audio."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    audio_path = settings.tts_audio_dir / config / f"{title}.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")
