"""POST /api/transcribe/{video_id} — Whisper transcription (issue 58f, fw-29a)."""

import json
import pathlib

from fastapi import APIRouter, HTTPException, Query, Request

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.main import get_whisper_model
from api.src.schemas.transcribe import TranscribeResponse, TranscribeSegment
from api.src.services.transcription_service import TranscriptionService

router = APIRouter(prefix="/api")


def _youtube_captions_to_segments(caption_path: pathlib.Path) -> dict:
    """Convert YouTube line-delimited JSON captions to Whisper-compatible result dict."""
    segments = []
    full_text_parts = []
    for i, line in enumerate(caption_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        seg = json.loads(line)
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        duration = seg.get("duration", 0)
        if not text or duration <= 0:
            continue
        segments.append({
            "id": i,
            "start": start,
            "end": start + duration,
            "text": text,
        })
        full_text_parts.append(text)
    return {
        "language": "en",
        "text": " ".join(full_text_parts),
        "segments": segments,
    }


@router.post("/transcribe/{video_id}", response_model=TranscribeResponse)
async def transcribe_endpoint(
    video_id: str,
    request: Request,
    use_youtube_captions: bool = Query(True, description="Use YouTube captions when available, skipping Whisper"),
    word_timestamps: bool = Query(False, description="Force Whisper with word-level timestamps (skips YT captions; required for speaker-turn resegmentation)"),
):
    """Run Whisper transcription on a downloaded video.

    When use_youtube_captions is True (default), YouTube captions are used if
    available, skipping Whisper entirely. When False, Whisper always runs.
    When word_timestamps is True, YT captions are skipped and Whisper runs
    with word-level timestamps so each segment carries a ``words`` list.
    """
    if word_timestamps:
        use_youtube_captions = False
    videos_dir = settings.videos_dir
    transcriptions_dir = settings.transcriptions_dir
    transcriptions_dir.mkdir(parents=True, exist_ok=True)

    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in index")

    transcript_path = transcriptions_dir / f"{title}.json"

    # Return cached Whisper result if it exists and we're not forcing re-run.
    # When word_timestamps is requested, only reuse the cache if it has them.
    def _has_words(doc: dict) -> bool:
        segs = doc.get("segments", [])
        return bool(segs) and isinstance(segs[0].get("words"), list)

    if transcript_path.exists() and word_timestamps:
        cached = json.loads(transcript_path.read_text())
        if _has_words(cached):
            return TranscribeResponse(
                video_id=video_id,
                language=cached.get("language", "en"),
                text=cached.get("text", ""),
                segments=cached.get("segments", []),
                skipped=True,
            )
    if transcript_path.exists() and use_youtube_captions:
        data = json.loads(transcript_path.read_text())
        return TranscribeResponse(
            video_id=video_id,
            language=data.get("language", "en"),
            text=data.get("text", ""),
            segments=data.get("segments", []),
            skipped=True,
        )

    # When not forcing STT, prefer YouTube captions over running Whisper
    if use_youtube_captions:
        yt_caption_path = settings.youtube_captions_dir / f"{title}.txt"
        if yt_caption_path.exists():
            result = _youtube_captions_to_segments(yt_caption_path)
            transcript_path.write_text(json.dumps(result))
            return TranscribeResponse(
                video_id=video_id,
                language=result["language"],
                text=result["text"],
                segments=result["segments"],
                skipped=True,
            )

    # Run Whisper STT
    svc = TranscriptionService(
        ui_dir=settings.data_dir,
        whisper_model=get_whisper_model(request.app),
    )
    video_path = videos_dir / f"{title}.mp4"
    result = svc.transcribe(str(video_path), word_timestamps=word_timestamps)

    # Persist result
    transcript_path.write_text(json.dumps(result))

    return TranscribeResponse(
        video_id=video_id,
        language=result.get("language", "en"),
        text=result.get("text", ""),
        segments=result.get("segments", []),
    )
