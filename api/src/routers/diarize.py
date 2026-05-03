"""POST /api/diarize/{video_id} — speaker diarization.

Steps:
1. Extract a 16kHz mono WAV from the source video via ffmpeg.
2. Run pyannote speaker-diarization-3.1 (in-process, optional dep).
3. Cache speakers + segments to ``diarizations/<title>.json``.
4. Merge speaker labels into the cached transcription JSON so downstream
   stages (translate, TTS) can switch voices per speaker.
"""

import json
import subprocess

from fastapi import APIRouter, HTTPException, Request

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from api.src.services.transcription_service import TranscriptionService
from foreign_whispers.diarization import assign_speakers, resegment_by_speaker_turns

router = APIRouter(prefix="/api")

_alignment_service = AlignmentService(settings=settings)


def _extract_audio(video_path, audio_path) -> None:
    """ffmpeg: video → 16kHz mono PCM WAV (pyannote requirement)."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            str(audio_path),
        ],
        check=True,
    )


def _merge_speakers_into_transcript(title: str, diar_segments: list[dict]) -> None:
    """Add a ``speaker`` field to every segment in the cached transcription JSON
    *and* the cached translation JSON (the TTS voice_map reads from the latter).
    """
    for path in (
        settings.transcriptions_dir / f"{title}.json",
        settings.translations_dir / f"{title}.json",
    ):
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        doc["segments"] = assign_speakers(doc.get("segments", []), diar_segments)
        path.write_text(json.dumps(doc))


def _has_word_timestamps(doc: dict) -> bool:
    segs = doc.get("segments", [])
    return bool(segs) and isinstance(segs[0].get("words"), list) and bool(segs[0].get("words"))


def _ensure_word_level_transcription(request: Request, title: str) -> dict:
    """Return a Whisper transcription with per-word timestamps for *title*.

    Re-runs Whisper with ``word_timestamps=True`` if the cached transcription
    lacks word-level timing (e.g. it came from YouTube captions).
    """
    from api.src.main import get_whisper_model

    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if transcript_path.exists():
        doc = json.loads(transcript_path.read_text())
        if _has_word_timestamps(doc):
            return doc

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Source video missing: {video_path.name}")

    svc = TranscriptionService(ui_dir=settings.data_dir, whisper_model=get_whisper_model(request.app))
    result = svc.transcribe(str(video_path), word_timestamps=True)
    settings.transcriptions_dir.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps(result))
    return result


def _resegment_and_invalidate(title: str, whisper_doc: dict, diar_segments: list[dict]) -> dict:
    """Overwrite the cached transcription with speaker-turn segments and
    invalidate the cached translation so it re-runs on the new segmentation.

    Returns the new transcription dict.
    """
    new_doc = resegment_by_speaker_turns(whisper_doc, diar_segments)
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    transcript_path.write_text(json.dumps(new_doc))

    translation_path = settings.translations_dir / f"{title}.json"
    if translation_path.exists():
        translation_path.unlink()

    return new_doc


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str, request: Request):
    """Run speaker diarization, then re-segment the transcript by speaker turn.

    Flow:
    1. pyannote on the video's audio → speaker turns.
    2. Whisper with word-level timestamps (re-run if cache lacks them).
    3. Bucket Whisper words into diarization turns to produce a new
       per-speaker-turn transcription that overwrites the cached one.
    4. Invalidate the cached translation so it re-runs against the new
       segmentation. Caller is expected to invoke /translate next.
    """
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        diar_segments = data.get("segments", [])
        whisper_doc = _ensure_word_level_transcription(request, title)
        _resegment_and_invalidate(title, whisper_doc, diar_segments)
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=diar_segments,
            skipped=True,
        )

    video_path = settings.videos_dir / f"{title}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Source video missing: {video_path.name}")

    audio_path = diar_dir / f"{title}.wav"
    try:
        _extract_audio(video_path, audio_path)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=500, detail=f"ffmpeg audio extract failed: {exc}")

    diar_segments = _alignment_service.diarize(str(audio_path))

    speakers = sorted({s["speaker"] for s in diar_segments}) if diar_segments else []
    result = {"speakers": speakers, "segments": diar_segments}
    diar_path.write_text(json.dumps(result))

    whisper_doc = _ensure_word_level_transcription(request, title)
    _resegment_and_invalidate(title, whisper_doc, diar_segments)

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
    )
