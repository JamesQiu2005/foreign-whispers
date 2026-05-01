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

from fastapi import APIRouter, HTTPException

from api.src.core.config import settings
from api.src.core.dependencies import resolve_title
from api.src.schemas.diarize import DiarizeResponse
from api.src.services.alignment_service import AlignmentService
from foreign_whispers.diarization import assign_speakers

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
    """Add a ``speaker`` field to every segment in the cached transcription JSON."""
    transcript_path = settings.transcriptions_dir / f"{title}.json"
    if not transcript_path.exists():
        return
    transcript = json.loads(transcript_path.read_text())
    transcript["segments"] = assign_speakers(
        transcript.get("segments", []),
        diar_segments,
    )
    transcript_path.write_text(json.dumps(transcript))


@router.post("/diarize/{video_id}", response_model=DiarizeResponse)
async def diarize_endpoint(video_id: str):
    """Run speaker diarization on a video's audio track."""
    title = resolve_title(video_id)
    if title is None:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")

    diar_dir = settings.diarizations_dir
    diar_dir.mkdir(parents=True, exist_ok=True)
    diar_path = diar_dir / f"{title}.json"

    if diar_path.exists():
        data = json.loads(diar_path.read_text())
        _merge_speakers_into_transcript(title, data.get("segments", []))
        return DiarizeResponse(
            video_id=video_id,
            speakers=data.get("speakers", []),
            segments=data.get("segments", []),
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

    _merge_speakers_into_transcript(title, diar_segments)

    return DiarizeResponse(
        video_id=video_id,
        speakers=speakers,
        segments=diar_segments,
    )
