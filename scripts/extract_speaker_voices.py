#!/usr/bin/env python3
"""Extract per-speaker reference WAVs from a source video using diarization output.

For each unique ``SPEAKER_NN`` in the diarization JSON, picks the longest single
clean turn (clipped to a configurable max — Chatterbox's sweet spot is 5-15s),
runs ``ffmpeg`` to slice it from the source MP4 as mono 16kHz WAV, and saves to
``pipeline_data/speakers/<lang>/SPEAKER_NN.wav``.

The TTS path's existing ``resolve_speaker_wav`` and ``_build_voice_map`` already
look up ``pipeline_data/speakers/<lang>/<speaker_id>.wav``, so dropping these
files in is enough — no backend changes required.

Prerequisites:
    1. Diarization endpoint has run successfully (non-empty
       ``pipeline_data/api/diarizations/<title>.json``).
    2. Source video at ``pipeline_data/api/videos/<title>.mp4``.
    3. ``ffmpeg`` on PATH.

Usage:
    uv run python scripts/extract_speaker_voices.py <video_id>
    uv run python scripts/extract_speaker_voices.py <video_id> --lang es --max-seconds 12
"""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DATA = PROJECT_ROOT / "pipeline_data"
REGISTRY_PATH = PROJECT_ROOT / "video_registry.yml"


def resolve_title(video_id: str) -> tuple[str, str]:
    """Return ``(title, target_language)`` from ``video_registry.yml``."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    for v in registry.get("videos", []):
        if v["id"] == video_id:
            return v["title"], v["target_language"]
    raise SystemExit(f"video_id {video_id!r} not found in {REGISTRY_PATH}")


def pick_reference_interval(
    intervals: list[dict],
    max_seconds: float,
    min_seconds: float,
) -> tuple[float, float] | None:
    """Choose ``(start_s, duration_s)`` for the cleanest reference clip.

    Strategy: longest single contiguous turn, clipped to ``max_seconds``,
    with a 0.2s lead-in trim to dodge crosstalk at boundaries. Returns
    ``None`` when no turn meets ``min_seconds``.
    """
    if not intervals:
        return None
    longest = max(intervals, key=lambda i: i["end_s"] - i["start_s"])
    duration = longest["end_s"] - longest["start_s"]
    if duration < min_seconds:
        return None
    start = longest["start_s"] + 0.2
    duration = min(duration - 0.2, max_seconds)
    return start, duration


def extract_wav(video_path: Path, start: float, duration: float, out_path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", f"{start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000",
            "-c:a", "pcm_s16le",
            str(out_path),
        ],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video_id", help="YouTube video id (key in video_registry.yml)")
    ap.add_argument("--lang", help="Override target language (defaults to registry value)")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--min-seconds", type=float, default=5.0)
    args = ap.parse_args()

    title, registry_lang = resolve_title(args.video_id)
    lang = args.lang or registry_lang

    video_path = PIPELINE_DATA / "api" / "videos" / f"{title}.mp4"
    diar_path = PIPELINE_DATA / "api" / "diarizations" / f"{title}.json"
    out_dir = PIPELINE_DATA / "speakers" / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise SystemExit(f"video file missing: {video_path}")
    if not diar_path.exists():
        raise SystemExit(
            f"diarization missing: {diar_path}\n"
            f"Run POST /api/diarize/{args.video_id} first."
        )

    diar = json.loads(diar_path.read_text())
    segments = diar.get("segments", [])
    if not segments:
        raise SystemExit(
            f"{diar_path} contains no diarization intervals.\n"
            "The diarize endpoint ran but produced empty output. Likely causes:\n"
            "  - FW_HF_TOKEN not set when uvicorn started\n"
            "  - pyannote/speaker-diarization-3.1 license not accepted on HuggingFace\n"
            "  - pyannote.audio not installed (uv sync --group alignment)\n"
            f"Delete {diar_path} and re-run the diarize endpoint."
        )

    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for s in segments:
        by_speaker[s["speaker"]].append(s)

    print(f"Source video: {video_path.name}")
    print(f"Diarization:  {len(segments)} intervals across {len(by_speaker)} speakers")
    print(f"Output dir:   {out_dir}")
    print()

    for speaker in sorted(by_speaker):
        intervals = by_speaker[speaker]
        total = sum(i["end_s"] - i["start_s"] for i in intervals)
        choice = pick_reference_interval(intervals, args.max_seconds, args.min_seconds)
        if choice is None:
            print(
                f"  {speaker}: total={total:.1f}s — no single turn >= {args.min_seconds}s, "
                f"skipping (will fall back to default.wav)"
            )
            continue
        start, duration = choice
        out_path = out_dir / f"{speaker}.wav"
        extract_wav(video_path, start, duration, out_path)
        print(
            f"  {speaker}: total={total:.1f}s -> {duration:.1f}s clip "
            f"from {start:.1f}s -> {out_path.name}"
        )

    print("\nDone. Re-run TTS to dub with the new per-speaker reference voices.")


if __name__ == "__main__":
    main()
