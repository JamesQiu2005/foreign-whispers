"""Speaker diarization using pyannote.audio.

Extracted from notebooks/foreign_whispers_pipeline.ipynb (M2-align).

Optional dependency: pyannote.audio
    pip install pyannote.audio
Requires accepting the pyannote/speaker-diarization-3.1 licence on HuggingFace
and providing an HF token.  Returns empty list with a warning if the dep is
absent or the token is missing.
"""
import gc
import logging

logger = logging.getLogger(__name__)

_pyannote_pipeline = None
_pyannote_token_used: str | None = None


def _release_torch_caches() -> None:
    """Best-effort release of MPS / CUDA allocator slabs after heavy inference."""
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _get_pyannote_pipeline(hf_token: str):
    """Lazy singleton for the pyannote diarization pipeline.

    Reloads only if the HF token changes between calls. Re-instantiating
    the pipeline is what was leaking ~1 GB of unified memory per /diarize
    call on MPS — the caching allocator never shrinks for orphaned modules.
    """
    global _pyannote_pipeline, _pyannote_token_used
    if _pyannote_pipeline is not None and _pyannote_token_used == hf_token:
        return _pyannote_pipeline

    from pyannote.audio import Pipeline

    _pyannote_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    _pyannote_token_used = hf_token
    return _pyannote_pipeline


def diarize_audio(audio_path: str, hf_token: str | None = None) -> list[dict]:
    """Return speaker-labeled intervals for *audio_path*.

    Returns:
        List of ``{start_s: float, end_s: float, speaker: str}``.
        Empty list when pyannote.audio is absent, token is missing, or diarization fails.
    """
    if not hf_token:
        logger.warning("No HF token provided — diarization skipped.")
        return []

    try:
        import pyannote.audio  # noqa: F401
    except (ImportError, TypeError):
        logger.warning("pyannote.audio not installed — returning empty diarization.")
        return []

    try:
        pipeline = _get_pyannote_pipeline(hf_token)
        diarization = pipeline(audio_path)
        result = [
            {"start_s": turn.start, "end_s": turn.end, "speaker": speaker}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
    except Exception as exc:
        logger.warning("Diarization failed for %s: %s", audio_path, exc)
        return []
    finally:
        _release_torch_caches()
    return result


def assign_speakers(
    segments: list[dict],
    diarization: list[dict],
) -> list[dict]:
    """Assign a speaker label to each transcription segment.

    For each segment, finds the diarization interval with the greatest
    temporal overlap and copies its speaker label. If diarization is
    empty, all segments default to ``SPEAKER_00``.

    Args:
        segments: Whisper-style ``[{id, start, end, text, ...}]``.
        diarization: pyannote-style ``[{start_s, end_s, speaker}]``.

    Returns:
        New list of segment dicts, each with an added ``speaker`` key.
        Original list is not mutated.
    """
    out: list[dict] = []
    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]

        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for d in diarization:
            overlap = max(0.0, min(seg_end, d["end_s"]) - max(seg_start, d["start_s"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d["speaker"]

        out.append({**seg, "speaker": best_speaker})
    return out


def _flatten_words(whisper_result: dict) -> list[dict]:
    """Return a flat list of ``{word, start, end}`` from a Whisper result.

    Whisper-with-word-timestamps stores per-word timing under
    ``segment["words"]``. Each entry is ``{"word": str, "start": float,
    "end": float, "probability": float}``. Words missing start/end are
    skipped.
    """
    words: list[dict] = []
    for seg in whisper_result.get("segments", []):
        for w in seg.get("words", []) or []:
            if "start" in w and "end" in w:
                words.append({
                    "word": w.get("word", "").strip() or w.get("text", "").strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })
    return words


def resegment_by_speaker_turns(
    whisper_result: dict,
    diarization: list[dict],
    *,
    merge_gap_s: float = 0.5,
) -> dict:
    """Rebuild a Whisper-style result whose segments follow speaker turns.

    Each output segment corresponds to one contiguous speaker turn (consecutive
    same-speaker pyannote intervals separated by < *merge_gap_s* are merged).
    The text for a segment is the concatenation of Whisper words whose
    midpoint falls within the merged turn window.

    Args:
        whisper_result: Whisper output dict with ``word_timestamps=True`` —
            each segment has a ``words`` list.
        diarization: pyannote output ``[{start_s, end_s, speaker}]``.
        merge_gap_s: max silence between same-speaker turns to merge.

    Returns:
        ``{"language", "text", "segments": [...]}`` where each segment is
        ``{id, start, end, text, speaker}``. Falls back to the original
        ``whisper_result`` when diarization is empty or no words are available.
    """
    if not diarization:
        return whisper_result

    words = _flatten_words(whisper_result)
    if not words:
        return whisper_result

    sorted_diar = sorted(diarization, key=lambda d: d["start_s"])
    merged: list[dict] = []
    for d in sorted_diar:
        if (
            merged
            and d["speaker"] == merged[-1]["speaker"]
            and d["start_s"] - merged[-1]["end_s"] <= merge_gap_s
        ):
            merged[-1]["end_s"] = max(merged[-1]["end_s"], d["end_s"])
        else:
            merged.append(dict(d))

    out_segments: list[dict] = []
    for idx, turn in enumerate(merged):
        t_start, t_end = turn["start_s"], turn["end_s"]
        bucket = [
            w for w in words
            if t_start <= 0.5 * (w["start"] + w["end"]) <= t_end
        ]
        if not bucket:
            continue
        text = " ".join(w["word"] for w in bucket if w["word"]).strip()
        if not text:
            continue
        out_segments.append({
            "id": idx,
            "start": min(w["start"] for w in bucket),
            "end": max(w["end"] for w in bucket),
            "text": text,
            "speaker": turn["speaker"],
            "words": bucket,
        })

    return {
        "language": whisper_result.get("language", "en"),
        "text": " ".join(s["text"] for s in out_segments),
        "segments": out_segments,
    }
