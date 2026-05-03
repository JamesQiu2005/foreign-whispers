"""HTTP-agnostic service wrapping Whisper transcription."""

import json
import pathlib
from pathlib import Path
from typing import Any


class TranscriptionService:
    """Thin wrapper around the Whisper model for transcription.

    Accepts *ui_dir* and a pre-loaded *whisper_model* via constructor injection.
    """

    def __init__(self, ui_dir: Path, whisper_model: Any) -> None:
        self.ui_dir = ui_dir
        self.whisper_model = whisper_model

    def transcribe(self, video_path: str, *, word_timestamps: bool = False) -> dict:
        """Run Whisper transcription on a video file and return the result dict.

        When *word_timestamps* is True, each segment carries a ``words`` list
        of ``{word, start, end, probability}`` — required for speaker-turn
        re-segmentation.
        """
        try:
            return self.whisper_model.transcribe(video_path, word_timestamps=word_timestamps)
        finally:
            self._release_torch_caches()

    @staticmethod
    def _release_torch_caches() -> None:
        """Drop MPS / CUDA allocator slabs after a transcribe call.

        Why: Whisper's word-timestamp DTW path allocates per-chunk
        cross-attention tensors that the MPS caching allocator never shrinks
        on its own. Without this, repeated /transcribe or /diarize calls
        balloon RSS by GBs each.
        """
        import gc
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    @staticmethod
    def title_for_video_id(video_id: str, search_dir: pathlib.Path) -> str | None:
        """Find a title by scanning *search_dir* for matching files.

        Returns the stem (title) of the first match, or None.
        """
        for f in search_dir.glob("*.mp4"):
            return f.stem
        return None
