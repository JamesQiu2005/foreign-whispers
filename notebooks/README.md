# Phase 2 Notebooks — Study Guide

Each subfolder is one notebook in the course's Phase 2 sequence. Read the
`NOTES.md` next to each `.ipynb` for a personal cheat sheet — what the
notebook is teaching, what code it touches, and what the graded task is.

## Order to work through

| # | Notebook | Tasks | Needs running services |
|---|---|---|---|
| 1 | [`download_integration`](download_integration/NOTES.md) | exploration only | API |
| 2 | [`transcription_integration`](transcription_integration/NOTES.md) | exploration only | API |
| 3 | [`translation_integration`](translation_integration/NOTES.md) | 1 — `get_shorter_translations()` | API |
| 4 | [`diarization_integration`](diarization_integration/NOTES.md) | 5 — speaker assignment + endpoint + UI | API + `FW_HF_TOKEN` |
| 5 | [`alignment_integration`](alignment_integration/NOTES.md) | 4 — `_estimate_duration`, DP align, scorecard | none (pure-Python) |
| 6 | [`tts_integration`](tts_integration/NOTES.md) | 4 — voice cloning + per-speaker | API + Chatterbox |
| 7 | [`stitch_integration`](stitch_integration/NOTES.md) | verification only | API |

`pipeline_end_to_end/` is the Phase 1 demo, not graded — useful as a
reference for how stages chain together.

## Running notebooks

VS Code: open the `.ipynb`, click the kernel picker, choose **Python (foreign-whispers)**.

Jupyter Lab from the terminal:
```bash
uv run jupyter lab notebooks/
```

To bring up the services these notebooks talk to:

```bash
# Chatterbox TTS (MPS, port 8020)
cd tools/chatterbox_server && uv run python serve.py

# Foreign Whispers API (port 8080)
FW_WHISPER_MODEL=base uv run uvicorn api.src.main:app --host 0.0.0.0 --port 8080
```
