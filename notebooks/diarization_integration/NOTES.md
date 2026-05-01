# Notebook 4 — `diarization_integration`

## What it teaches

Multi-speaker videos need *per-speaker* TTS voices, otherwise everyone in the dub sounds identical. This notebook adds **speaker diarization** (pyannote) as a new pipeline stage between transcription and translation. **5 coded tasks** spanning library + API + frontend.

```
BEFORE:  Download → Transcribe → Translate → TTS → Stitch
AFTER:   Download → Transcribe → Diarize → Translate → TTS (per-speaker) → Stitch
                                    ↑
                              YOUR WORK
```

This is the **largest** notebook in Phase 2 — read top to bottom before writing code.

## Prerequisites

- `FW_HF_TOKEN` exported (or shell-sourced from `.env`) **before** the API starts. The hosted-pyannote download is gated.
- The model license accepted at https://huggingface.co/pyannote/speaker-diarization-3.1 — without this, even a valid token gets `403 Forbidden`.
- A multi-speaker video (the Strait of Hormuz interview qualifies; the standalone Alysa Liu interview is mostly one speaker).

## Cell-by-cell map

| Cell | What it does | Code or read? |
|---|---|---|
| 0–1 | Setup, file map | read |
| 2 | sys.path shim | run |
| 3–5 | Read `foreign_whispers/diarization.py` to see what's already there | read |
| 6–7 | TDD: failing tests for `assign_speakers` | run (will fail) |
| 8–9 | The function stub | **copy into the source file** |
| 10–11 | Re-run the tests | run (must pass) |
| 12 | git commit reminder | — |
| 13–24 | Task 2: `POST /api/diarize/{id}` endpoint | edit + restart API |
| 25–29 | Task 3: merge speaker labels back into transcription JSON | edit + restart |
| 30–37 | Task 4: frontend wiring (TS + React) | edit + `pnpm dev` (Mac path; ignore the docker rebuild cell) |
| 38–42 | Task 5: per-speaker voice mapping in TTS | edit + restart |
| 43 | Evaluation criteria | — |

## The 5 graded tasks

### Task 1 — `assign_speakers(segments, diarization)` in `foreign_whispers/diarization.py`

Pure function. Each transcription segment gets the speaker label of the diarization interval it overlaps most with. Tests in cell 7 are tight; meet them and move on. Algorithm fits in 15 lines.

### Task 2 — `POST /api/diarize/{video_id}` endpoint

Cells 14–18 walk you through it concretely:
1. Add `diarizations_dir` property to `Settings` (`api/src/core/config.py`).
2. New schema file `api/src/schemas/diarize.py` (cell 16 writes it for you).
3. New router file `api/src/routers/diarize.py` (cell 18 writes the stub; you fill in 5 numbered steps in the `YOUR CODE HERE` block).
4. Register the router in `api/src/main.py` (cell 19).

The router stub already includes the file-exists cache check — you just fill in the ffmpeg audio extract → `_alignment_service.diarize(audio_path)` → cache → return path.

### Task 3 — Merge speaker labels into the transcription JSON

After diarization succeeds, re-write `transcriptions/whisper/<title>.json` so each segment has a `speaker` field. Cell 26 gives the exact 5 lines of code to add.

This is what unblocks per-speaker TTS later — your `assign_speakers` from Task 1 finally gets called from the API path.

### Task 4 — Frontend pipeline integration (TypeScript)

5 files to edit (paths in cell 30). The chain is:
- `frontend/src/lib/api.ts` — new `diarizeVideo()` client function
- `frontend/src/lib/types.ts` — extend the `PipelineStage` union
- `frontend/src/hooks/use-pipeline.ts` — call `diarizeVideo` between transcribe and translate, *only* when `settings.diarization.length > 0`
- `frontend/src/components/pipeline-table.tsx` — render the new row
- `frontend/src/components/pipeline-status-bar.tsx` — status string

This is purely "wire one new stage into the existing state machine" — no algorithmic work.

### Task 5 — Per-speaker TTS voice mapping

Open-ended. Pick one of:
- **Round-robin**: `SPEAKER_00 → es/voice_1.wav`, `SPEAKER_01 → es/voice_2.wav`, …
- **Filename-based**: `SPEAKER_NN.wav` direct match
- **Default + override**: all speakers use `default.wav` unless a `SPEAKER_NN.wav` exists

Document the strategy in cell 41 (markdown), implement in `tts_service.py` (cell 42 hint).

## Mac/MPS-specific notes

- **Ignore every `!docker compose ... build api` cell.** On bare-metal Mac, you restart by killing uvicorn and starting it again. The current process IDs are in `/tmp/fw-api.log`. Quick recipe:
  ```bash
  pkill -f "uvicorn api.src.main"
  set -a && source .env && set +a
  FW_WHISPER_MODEL=base uv run uvicorn api.src.main:app --host 0.0.0.0 --port 8080 > /tmp/fw-api.log 2>&1 &
  ```
- **`FW_HF_TOKEN` reading.** The current `config.py` doesn't auto-load `.env`. Either export it before launching uvicorn (the `set -a; source .env; set +a` pattern), or add `env_file=".env"` to `model_config` in `config.py`.
- **pyannote on MPS.** It runs (CPU + a few MPS-accelerated ops). A 13-min interview takes ~30–60s on M3 Max. Slower than CUDA but tractable.
- **Frontend rebuild on Mac.** Replace cells 36 (`docker compose build frontend`) with the dev-mode workflow:
  ```bash
  cd frontend && pnpm dev    # already running? Hot-reload picks up changes automatically
  ```

## Cross-notebook connections

- **Diarization output → `tts_integration` Task 4.** Per-speaker voice assignment in TTS depends on segments having a `speaker` field — that's exactly what Task 3 here writes.
- **Diarization → `alignment_integration`.** A future-extension idea: penalize the global aligner for borrowing silence across speaker boundaries. Out of scope here, but the labeled JSON makes it possible.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `403` on first diarize call | HF gated model not accepted | Visit pyannote-3.1 model page, click "Agree" |
| `401` even with HF token | Token without correct scopes | Generate a new token with at least `read` scope |
| `ImportError: pyannote.audio` | Optional dep not installed | `uv sync --group alignment` (the `alignment` extra in pyproject) |
| Diarize endpoint hangs | First-time model download | One-time, ~500 MB |
| Test 4 fails ("no mutation") | You did `seg["speaker"] = ...` instead of building a new dict | Use `dict(seg, speaker=...)` or `{**seg, "speaker": ...}` |
