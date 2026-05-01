# Notebook 6 — `tts_integration`

## What it teaches

The text → audio leg, with two upgrades on top of the baseline: **alignment** (time-stretch each segment to fit its source window) and **voice cloning** (different speakers get different voices). **4 coded tasks**, mostly plumbing — but the plumbing has high payoff.

```
es_segments.json ──► Chatterbox /v1/audio/speech ──► tts_audio/<config>/<title>.wav
                              ▲
                              └── speaker_wav (reference) ──► voice cloning
```

## Notebook ↔ source path mismatch (read this first)

The notebook hints reference `tts.py` at the *repo root*. **That file no longer exists.** Per the recent commit `81d4087: fix: move download_video and translate_en_to_es into api/src/services/`, the Chatterbox client and segment-orchestration logic now live in:

```
api/src/services/tts_engine.py    ← ChatterboxClient + text_file_to_speech
```

So when the notebook says *"open `tts.py`"* — open `api/src/services/tts_engine.py` instead. Cell 12 (which `Path("tts.py").read_text()`) will fail; just read the file directly in your editor.

## Prerequisites

- API up at :8080 *and* Chatterbox up at :8020 (we have both running locally).
- An EN→ES translation already on disk (notebook 3 produced one).
- Speaker reference WAVs at `pipeline_data/speakers/{lang}/*.wav`. Inspect what's there in cell 10.

## Cell-by-cell map

| Cell | What | Action |
|---|---|---|
| 0–2 | Setup, FWClient, healthz | run |
| 3–4 | Baseline (no alignment) TTS | run — produces `tts_audio/c-fb1074a/<title>.wav` |
| 5–6 | Aligned TTS (`alignment=True`) | run — produces `tts_audio/c-86ab861/<title>.wav` |
| 7–8 | Compare WAV durations | run |
| 9–10 | List `pipeline_data/speakers/` content | run |
| 11 | Task 1: study existing Chatterbox client | **read `api/src/services/tts_engine.py` instead of `tts.py`** |
| 12–13 | Cell 12 will fail (`tts.py` not at root) | skip cell 12, run cell 13 |
| 14–18 | Task 2: TDD for `resolve_speaker_wav` | run failing tests, then implement |
| 19 | Stub generator for `foreign_whispers/voice_resolution.py` | run once, then edit the file |
| 20–22 | Re-run tests, commit | run |
| 23–28 | Task 3: thread `speaker_wav` through API + service + engine | edit + restart |
| 29 | Cell to test `?speaker_wav=es/default.wav` end-to-end | run after Task 3 |
| 30 | Commit reminder | — |
| 31–37 | Task 4: per-speaker voice assignment | edit + restart |

## The four graded tasks

### Task 1 — Read the existing Chatterbox client

No code. Skim `api/src/services/tts_engine.py:34-125`. Notice:
- `ChatterboxClient.tts_to_file(text, file_path, **kwargs)` accepts a `speaker_wav` kwarg.
- `_synthesize_with_voice` already POSTs to `/v1/audio/speech/upload` with the reference WAV.
- The `speaker_wav` path is resolved relative to `pipeline_data/speakers/`.
- **Our local Chatterbox server** (`tools/chatterbox_server/serve.py`) implements the same `/v1/audio/speech/upload` contract, so voice cloning will work bare-metal on Mac.

### Task 2 — `resolve_speaker_wav(speakers_dir, target_language, speaker_id)`

**File:** create `foreign_whispers/voice_resolution.py`. Cell 19 writes a stub for you on first run.

Resolution chain (defined by tests in cell 18):
1. `speakers/{lang}/{speaker_id}.wav` if both present
2. `speakers/{lang}/default.wav`
3. `speakers/default.wav`

Returns the *relative* path string (e.g. `"es/SPEAKER_00.wav"`) — the Chatterbox client joins it onto `pipeline_data/speakers/`. 5 tests; passes with ~10 lines.

### Task 3 — Thread `speaker_wav` through the API

Three files:
- `api/src/core/config.py` — add a `speakers_dir` property (cell 28 gives the snippet).
- `api/src/routers/tts.py` — add `speaker_wav: str = Query(None)`. If `None`, resolve via `resolve_speaker_wav(settings.speakers_dir, "es")`.
- `api/src/services/tts_service.py` — accept `speaker_wav` and forward to `tts_engine.text_file_to_speech`.
- `api/src/services/tts_engine.py` — `text_file_to_speech` must pass `speaker_wav` to `ChatterboxClient.tts_to_file()` calls. Find the existing `_synthesize_raw` and the synth loop, propagate the kwarg.

After this you can curl:
```bash
curl -X POST "http://localhost:8080/api/tts/GYQ5yGV_-Oc?config=c-86ab861&alignment=true&speaker_wav=es/default.wav"
```

### Task 4 — Per-speaker voice mapping

**Prerequisite:** Notebook 4 Task 3 (segments have `speaker` field).

In `api/src/routers/tts.py`, before invoking the service:
```python
trans = json.loads((settings.translations_dir / f"{title}.json").read_text())
speakers = sorted({s.get("speaker", "SPEAKER_00") for s in trans.get("segments", [])})
voice_map = {sp: resolve_speaker_wav(settings.speakers_dir, "es", sp) for sp in speakers}
```

Then plumb `voice_map: dict[str, str]` down through `tts_service.text_file_to_speech` → `tts_engine.text_file_to_speech`. Inside `tts_engine`, the synth loop already iterates over `seg_metas`. For each segment, look up `voice_map[seg.speaker]` and pass it as `speaker_wav` to `_synthesize_raw` / `tts_to_file`.

Open-ended part: how do you handle an unknown `SPEAKER_NN` you don't have a WAV for? Round-robin through what you have? Or fall back to `default.wav`? Document the choice in the markdown of cell 41 (notebook 4) or in code comments.

## Mac/MPS-specific notes

- **The notebook's `!docker compose build` cells (cell 30, 36) are for the Docker setup.** On Mac bare-metal, restart by killing uvicorn:
  ```bash
  pkill -f "uvicorn api.src.main"
  FW_WHISPER_MODEL=base uv run uvicorn api.src.main:app --host 0.0.0.0 --port 8080 > /tmp/fw-api.log 2>&1 &
  ```
- **Chatterbox server first-call is slow.** Cold-start from kernel restart adds ~25s before the first synthesis. After that ~20 it/s on MPS.
- **Voice cloning works locally** — our `tools/chatterbox_server/serve.py` implements `/v1/audio/speech/upload` (Task 3 of cells we already implemented).
- **`speakers/default.wav`**: if it doesn't exist, you must create one before Task 4 fallback works. Pick any 5–15s WAV of a clean voice. Drop it at `pipeline_data/speakers/default.wav`.

## Cross-notebook connections

- **Notebook 4 Task 3 → this notebook Task 4.** Without speaker labels in the translation JSON, Task 4 has nothing to switch on.
- **Notebook 5 alignment work feeds the same TTS path.** When `alignment=True`, the engine uses the global aligner you improve in notebook 5. Better aligner → better TTS output through this same code.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Cell 12 `FileNotFoundError: tts.py` | Old notebook hint, file moved | Open `api/src/services/tts_engine.py` directly |
| `409 Conflict` on `/v1/audio/speech/upload` | Chatterbox server not running | Start it: `cd tools/chatterbox_server && uv run python serve.py` |
| Voice cloning ignored, default voice used | `speaker_wav` kwarg dropped somewhere | Add a `print()` in `_synthesize_with_voice` to confirm it's reached |
| `KeyError: 'speaker'` in voice_map lookup | Some segments lack speaker field | Use `voice_map.get(spk, default_voice_path)` |
