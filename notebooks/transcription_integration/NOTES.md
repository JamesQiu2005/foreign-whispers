# Notebook 2 — `transcription_integration`

## What it teaches

How the same `transcribe` step has two backends, a "slow but accurate" one and a "fast but coarse" one, and how to pick between them. **No code to write — read-only.**

```
video.mp4 ──┬─► YouTube captions (default, fast)   ─► transcriptions/whisper/<title>.json
            └─► Whisper STT (use_youtube_captions=False)
```

## Cell-by-cell

| Cell | What it does | Why it matters |
|---|---|---|
| 0–2 | Markdown + setup (kernel, `.env`, `FWClient`, video pick) | Same boilerplate as notebook 1. |
| 3 | Markdown — modes | — |
| 4 | `fw.transcribe(video_id)` | Default = YouTube captions. Output cached as JSON. Returns `{language, segments, skipped}`. |
| 5 | Markdown — why force Whisper | — |
| 6 | `requests.post(/api/transcribe/{id}, params={use_youtube_captions: False})` | Bypasses YT captions and runs the actual Whisper model. **Slow on Mac MPS**: a 13-min video on `whisper-base` takes ~3–5 min. |
| 7 | Markdown — comparison | — |
| 8 | Histogram of segment durations, YT vs Whisper | Whisper segments are usually more uniform; YT captions cluster around chunked-line lengths (often forced into 2-line cues). |
| 9 | Markdown — JSON schema | — |
| 10 | Pretty-print first 2 segments of one transcript | Confirms shape: `{id, start, end, text}` per segment. |
| 11 | Summary | — |

## Where the work happens (server-side)

`fw.transcribe()` → `POST /api/transcribe/{video_id}` (`api/src/routers/transcribe.py`). The router has a clever short-circuit at `transcribe.py:67-89`:

1. If a cached `transcriptions/whisper/<title>.json` exists → return it (`skipped=True`).
2. Else if `use_youtube_captions=True` and a YT caption file exists → convert the YT line-delimited JSON into Whisper-segment shape on the fly. **No model is loaded**.
3. Else → call `whisper.load_model(...)` and run STT on the video file.

So Whisper inference only runs when you explicitly disable captions, *and* there's no cached transcript.

## Mac/MPS-specific notes

- **First Whisper call is slow.** Loading `whisper-base` on first use takes ~30s; inference on a 13-min video takes a few minutes. The notebook caches the result, so subsequent runs return instantly.
- **Force-rerun**: delete `pipeline_data/api/transcriptions/whisper/<title>.json` before calling cell 6.
- The `use_youtube_captions=False` branch will use the local Whisper Python package (CPU + MPS for some ops). It's not as fast as a CUDA box but it works.

## What to look at while running

- **Segment count comparison** in cell 8: typically YT < Whisper. The Strait of Hormuz video should have ~170 YT cues, and Whisper-base will produce more (~200+) shorter segments.
- **Why this matters for Phase 2 alignment**: a longer source segment leaves more room to absorb a longer Spanish translation, while a shorter source segment forces a higher stretch ratio. So the choice between YT captions and Whisper directly affects how aggressive the alignment work has to be.
- **First 2 segments JSON in cell 10**: note the absence of word-level timestamps. Whisper *can* produce them but this pipeline doesn't request them.

## Tasks (graded)

**None.** Move on to `translation_integration` once cells 4–10 run cleanly.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Cell 6 hangs > 5 minutes | `whisper-base` running on CPU and your video is long | Wait, or set `FW_WHISPER_MODEL=tiny` and restart the API |
| `KeyError: 'segments'` in cell 6 | API returned an error JSON instead of the schema | Check the API log: `tail /tmp/fw-api.log` |
| Histogram empty for one side | One transcription mode wasn't run yet | Run both cells 4 and 6 first |
