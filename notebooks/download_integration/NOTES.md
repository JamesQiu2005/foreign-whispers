# Notebook 1 — `download_integration`

## What it teaches

The first stage of the pipeline: how a YouTube URL becomes two artifacts on
disk that everything downstream depends on. **No code to write — read-only.**

```
YouTube URL ──► fw.download() ──► pipeline_data/api/videos/<title>.mp4
                                  pipeline_data/api/youtube_captions/<title>.txt
```

## Cell-by-cell

| Cell | What it does | Why it matters |
|---|---|---|
| 0 | Markdown intro | — |
| 1 | "Setup" header | — |
| 2 | `sys.path` shim + load `.env` + Logfire shim | Lets the notebook import `foreign_whispers` from the repo. The Logfire fallback is a no-op so the notebook still works without an observability token. |
| 3 | `FWClient("http://localhost:8080").healthz()` | First API call. If this raises, the API isn't running — start it (see `notebooks/README.md`). |
| 4 | Markdown header | — |
| 5 | `fw.download(VIDEO_URL)` for the Strait of Hormuz video | The actual download. Returns `{"video_id", "title", "caption_segments"}`. The video file and caption JSON are written server-side, not returned in the response. |
| 6 | Markdown explaining artifacts | — |
| 7 | List `videos/` and `youtube_captions/` directories | Confirms files landed on disk. |
| 8 | Markdown — caption timeline | — |
| 9 | Matplotlib horizontal bar chart of caption segments | Each bar = one caption segment, x-axis = time. Useful sanity check that captions span the whole video and roughly match speech density. |
| 10 | Summary markdown | — |

## Where the work actually happens (server-side)

`fw.download()` hits `POST /api/download`, handled by:

- `api/src/routers/download.py` — thin route wrapper
- `api/src/services/download_engine.py` — calls **yt-dlp** for video + auto-captions

**On the Mac bare-metal setup**, yt-dlp may hit YouTube's bot check. If you
see `Sign in to confirm you're not a bot`, options:

1. Drop a `cookies.txt` (Netscape format from a browser extension) into the
   repo root. The download engine picks it up automatically.
2. Manually `yt-dlp <url>` and copy the `.mp4` into
   `pipeline_data/api/videos/<title>.mp4` — the rest of the notebook still
   works because it just lists what's on disk.

## Things to look at as you run it

- **Compare `caption_segments` count vs. video length.** This 60-Minutes
  segment is ~13 minutes. ~150–200 caption segments is normal.
- **Look at the first 5 segments printed by cell 5.** Notice the start
  times — the first ~10 seconds of the video is intro music with no
  captions. That offset matters in the alignment notebook later.
- **Caption timeline plot** — gaps in the bars are silence/music, not
  bugs. Phase 2 alignment uses these gaps as places to absorb stretching.

## Tasks (graded)

**None.** Move on to `transcription_integration` once the cells run end-to-end.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConnectionRefusedError` on cell 3 | API not running | Start uvicorn (see `notebooks/README.md`) |
| `Sign in to confirm you're not a bot` | YouTube anti-bot on cell 5 | Add `cookies.txt`, or manually download the MP4 |
| `ModuleNotFoundError: dotenv` | `python-dotenv` missing | Already in our `uv.lock`; if missing, `uv add python-dotenv` |
| Kernel not found in VS Code | Kernel not registered | `uv run python -m ipykernel install --user --name foreign-whispers` |
