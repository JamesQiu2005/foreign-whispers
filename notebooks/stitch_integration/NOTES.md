# Notebook 7 — `stitch_integration`

## What it teaches

The final assembly step: take the original video stream + the synthesized Spanish audio + the translated text, and produce one MP4 with a synthesized audio track and a sidecar VTT subtitle file. **No code to write — verification only.**

```
videos/<title>.mp4   ──┐
tts_audio/<cfg>/.wav ──┼─► ffmpeg remux ──► dubbed_videos/<cfg>/<title>.mp4
translations/...json ──┘                    dubbed_captions/<title>.vtt
```

Key design choice: **audio-only remux**. ffmpeg passes the video stream through with `-c:v copy` (no re-encoding, no quality loss, fast), and only swaps in the new audio track.

## Cell-by-cell

| Cell | What it does | Why it matters |
|---|---|---|
| 0–2 | Setup, `FWClient`, healthz | Standard. |
| 3 | Markdown — stitch | — |
| 4 | `fw.stitch(video_id)` | Calls `POST /api/stitch/{id}`. Returns `{video_id, video_path, config}`. The actual remux happens server-side. |
| 5 | Markdown — output paths | — |
| 6 | List `dubbed_videos/` and `dubbed_captions/` directories | Confirms files landed. The MP4 is roughly the same size as the original (audio is a small fraction of MP4 byte size). |
| 7 | Markdown — VTT viewer | — |
| 8 | Print first 30 lines of the VTT | Notice the **rolling two-line** pattern: each cue shows the current translated line + the previous one for continuity. Makes long-form interview content readable. |
| 9 | Markdown — playback options | — |
| 10 | Summary | — |

## Where the work happens (server-side)

`fw.stitch()` → `POST /api/stitch/{id}` (`api/src/routers/stitch.py`). The interesting bits are in `api/src/services/stitch_engine.py`:
- ffmpeg invocation: `ffmpeg -i video.mp4 -i tts.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest output.mp4`
- VTT writer: walks the translated segments, builds two-line rolling cues with `<v Speaker>` voice tags when speaker labels exist.

## What to look at while running

- **Compare the dubbed MP4 file size to the original.** Should be within ~1% — confirms `-c:v copy` did its job (no video re-encoding).
- **Open the VTT in a text editor.** The two-line format is visually distinctive: every cue has two `<v>` lines.
- **Play it.** Easiest:
  ```bash
  open pipeline_data/api/dubbed_videos/c-fb1074a/<title>.mp4
  ```
  This pops it open in QuickTime. Or use VLC if you want to load the VTT as an external subtitle track.

## Cross-notebook connections

- **Stitch is the integration test for everything else.** If your alignment work (notebook 5) is good, the audio sync should *feel* right when watching. If voice cloning (notebook 6) is wired, you'll hear the right speakers.
- **No graded extension here.** The course intentionally ends with verification — this is where you watch the result of the previous six notebooks.

## Mac/MPS-specific notes

- **ffmpeg on macOS** is what you installed via `brew install ffmpeg` early on. The stitch engine just shells out — no GPU involved.
- **Original-vs-dubbed playback in the frontend** at http://localhost:8501 lets you A/B with a click. Useful for spotting regressions.

## Tasks (graded)

**None.** This notebook is a smoke test for the rest of your work. Once it produces a watchable MP4, you've completed Phase 2.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Cell 4 `404` | TTS step never ran for this video/config | Run `tts_integration` first |
| Dubbed MP4 plays but has the original audio | Stage cache returned stale data | Delete `pipeline_data/api/dubbed_videos/<config>/<title>.mp4` and retry |
| VTT has no `<v>` voice tags | Translation segments lack `speaker` field | Diarization (notebook 4) wasn't run for this video |
| Audio out of sync | Either alignment was off, or the YouTube speech-offset wasn't applied | Re-run TTS with `alignment=true` |
