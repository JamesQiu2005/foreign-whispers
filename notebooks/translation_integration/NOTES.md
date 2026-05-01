# Notebook 3 — `translation_integration`

## What it teaches

The translator (`argostranslate`) is *unconstrained*: it doesn't know how long the source speaker had to say something, so Spanish translations routinely don't fit the original time window. This is the root cause of most dubbing artifacts. **One coded task: implement `get_shorter_translations()`.**

```
en_segments.json ─► argostranslate (offline, OpenNMT) ─► es_segments.json
                                                          (no duration budget!)
```

## Cell-by-cell

| Cell | What it does | Why it matters |
|---|---|---|
| 0–2 | Markdown + setup | Standard boilerplate. |
| 3 | Markdown — translate | — |
| 4 | `fw.translate(video_id)`; print first 3 EN→ES pairs | Result is cached at `translations/argos/<title>.json`. |
| 5 | Markdown — length analysis | — |
| 6 | Histogram of `len(es_text) / len(en_text)` per segment | Spanish text is typically 1.10–1.30× the English. **Important caveat: this is character ratio, not duration ratio.** Phonetic density matters more for time. |
| 7 | Markdown — Task: `get_shorter_translations()` | — |
| 8 | Calls the stub (currently returns `[]`) | Demonstrates what your implementation should produce. |
| 9 | Summary | — |

## Where the work happens

- Server-side translate: `api/src/routers/translate.py` → `api/src/services/translation_service.py` → `api/src/services/translation_engine.py`. The engine wraps `argostranslate.translate.translate()`.
- **Your code** lives in `foreign_whispers/reranking.py`. Read the existing module first to see the `TranslationCandidate` dataclass and the stub's docstring.

## The graded task

**Function:** `get_shorter_translations(source_text, baseline_es, target_duration_s) -> list[TranslationCandidate]`

**Contract** (per the docstring in `reranking.py`):
- Return 0 or more `TranslationCandidate` objects, each with `text` (str), `char_count` (int), `brevity_rationale` (str).
- Each candidate must be *shorter* than the baseline (smaller `char_count`).
- Each candidate should plausibly fit within `target_duration_s` at the duration heuristic (~15 chars/s for Spanish → `~target_duration_s * 15` chars max).
- Order: shortest first (or score-sorted — caller takes `[0]`).

**Suggested implementation strategies** (from least to most ambitious):

1. **Rule-based truncation.** Drop discourse markers ("Bueno,", "Pues,", "Entonces,"), cut redundant adjectives, prefer shorter synonyms (`comenzar` → `empezar`). Cheapest. Often produces ~10–15% length reduction.
2. **Multiple translator backends.** Run argostranslate, then optionally a different model (e.g. `googletrans` if you have internet, or a HF translation model). Pick the shortest output.
3. **LLM-based summarization.** Call a local Llama model with a prompt like *"Rewrite this Spanish sentence to be at most N characters while preserving meaning."*. Best quality, but slow and depends on having an LLM available.
4. **Hybrid.** Use rule-based as the cheap default; fall back to LLM only for hard cases.

For coursework, **strategy 1 is the path of least resistance** and gets you a working implementation in ~30 minutes. Strategies 3/4 are research-grade extensions.

## How this connects to other notebooks

- **`alignment_integration` Task 2** is *the same code task* viewed from the alignment side: a segment with `decide_action() == REQUEST_SHORTER` calls `get_shorter_translations()` to fix itself. Implementing it here directly improves alignment metrics there.
- **`alignment_integration` Task 1** (better duration prediction) tells you which segments need shorter translations in the first place.

## Mac/MPS-specific notes

- argostranslate is **CPU-only**, so this notebook is fast on any machine.
- First call downloads the EN→ES language package (~500 MB). It caches it under `~/.local/share/argos-translate/`.

## Tasks (graded)

**1 task.** Implement `get_shorter_translations` in `foreign_whispers/reranking.py`.

Validation:
1. Re-run cell 8 — should print at least one candidate, not the empty-list message.
2. Re-run the alignment notebook later (`alignment_integration` Task 2 cell) — `REQUEST_SHORTER` segment count should drop after your re-ranker is wired in.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Cell 4 hangs on first call | argostranslate downloading EN→ES package | Wait — only happens once |
| `ImportError: get_shorter_translations` | Stub not yet exported | Check `foreign_whispers/__init__.py` exports the symbol |
| Candidates are *longer* than baseline | Logic bug in your filter | Add an assertion: `assert all(c.char_count < len(baseline_es) for c in candidates)` |
