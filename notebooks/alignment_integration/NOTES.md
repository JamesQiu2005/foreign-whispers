# Notebook 5 — `alignment_integration`

## What it teaches

Where dubbing actually breaks: making the Spanish TTS *fit the time the speaker had to say it*. A 3-second English phrase translated to 5 seconds of Spanish becomes a wobbly time-stretched mess. This notebook frames timing as an optimization problem and walks you through four progressively richer attempts at solving it. **Pure-Python, no servers needed.**

## Why this is your highest-leverage notebook (on Mac)

- **No GPU dependency.** Runs entirely on CPU. Iteration is fast.
- **All four tasks are *measurable*.** Every change has a numeric scorecard, so you can A/B easily.
- **It's the dominant artifact in your dub.** From the `.align.json` we read earlier, 37% of segments hit the slow-stretch clamp on the Coqui run; even with Chatterbox-MPS, length mismatches drive most of what you'll hear.

## Cell-by-cell map

| Cell | What | Code or read? |
|---|---|---|
| 0–2 | Setup | run |
| 3–4 | Load cached `en_transcript` and `es_transcript` from disk (no API needed) | run |
| 5–6 | `compute_segment_metrics`, list worst 5 stretch offenders | run |
| 7–8 | Color-coded histogram of stretch factors by `AlignAction` | run |
| 9 | **Task 1**: improve `_estimate_duration` | edit `foreign_whispers/alignment.py` |
| 10 | Baseline measurement of current heuristic vs ground-truth `.align.json` | run |
| 11–12 | Action-distribution table + bar chart | run |
| 13 | **Task 2**: implement `get_shorter_translations` (same as notebook 3) | edit `foreign_whispers/reranking.py` |
| 14 | Identify segments needing shorter translations | run |
| 15–18 | Greedy `global_align` + visualization of original vs scheduled timing | run |
| 19 | **Task 3**: write a `global_align_dp` to beat the greedy optimizer | edit `foreign_whispers/alignment.py` |
| 20 | Print greedy baseline numbers — your DP must beat these | run |
| 21 | **Task 4**: design a multi-dimensional dubbing scorecard | edit `foreign_whispers/evaluation.py` |
| 22 | Summary table of all four tasks | — |

## The four graded tasks

### Task 1 — Better duration prediction

**File:** `foreign_whispers/alignment.py` — modify `_estimate_duration` (recently extracted into its own helper per the latest commit).

The current heuristic is `len(text) / 15` (chars/sec). It's wrong for short fragments, punctuation-heavy text, and Spanish words with high syllable density.

Better predictors, in order of cost:

1. **Syllable counter.** Use a Spanish syllabifier (rule-based: count vowel groups). Estimate at ~4.5 syllables/sec.
2. **Phoneme model.** `phonemizer` gives you IPA → count phones → ~13 phones/sec.
3. **Regression on observed TTS durations.** Run TTS on 30 segments, fit `linear_regression(features, actual_duration_s)` where features = `[char_count, syllable_count, comma_count, has_question_mark]`.
4. **Learned duration model.** A neural duration predictor — overkill for one assignment.

**Validation** (cell 10): mean absolute error vs `raw_duration_s` from `.align.json`. The current heuristic's error is your baseline to beat.

### Task 2 — Translation re-ranking

This is the **same** `get_shorter_translations` task as notebook 3. If you already implemented it there, this notebook just consumes it. If you skipped notebook 3, do it here.

**Validation** (cell 12 re-run): `REQUEST_SHORTER` and `FAIL` counts should drop. Aim for at least half of `REQUEST_SHORTER` segments moving to `MILD_STRETCH` or `ACCEPT`.

### Task 3 — Beat the greedy global aligner

**File:** add `global_align_dp(metrics, silence_regions)` next to the existing `global_align()`.

The greedy algorithm picks locally-best moves. It can starve a later segment by greedily borrowing silence early. Three reasonable approaches:

1. **DP**: state = `(segment_index, cumulative_drift_quantum)`, transitions = stretch / shift / shorter / fail, minimize sum-of-penalties. Pseudo-polynomial in drift quantization.
2. **ILP with PuLP** (≤ 200 segments → solves in seconds): variables are per-segment `start_time`; constraints enforce monotone non-overlap and stretch bounds; objective minimizes `Σ severe_stretch_penalty`.
3. **Beam search**: keep top-K trajectories by cumulative penalty.

**Validation** (cell 20): re-run `clip_evaluation_report(metrics, your_aligned)` and compare against the printed greedy baseline. Both `total_cumulative_drift_s` and the count of severe stretches should improve.

### Task 4 — Dubbing quality scorecard

**File:** `foreign_whispers/evaluation.py` — add `dubbing_scorecard(metrics, aligned, align_report)` returning a dict of normalized scores.

Suggested dimensions:
- **Timing accuracy**: function of `mean_abs_duration_error_s`, `pct_severe_stretch`, drift.
- **Intelligibility (round-trip)**: TTS the Spanish, then run Whisper STT on the result, compute WER vs the original Spanish text.
- **Semantic fidelity**: `cosine(embed(en_text), embed(back_translate(es_text)))`. Use sentence-transformers (CPU-fast).
- **Naturalness**: variance of `speed_factor` across segments — high variance = jarring pace shifts.

This is the most open-ended task and benefits from a radar chart at the end (cell stays as a stub).

## Mac/MPS-specific notes

- **No services needed.** Just the kernel + `pipeline_data/` content.
- **Round-trip STT for Task 4** can use `openai-whisper` directly (already installed). It's slow on Mac for hundreds of round-trips — sample 20 segments, not all 170.
- **Sentence embeddings** (`sentence-transformers`) run fine on CPU/MPS.
- **PuLP** for ILP is in PyPI and pure-Python solver-friendly. `uv add pulp` if you go that route.

## Cross-notebook connections

- **Task 1 directly improves Task 3.** A better duration predictor changes which segments enter the optimizer in `REQUEST_SHORTER` state.
- **Task 4 quantifies improvements from notebooks 3, 5, and 6.** Run the scorecard once before any changes, again after notebook 3's reranker, again after Task 3's DP optimizer. The deltas are your evidence of impact.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Cell 4 `assert en_files` fails | No transcripts on disk yet | Run notebooks 1+2+3 first, or run the full `pipeline_end_to_end` once |
| Cell 10 prints "No raw_duration_s" | Never ran TTS at all | Run the TTS notebook (or `pipeline_end_to_end`) first |
| `ImportError: clip_evaluation_report` | Symbol not exported | Check `foreign_whispers/__init__.py` |
| DP optimizer slower than the whole notebook | Drift quantization too fine | Use 0.1s buckets; ~50 buckets is plenty |
