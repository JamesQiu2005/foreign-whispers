"""Duration-aware alignment data model and decision logic.

This module is the core of the ``foreign_whispers`` library.  It answers the
central question of the dubbing pipeline: *how do we fit a target-language
translation into the same time window as the original source-language speech?*

The module provides:

- ``SegmentMetrics`` — measures the timing mismatch for each segment.
- ``decide_action`` — per-segment policy that chooses accept / stretch / shift / retry / fail.
- ``global_align`` — greedy left-to-right pass that schedules all segments
  on a shared timeline, tracking cumulative drift from gap shifts.

No external dependencies — stdlib only.
"""
import dataclasses
import re
import unicodedata
from enum import Enum


def _count_syllables(text: str) -> int:
    """Count syllables in target-language text via vowel-cluster counting.

    Designed for Romance languages (Spanish, French, Italian, Portuguese).
    Strips accents then counts contiguous vowel runs. Each run = one syllable.
    Returns at least 1 for any non-empty text so the rate never divides by zero.
    """
    nfkd = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    clusters = re.findall(r"[aeiou]+", ascii_text)
    return max(1, len(clusters))


# ── Empirically tuned duration model ──────────────────────────────────────
# Calibrated against 70 ground-truth Chatterbox-MPS segments from the Strait
# of Hormuz dub (`pipeline_data/.../*.align.json`).
# Original heuristic (4.5 syll/s, no pause model): MAE 0.416s
# Tuned (5.40 syll/s + punctuation pauses):        MAE 0.281s (-32%)
# These constants can be re-fit per TTS engine; see notebooks/alignment_integration.
_SYLLABLE_RATE = 5.40         # syllables per second for Romance-language TTS
_COMMA_PAUSE_S = 0.15         # added per `,` `;` `:`
_TERMINAL_PAUSE_S = 0.30      # added per `.` `!` `?`
_UTTERANCE_OVERHEAD_S = 0.10  # constant onset/offset breath


def _count_punctuation_pause(text: str) -> float:
    """Sum of expected pause durations for punctuation in *text* (seconds)."""
    soft = len(re.findall(r"[,;:]", text)) * _COMMA_PAUSE_S
    hard = len(re.findall(r"[.!?]", text)) * _TERMINAL_PAUSE_S
    return soft + hard


def _estimate_duration(text: str) -> float:
    """Estimate TTS duration in seconds.

    Combines a syllable-rate model with punctuation pause overhead and a
    small fixed utterance onset/offset cost. Calibrated against
    Chatterbox-MPS ground truth — see module-level constants.
    """
    if not text or not text.strip():
        return 0.0
    syllable_seconds = _count_syllables(text) / _SYLLABLE_RATE
    return _UTTERANCE_OVERHEAD_S + syllable_seconds + _count_punctuation_pause(text)


@dataclasses.dataclass
class SegmentMetrics:
    """Timing measurements for one source/target transcript segment pair.

    For each segment we know the original source-language duration (from Whisper
    timestamps) and the translated target-language text.  The question is:
    *will the target-language TTS audio fit inside the source time window?*

    We estimate the TTS duration using a syllable-rate heuristic
    (~4.5 syllables/second for Romance languages) and derive three key numbers:

    Attributes:
        index: Zero-based segment position in the transcript.
        source_start: Source-language segment start time (seconds).
        source_end: Source-language segment end time (seconds).
        source_duration_s: ``source_end - source_start``.
        source_text: Original source-language text.
        translated_text: Target-language translation.
        src_char_count: Character count of the source text.
        tgt_char_count: Character count of the target text.
        predicted_tts_s: Estimated TTS duration (syllables / 4.5).
        predicted_stretch: Ratio ``predicted_tts_s / source_duration_s``.
            A value of 1.3 means the target-language audio is predicted to be
            30% longer than the available window.
        overflow_s: How many seconds the target-language audio exceeds the
            window (zero when it fits).
    """
    index:             int
    source_start:      float
    source_end:        float
    source_duration_s: float
    source_text:       str
    translated_text:   str
    src_char_count:    int
    tgt_char_count:    int
    predicted_tts_s:   float = dataclasses.field(init=False)
    predicted_stretch: float = dataclasses.field(init=False)
    overflow_s:        float = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.predicted_tts_s = _estimate_duration(self.translated_text)
        self.predicted_stretch = (
            self.predicted_tts_s / self.source_duration_s
            if self.source_duration_s > 0 else 1.0
        )
        self.overflow_s = max(0.0, self.predicted_tts_s - self.source_duration_s)


class AlignAction(str, Enum):
    """Decision outcomes for the per-segment alignment policy.

    Each segment gets exactly one action based on its ``predicted_stretch``:

    - ``ACCEPT`` — fits within 10% of the original duration, no change needed.
    - ``MILD_STRETCH`` — 10–40% over; apply pyrubberband time-stretch.
    - ``GAP_SHIFT`` — 40–80% over but adjacent silence can absorb the overflow.
    - ``REQUEST_SHORTER`` — 80–150% over; needs a shorter translation (P8).
    - ``FAIL`` — >150% over; no fix available, log and fall back to silence.
    """
    ACCEPT          = "accept"
    MILD_STRETCH    = "mild_stretch"
    GAP_SHIFT       = "gap_shift"
    REQUEST_SHORTER = "request_shorter"
    FAIL            = "fail"


@dataclasses.dataclass
class AlignedSegment:
    """A segment with its scheduled position on the global timeline.

    Produced by ``global_align``.  The ``scheduled_start`` and
    ``scheduled_end`` incorporate cumulative drift from earlier gap shifts,
    so they may differ from the original Whisper timestamps.

    Attributes:
        index: Segment position (matches ``SegmentMetrics.index``).
        original_start: Whisper start time (seconds).
        original_end: Whisper end time (seconds).
        scheduled_start: Start time after global alignment (seconds).
        scheduled_end: End time after global alignment (seconds).
        text: Target-language translated text for this segment.
        action: The ``AlignAction`` chosen by ``decide_action``.
        gap_shift_s: Seconds borrowed from adjacent silence (0.0 if none).
        stretch_factor: Speed factor for pyrubberband (1.0 = no stretch).
    """
    index:           int
    original_start:  float
    original_end:    float
    scheduled_start: float
    scheduled_end:   float
    text:            str
    action:          AlignAction
    gap_shift_s:     float = 0.0
    stretch_factor:  float = 1.0


def decide_action(m: SegmentMetrics, available_gap_s: float = 0.0) -> AlignAction:
    """Choose the alignment action for a single segment.

    Maps the predicted stretch factor to one of five actions using fixed
    thresholds.  ``GAP_SHIFT`` additionally requires that enough silence
    follows the segment to absorb the overflow.

    Thresholds::

        predicted_stretch   Action            Condition
        ─────────────────   ────────────────  ─────────────────────────
        <= 1.1              ACCEPT            fits naturally
        1.1 – 1.4          MILD_STRETCH      pyrubberband safe range
        1.4 – 1.8          GAP_SHIFT         only if gap >= overflow
        1.8 – 2.5          REQUEST_SHORTER   needs shorter translation
        > 2.5              FAIL              unfixable

    Args:
        m: Timing metrics for one segment.
        available_gap_s: Silence duration (seconds) after this segment,
            from VAD.  Defaults to 0.0 (no gap available).

    Returns:
        The ``AlignAction`` to apply.
    """
    sf = m.predicted_stretch
    if sf <= 1.1:
        return AlignAction.ACCEPT
    if sf <= 1.4:
        return AlignAction.MILD_STRETCH
    if sf <= 1.8 and available_gap_s >= m.overflow_s:
        return AlignAction.GAP_SHIFT
    if sf <= 2.5:
        return AlignAction.REQUEST_SHORTER
    return AlignAction.FAIL


def compute_segment_metrics(
    en_transcript: dict,
    es_transcript: dict,
) -> list[SegmentMetrics]:
    """Pair source and target segments and compute per-segment timing metrics.

    Zips the ``"segments"`` lists from both transcripts positionally
    (segment 0 ↔ segment 0, etc.) and builds a ``SegmentMetrics`` for each
    pair.  The source segment provides the time window; the target segment
    provides the text whose TTS duration we need to predict.

    Args:
        en_transcript: Source-language Whisper output dict with
            ``{"segments": [{"start", "end", "text"}, ...]}``.
        es_transcript: Target-language translation dict with the same structure.

    Returns:
        List of ``SegmentMetrics``, one per paired segment.  If the transcripts
        have different lengths, the shorter one determines the output length.
    """
    metrics = []
    for i, (en_seg, es_seg) in enumerate(
        zip(en_transcript.get("segments", []), es_transcript.get("segments", []))
    ):
        src_text = en_seg["text"].strip()
        tgt_text = es_seg["text"].strip()
        metrics.append(SegmentMetrics(
            index             = i,
            source_start      = en_seg["start"],
            source_end        = en_seg["end"],
            source_duration_s = en_seg["end"] - en_seg["start"],
            source_text       = src_text,
            translated_text   = tgt_text,
            src_char_count    = len(src_text),
            tgt_char_count    = len(tgt_text),
        ))
    return metrics


def global_align(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
) -> list[AlignedSegment]:
    """Greedy left-to-right global alignment of dubbed segments.

    Segments are timed independently by ``decide_action`` (P7), but they are
    sequential — if segment 5 borrows 0.3s from a silence gap, every segment
    after it shifts by 0.3s.  This function tracks that cumulative drift.

    Algorithm (single pass, O(n)):

    1. For each segment, call ``decide_action(m, available_gap_s)`` where
       *available_gap_s* comes from VAD silence regions after this segment.
    2. Based on the action:

       - ``GAP_SHIFT`` — the segment expands into the silence after it
         (``gap_shift = overflow_s``).
       - ``MILD_STRETCH`` — time-stretch capped at *max_stretch* (default 1.4x).
       - ``ACCEPT``, ``REQUEST_SHORTER``, ``FAIL`` — no modification.

    3. Schedule the segment with cumulative drift applied::

           scheduled_start = original_start + cumulative_drift
           scheduled_end   = scheduled_start + original_duration + gap_shift

    4. Every ``gap_shift`` adds to *cumulative_drift*, pushing all subsequent
       segments forward.

    Limitations:

    - **Greedy** — never looks ahead.  If segment 10 has a huge overflow and
      segment 9 has a large silence gap, it will not save that gap for
      segment 10.
    - **No backtracking** — once a decision is made, it is final.
    - A dynamic-programming or constraint-solver approach would produce
      better schedules, but this is the baseline to start from.

    Args:
        metrics: Per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output — list of ``{"start_s", "end_s", "label"}``
            dicts.  Pass ``[]`` if VAD is unavailable (gap_shift disabled).
        max_stretch: Upper bound for ``MILD_STRETCH`` speed factor.

    Returns:
        One ``AlignedSegment`` per input metric, in order.
    """
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    aligned, cumulative_drift = [], 0.0

    for m in metrics:
        action    = decide_action(m, available_gap_s=_silence_after(m.source_end))
        gap_shift = 0.0
        stretch   = 1.0

        if action == AlignAction.GAP_SHIFT:
            gap_shift = m.overflow_s
        elif action == AlignAction.MILD_STRETCH:
            stretch = min(m.predicted_stretch, max_stretch)
        # ACCEPT, REQUEST_SHORTER, FAIL → stretch stays at 1.0

        sched_start = m.source_start + cumulative_drift
        sched_end   = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))

        cumulative_drift += gap_shift

    return aligned


def _stretch_penalty(stretch: float) -> float:
    """Convex penalty for stretching: 0 at 1.0x, rises sharply outside [0.85, 1.25]."""
    if stretch <= 0:
        return 1e6  # silence — heavily penalized
    deviation = abs(stretch - 1.0)
    if deviation <= 0.10:
        return 0.0
    if deviation <= 0.25:
        return (deviation - 0.10) ** 2
    return (deviation - 0.10) ** 2 + 2.0 * (deviation - 0.25)


def global_align_dp(
    metrics:         list[SegmentMetrics],
    silence_regions: list[dict],
    max_stretch:     float = 1.4,
    drift_quantum_s: float = 0.1,
    drift_window_s:  float = 5.0,
) -> list[AlignedSegment]:
    """Dynamic-programming global alignment.

    Beats the greedy ``global_align`` by considering the future cost of
    consuming silence early. State is ``(segment_index, cumulative_drift)``,
    transitions are the available alignment actions for that segment, and the
    objective is the total stretch penalty plus a drift regularizer.

    Algorithm:

    1. Discretize cumulative drift into buckets of *drift_quantum_s* seconds,
       within ``[-drift_window_s, +drift_window_s]``.
    2. For each segment, enumerate candidate actions:

       - ``ACCEPT``: stretch=1.0 if predicted_stretch <= 1.1
       - ``MILD_STRETCH``: clamp predicted_stretch into [1/max_stretch, max_stretch]
       - ``GAP_SHIFT``: borrow from following silence (changes drift)
       - ``REQUEST_SHORTER`` / ``FAIL``: stretch=1.0, no drift change

    3. Solve via DP: ``dp[i][drift] = min over actions of (action_cost +
       dp[i+1][new_drift])``. Pseudo-polynomial in number of drift buckets.

    Complexity: O(n * D * A) where D = 2 * drift_window_s / drift_quantum_s,
    A ≈ 4. For 200 segments and 100 buckets, ~80k transitions. Sub-second.

    Args:
        metrics: per-segment timing metrics from ``compute_segment_metrics``.
        silence_regions: VAD output (or ``[]`` to disable gap-shift).
        max_stretch: ceiling for ``MILD_STRETCH``.
        drift_quantum_s: drift bucket size (smaller = more accurate, slower).
        drift_window_s: max allowed |cumulative drift|.

    Returns:
        One ``AlignedSegment`` per input metric, in order. Falls back to
        ``global_align`` if metrics is empty.
    """
    if not metrics:
        return []

    # ── Pre-compute available silence after each segment (in seconds) ─────
    def _silence_after(end_s: float) -> float:
        for r in silence_regions:
            if r.get("label") == "silence" and r["start_s"] >= end_s - 0.1:
                return r["end_s"] - r["start_s"]
        return 0.0

    gaps = [_silence_after(m.source_end) for m in metrics]

    # ── Drift bucket discretization ───────────────────────────────────────
    n_buckets = max(1, int(2 * drift_window_s / drift_quantum_s) + 1)
    zero_bucket = n_buckets // 2

    def _bucket_to_drift(b: int) -> float:
        return (b - zero_bucket) * drift_quantum_s

    def _drift_to_bucket(d: float) -> int:
        return max(0, min(n_buckets - 1, int(round(d / drift_quantum_s)) + zero_bucket))

    # ── Enumerate (action, gap_shift, stretch, drift_delta) per segment ───
    Action = tuple[AlignAction, float, float, float]

    def _candidates(m: SegmentMetrics, gap: float) -> list[Action]:
        cands: list[Action] = []
        sf = m.predicted_stretch
        # Stretch (always available)
        clamped = max(1.0 / max_stretch, min(max_stretch, sf))
        if clamped == 1.0 or sf <= 1.1:
            cands.append((AlignAction.ACCEPT, 0.0, 1.0, 0.0))
        else:
            cands.append((AlignAction.MILD_STRETCH, 0.0, clamped, 0.0))
        # Gap shift — only if surrounding silence covers the overflow
        if 1.1 < sf <= 1.8 and gap >= m.overflow_s and m.overflow_s > 0:
            cands.append((AlignAction.GAP_SHIFT, m.overflow_s, 1.0, m.overflow_s))
        # Last resort
        if sf > 2.5:
            cands.append((AlignAction.FAIL, 0.0, 1.0, 0.0))
        elif sf > 1.8 and clamped == max_stretch:
            cands.append((AlignAction.REQUEST_SHORTER, 0.0, 1.0, 0.0))
        # De-duplicate by stretch (avoid identical-action ties)
        seen: set[float] = set()
        out: list[Action] = []
        for c in cands:
            key = (c[0].value, round(c[2], 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _action_cost(action: AlignAction, stretch: float, drift_after: float) -> float:
        cost = _stretch_penalty(stretch)
        # Severity multipliers for non-stretch actions
        if action == AlignAction.REQUEST_SHORTER:
            cost += 1.5
        elif action == AlignAction.FAIL:
            cost += 5.0
        # Drift regularizer — keep cumulative drift near zero (gentle).
        cost += 0.2 * abs(drift_after)
        return cost

    # ── Backward DP: dp[i][b] = min total cost from segment i onward ──────
    n = len(metrics)
    INF = float("inf")
    dp = [[INF] * n_buckets for _ in range(n + 1)]
    choice = [[None] * n_buckets for _ in range(n)]
    dp[n] = [0.0] * n_buckets

    for i in range(n - 1, -1, -1):
        cands = _candidates(metrics[i], gaps[i])
        for b in range(n_buckets):
            drift_now = _bucket_to_drift(b)
            best_cost = INF
            best_choice = None
            for action, gap_shift, stretch, drift_delta in cands:
                drift_after = drift_now + drift_delta
                if abs(drift_after) > drift_window_s:
                    continue
                b_after = _drift_to_bucket(drift_after)
                cost = _action_cost(action, stretch, drift_after) + dp[i + 1][b_after]
                if cost < best_cost:
                    best_cost = cost
                    best_choice = (action, gap_shift, stretch, b_after)
            dp[i][b] = best_cost
            choice[i][b] = best_choice

    # ── Forward reconstruct from drift = 0 ────────────────────────────────
    aligned: list[AlignedSegment] = []
    b = zero_bucket
    cumulative_drift = 0.0
    for i, m in enumerate(metrics):
        ch = choice[i][b]
        if ch is None:
            # Should not happen — fall back to ACCEPT
            action, gap_shift, stretch, b_next = AlignAction.ACCEPT, 0.0, 1.0, b
        else:
            action, gap_shift, stretch, b_next = ch

        sched_start = m.source_start + cumulative_drift
        sched_end = sched_start + m.source_duration_s + gap_shift

        aligned.append(AlignedSegment(
            index           = m.index,
            original_start  = m.source_start,
            original_end    = m.source_end,
            scheduled_start = sched_start,
            scheduled_end   = sched_end,
            text            = m.translated_text,
            action          = action,
            gap_shift_s     = gap_shift,
            stretch_factor  = stretch,
        ))
        cumulative_drift += gap_shift
        b = b_next

    return aligned
