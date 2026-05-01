"""Clip-level alignment quality metrics.

Imports from foreign_whispers.alignment — no other dependencies.
"""
import statistics as _stats

from foreign_whispers.alignment import (
    AlignAction,
    AlignedSegment,
    SegmentMetrics,
    decide_action,
)


# Per-action quality weights for the scorecard.
# Tuned so ACCEPT > MILD_STRETCH > GAP_SHIFT > REQUEST_SHORTER > FAIL.
_ACTION_WEIGHT = {
    AlignAction.ACCEPT:          1.00,
    AlignAction.MILD_STRETCH:    0.75,
    AlignAction.GAP_SHIFT:       0.70,
    AlignAction.REQUEST_SHORTER: 0.40,
    AlignAction.FAIL:            0.00,
}


def clip_evaluation_report(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
) -> dict:
    """Return a summary dict of alignment quality metrics for one clip.

    Keys:
        mean_abs_duration_error_s: Mean |predicted_tts_s - source_duration_s| per segment.
        pct_severe_stretch: % of aligned segments with stretch_factor > 1.4.
        n_gap_shifts: Number of segments resolved via gap-shift.
        n_translation_retries: Number of segments that required re-ranking.
        total_cumulative_drift_s: End-to-end drift introduced by gap-shifts.
    """
    if not metrics:
        return {
            "mean_abs_duration_error_s": 0.0,
            "pct_severe_stretch":        0.0,
            "n_gap_shifts":              0,
            "n_translation_retries":     0,
            "total_cumulative_drift_s":  0.0,
        }

    errors    = [abs(m.predicted_tts_s - m.source_duration_s) for m in metrics]
    n_severe  = sum(1 for a in aligned if a.stretch_factor > 1.4)
    n_shifted = sum(1 for a in aligned if a.action == AlignAction.GAP_SHIFT)
    n_retry   = sum(1 for m in metrics if decide_action(m) == AlignAction.REQUEST_SHORTER)
    drift     = (
        aligned[-1].scheduled_end - aligned[-1].original_end
        if aligned else 0.0
    )

    return {
        "mean_abs_duration_error_s": round(_stats.mean(errors), 3),
        "pct_severe_stretch":        round(100 * n_severe / max(len(metrics), 1), 1),
        "n_gap_shifts":              n_shifted,
        "n_translation_retries":     n_retry,
        "total_cumulative_drift_s":  round(drift, 3),
    }


def _normalize_inverse(value: float, scale: float) -> float:
    """Map [0, scale]+ → [1, 0]: 0 → 1.0, ≥ scale → 0.0."""
    if value <= 0:
        return 1.0
    return max(0.0, 1.0 - min(value, scale) / scale)


def dubbing_scorecard(
    metrics: list[SegmentMetrics],
    aligned: list[AlignedSegment],
    align_report: dict | None = None,
) -> dict:
    """Multi-dimensional dubbing quality scorecard.

    Each dimension is normalised to ``[0, 1]`` (1 = ideal, 0 = worst).
    The ``overall`` score is the unweighted mean of the four core dimensions.

    Dimensions:

    - **timing_accuracy**: how well ``predicted_tts_s`` matches the source
      window. Derived from mean absolute duration error.
    - **action_quality**: average ``_ACTION_WEIGHT`` across segments —
      penalizes REQUEST_SHORTER and FAIL outcomes.
    - **naturalness**: 1 - normalized(variance of stretch_factor). High
      variance = jarring pace shifts between segments.
    - **drift_control**: 1 - normalized(|cumulative drift|). Heavy drift
      (>10s) → 0; no drift → 1.

    The optional *align_report* lets the caller pass in a pre-computed
    ``clip_evaluation_report`` so we don't redo the work.

    Returns:
        Dict with the four normalized scores plus ``overall`` and a
        ``raw`` sub-dict carrying the underlying numbers for plotting.
    """
    if not metrics or not aligned:
        return {
            "overall":         0.0,
            "timing_accuracy": 0.0,
            "action_quality":  0.0,
            "naturalness":     0.0,
            "drift_control":   1.0,
            "raw": {},
        }

    report = align_report if align_report is not None else clip_evaluation_report(metrics, aligned)

    # 1. Timing accuracy — MAE of 0s ⇒ 1.0; MAE of 4s+ ⇒ 0.0.
    # Scale set so a typical un-shortened argostranslate output (MAE ~2s) lands
    # around 0.5, and a well-rerank/aligned clip lands above 0.7.
    mae = report["mean_abs_duration_error_s"]
    timing_accuracy = _normalize_inverse(mae, scale=4.0)

    # 2. Action quality — weighted average of per-segment outcomes.
    action_weights = [_ACTION_WEIGHT.get(a.action, 0.5) for a in aligned]
    action_quality = sum(action_weights) / len(action_weights)

    # 3. Naturalness — variance of stretch factors. var=0 ⇒ 1.0; var≥0.05 ⇒ 0.0.
    stretches = [a.stretch_factor for a in aligned]
    var_stretch = _stats.pvariance(stretches) if len(stretches) > 1 else 0.0
    naturalness = _normalize_inverse(var_stretch, scale=0.05)

    # 4. Drift control — |drift| of 0s ⇒ 1.0; ≥10s ⇒ 0.0.
    drift_abs = abs(report["total_cumulative_drift_s"])
    drift_control = _normalize_inverse(drift_abs, scale=10.0)

    overall = _stats.mean([timing_accuracy, action_quality, naturalness, drift_control])

    return {
        "overall":         round(overall, 3),
        "timing_accuracy": round(timing_accuracy, 3),
        "action_quality":  round(action_quality, 3),
        "naturalness":     round(naturalness, 3),
        "drift_control":   round(drift_control, 3),
        "raw": {
            "mean_abs_duration_error_s": mae,
            "var_stretch_factor":        round(var_stretch, 4),
            "total_cumulative_drift_s":  report["total_cumulative_drift_s"],
            "n_segments":                len(aligned),
            "action_breakdown":          {
                a.value: sum(1 for x in aligned if x.action == a)
                for a in AlignAction
            },
        },
    }
