"""Deterministic failure analysis and translation re-ranking.

The failure analysis function uses simple threshold rules derived from
SegmentMetrics.  ``get_shorter_translations`` implements a rule-based
Spanish shortener: filler removal → short-synonym substitution →
phrase contraction.  Pure-Python, deterministic, no LLM dependency.
"""

import dataclasses
import logging
import re

logger = logging.getLogger(__name__)


# ── Spanish shortening tables ──────────────────────────────────────────────
#
# Discourse markers / fillers commonly added by argostranslate that carry
# little meaning and consume a lot of speech time.  Stripped at sentence
# starts and after commas where they do not change the proposition.
_DISCOURSE_MARKERS = (
    "Bueno,", "Pues,", "Entonces,", "Así que,", "Mira,", "Sabes,",
    "O sea,", "Es decir,", "Por supuesto,", "De hecho,", "Por cierto,",
    "Por lo tanto,", "Sin embargo,", "Por ejemplo,", "Es como,",
)

# Multi-word → shorter equivalent.  Each pair preserves meaning.
# Ordered longest-first so we match greedy-longest at substitution time.
_PHRASE_SHORTENINGS = [
    ("en este momento", "ahora"),
    ("en este instante", "ahora"),
    ("a partir de este momento", "desde ahora"),
    ("la mayor parte de", "casi todo"),
    ("la mayoría de los", "casi todos los"),
    ("la mayoría de las", "casi todas las"),
    ("a pesar de que", "aunque"),
    ("a pesar de", "pese a"),
    ("debido a que", "porque"),
    ("a causa de que", "porque"),
    ("a causa de", "por"),
    ("con el fin de", "para"),
    ("con el objeto de", "para"),
    ("con el propósito de", "para"),
    ("en virtud de", "por"),
    ("en relación con", "sobre"),
    ("en cuanto a", "sobre"),
    ("respecto a", "sobre"),
    ("acerca de", "sobre"),
    ("en torno a", "sobre"),
    ("por medio de", "mediante"),
    ("por el cual", "que"),
    ("la cual", "que"),
    ("el cual", "que"),
    ("hacer una pregunta", "preguntar"),
    ("dar una respuesta", "responder"),
    ("hacer un comentario", "comentar"),
    ("tomar una decisión", "decidir"),
    ("hacer mención", "mencionar"),
    ("llevar a cabo", "hacer"),
    ("dar comienzo a", "empezar"),
    ("hacer referencia a", "referirse a"),
    ("tener la posibilidad de", "poder"),
    ("no obstante", "pero"),
    ("a continuación", "después"),
    ("de forma inmediata", "ya"),
    ("de manera inmediata", "ya"),
    ("de igual manera", "igual"),
    ("de la misma forma", "igual"),
    ("hoy en día", "hoy"),
    ("una gran cantidad de", "muchos"),
    ("una serie de", "varios"),
    ("un montón de", "muchos"),
]

# Single-word longer→shorter near-synonyms.  Conservative — only swaps where
# both words share register (neutral / informational).
_WORD_SHORTENINGS = {
    "comenzar":     "empezar",
    "comienzo":     "inicio",
    "finalizar":    "acabar",
    "utilizar":     "usar",
    "utilización":  "uso",
    "necesitar":    "querer",   # only safe when context demands ≠ obligation
    "obtener":      "lograr",
    "encontrar":    "hallar",
    "demostrar":    "mostrar",
    "considerar":   "ver",
    "extremadamente": "muy",
    "completamente":  "del todo",
    "absolutamente":  "del todo",
    "actualmente":    "hoy",
    "frecuentemente": "a menudo",
    "habitualmente":  "siempre",
    "anteriormente":  "antes",
    "posteriormente": "luego",
    "generalmente":   "casi siempre",
    "verdaderamente": "de verdad",
    "ciertamente":    "sí",
    "evidentemente":  "claro",
    "específicamente": "en concreto",
    "principalmente":  "sobre todo",
    "aproximadamente": "unos",
}


@dataclasses.dataclass
class TranslationCandidate:
    """A candidate translation that fits a duration budget.

    Attributes:
        text: The translated text.
        char_count: Number of characters in *text*.
        brevity_rationale: Short explanation of what was shortened.
    """
    text: str
    char_count: int
    brevity_rationale: str = ""


@dataclasses.dataclass
class FailureAnalysis:
    """Diagnostic summary of the dominant failure mode in a clip.

    Attributes:
        failure_category: One of "duration_overflow", "cumulative_drift",
            "stretch_quality", or "ok".
        likely_root_cause: One-sentence description.
        suggested_change: Most impactful next action.
    """
    failure_category: str
    likely_root_cause: str
    suggested_change: str


def analyze_failures(report: dict) -> FailureAnalysis:
    """Classify the dominant failure mode from a clip evaluation report.

    Pure heuristic — no LLM needed.  The thresholds below match the policy
    bands defined in ``alignment.decide_action``.

    Args:
        report: Dict returned by ``clip_evaluation_report()``.  Expected keys:
            ``mean_abs_duration_error_s``, ``pct_severe_stretch``,
            ``total_cumulative_drift_s``, ``n_translation_retries``.

    Returns:
        A ``FailureAnalysis`` dataclass.
    """
    mean_err = report.get("mean_abs_duration_error_s", 0.0)
    pct_severe = report.get("pct_severe_stretch", 0.0)
    drift = abs(report.get("total_cumulative_drift_s", 0.0))
    retries = report.get("n_translation_retries", 0)

    if pct_severe > 20:
        return FailureAnalysis(
            failure_category="duration_overflow",
            likely_root_cause=(
                f"{pct_severe:.0f}% of segments exceed the 1.4x stretch threshold — "
                "translated text is consistently too long for the available time window."
            ),
            suggested_change="Implement duration-aware translation re-ranking (P8).",
        )

    if drift > 3.0:
        return FailureAnalysis(
            failure_category="cumulative_drift",
            likely_root_cause=(
                f"Total drift is {drift:.1f}s — small per-segment overflows "
                "accumulate because gaps between segments are not being reclaimed."
            ),
            suggested_change="Enable gap_shift in the global alignment optimizer (P9).",
        )

    if mean_err > 0.8:
        return FailureAnalysis(
            failure_category="stretch_quality",
            likely_root_cause=(
                f"Mean duration error is {mean_err:.2f}s — segments fit within "
                "stretch limits but the stretch distorts audio quality."
            ),
            suggested_change="Lower the mild_stretch ceiling or shorten translations.",
        )

    return FailureAnalysis(
        failure_category="ok",
        likely_root_cause="No dominant failure mode detected.",
        suggested_change="Review individual outlier segments if any remain.",
    )


def get_shorter_translations(
    source_text: str,
    baseline_es: str,
    target_duration_s: float,
    context_prev: str = "",
    context_next: str = "",
) -> list[TranslationCandidate]:
    """Return shorter translation candidates that fit *target_duration_s*.

    .. admonition:: Student Assignment — Duration-Aware Translation Re-ranking

       This function is intentionally a **stub that returns an empty list**.
       Your task is to implement a strategy that produces shorter
       target-language translations when the baseline translation is too long
       for the time budget.

       **Inputs**

       ============== ======== ==================================================
       Parameter      Type     Description
       ============== ======== ==================================================
       source_text    str      Original source-language segment text
       baseline_es    str      Baseline target-language translation (from argostranslate)
       target_duration_s float Time budget in seconds for this segment
       context_prev   str      Text of the preceding segment (for coherence)
       context_next   str      Text of the following segment (for coherence)
       ============== ======== ==================================================

       **Outputs**

       A list of ``TranslationCandidate`` objects, sorted shortest first.
       Each candidate has:

       - ``text``: the shortened target-language translation
       - ``char_count``: ``len(text)``
       - ``brevity_rationale``: short note on what was changed

       **Duration heuristic**: target-language TTS produces ~15 characters/second
       (or ~4.5 syllables/second for Romance languages).  So a 3-second budget
       ≈ 45 characters.

       **Approaches to consider** (pick one or combine):

       1. **Rule-based shortening** — strip filler words, use shorter synonyms
          from a lookup table, contract common phrases
          (e.g. "en este momento" → "ahora").
       2. **Multiple translation backends** — call argostranslate with
          paraphrased input, or use a second translation model, then pick
          the shortest output that preserves meaning.
       3. **LLM re-ranking** — use an LLM (e.g. via an API) to generate
          condensed alternatives.  This was the previous approach but adds
          latency, cost, and a runtime dependency.
       4. **Hybrid** — rule-based first, fall back to LLM only for segments
          that still exceed the budget.

       **Evaluation criteria**: the caller selects the candidate whose
       ``len(text) / 15.0`` is closest to ``target_duration_s``.

    Returns:
        Empty list (stub).  Implement to return ``TranslationCandidate`` items.
    """
    if not baseline_es or not baseline_es.strip():
        return []

    candidates: list[TranslationCandidate] = []
    seen_texts: set[str] = {baseline_es.strip()}

    def _add(text: str, rationale: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # tidy punctuation spacing
        if not text or text in seen_texts:
            return
        # Reject candidates that are not actually shorter, or that lost so much
        # content they're suspect (more than 60% reduction).
        if len(text) >= len(baseline_es):
            return
        if len(text) < max(8, int(len(baseline_es) * 0.4)):
            return
        seen_texts.add(text)
        candidates.append(TranslationCandidate(
            text=text,
            char_count=len(text),
            brevity_rationale=rationale,
        ))

    # ── Strategy 1 — strip leading discourse markers ───────────────────────
    s1 = baseline_es
    stripped_marker = ""
    for marker in _DISCOURSE_MARKERS:
        if s1.lstrip().lower().startswith(marker.lower()):
            s1 = s1.lstrip()[len(marker):].lstrip().capitalize()
            stripped_marker = marker.rstrip(",")
            break
    if stripped_marker:
        _add(s1, f"removed discourse marker '{stripped_marker}'")

    # ── Strategy 2 — phrase-level shortenings (greedy longest-match) ──────
    s2 = baseline_es
    phrase_changes: list[tuple[str, str]] = []
    for long_phrase, short_phrase in _PHRASE_SHORTENINGS:
        pat = re.compile(r"\b" + re.escape(long_phrase) + r"\b", re.IGNORECASE)
        if pat.search(s2):
            s2 = pat.sub(short_phrase, s2)
            phrase_changes.append((long_phrase, short_phrase))
    if phrase_changes:
        rationale = "phrase contractions: " + "; ".join(
            f"'{a}'→'{b}'" for a, b in phrase_changes[:3]
        )
        _add(s2, rationale)

    # ── Strategy 3 — single-word substitutions on top of strategy 2 ───────
    s3 = s2
    word_changes: list[tuple[str, str]] = []
    for long_word, short_word in _WORD_SHORTENINGS.items():
        pat = re.compile(r"\b" + re.escape(long_word) + r"\b", re.IGNORECASE)
        if pat.search(s3):
            s3 = pat.sub(short_word, s3)
            word_changes.append((long_word, short_word))
    if word_changes:
        rationale = "word substitutions: " + "; ".join(
            f"'{a}'→'{b}'" for a, b in word_changes[:3]
        )
        if phrase_changes:
            rationale = "phrase + " + rationale
        _add(s3, rationale)

    # ── Strategy 4 — combined: discourse strip + phrase + word ────────────
    s4 = s3
    if stripped_marker:
        for marker in _DISCOURSE_MARKERS:
            if s4.lstrip().lower().startswith(marker.lower()):
                s4 = s4.lstrip()[len(marker):].lstrip().capitalize()
                break
        if s4 != s3:
            _add(s4, "combined: discourse marker + phrase + word substitutions")

    # ── Strategy 5 — drop trailing parenthetical / appositive ─────────────
    # Only when the result is still well over the duration target's lower bound.
    s5 = re.sub(r"\s*\([^)]*\)\s*$", "", baseline_es).rstrip(",;.: ")
    if s5 != baseline_es and s5:
        if not s5.endswith((".", "!", "?")):
            s5 += "."
        _add(s5, "dropped trailing parenthetical")

    # Sort shortest first so the caller's `candidates[0]` heuristic picks
    # the most aggressive rewrite that preserves meaning.
    candidates.sort(key=lambda c: c.char_count)

    logger.info(
        "get_shorter_translations: budget=%.1fs baseline=%d chars → "
        "%d candidate(s) [shortest=%d chars]",
        target_duration_s,
        len(baseline_es),
        len(candidates),
        candidates[0].char_count if candidates else len(baseline_es),
    )
    return candidates
