#!/usr/bin/env python3
"""Relabel pyannote diarization intervals against trusted reference WAVs.

Pyannote's clustering can drift across a long video — the same physical
speaker gets cluster ID `SPEAKER_00` in the first half and `SPEAKER_02`
in the second half. The TTS path then pulls the wrong reference voice
for those segments, so dubbed voices feel scrambled.

This script bypasses pyannote's clustering for the *labeling* step:

    1. Embed each reference WAV in pipeline_data/speakers/<lang>/SPEAKER_*.wav
       with the SpeechBrain ECAPA-TDNN model → 3 fixed centroid vectors.
    2. Embed every pyannote diarization interval the same way.
    3. Assign each interval the label whose centroid is nearest by cosine
       similarity.
    4. Write the relabeled intervals back to the cached diarization JSON.

After this, calling POST /api/diarize/<video_id> hits the cache and
merges the corrected labels into transcription + translation JSONs.

Usage:
    uv run python scripts/relabel_speakers.py <video_id>
    uv run python scripts/relabel_speakers.py <video_id> --lang es

Prerequisites:
    - Diarization has run (cached pipeline_data/api/diarizations/<title>.json)
    - Reference WAVs exist (pipeline_data/speakers/<lang>/SPEAKER_*.wav).
      Run scripts/extract_speaker_voices.py first.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DATA = PROJECT_ROOT / "pipeline_data"
REGISTRY_PATH = PROJECT_ROOT / "video_registry.yml"

# ECAPA-TDNN expects 16 kHz mono input and produces 192-dim embeddings.
TARGET_SR = 16000
MIN_INTERVAL_SECONDS = 0.5  # below this we don't have enough signal to embed reliably

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def resolve_title(video_id: str) -> tuple[str, str]:
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    for v in registry.get("videos", []):
        if v["id"] == video_id:
            return v["title"], v["target_language"]
    raise SystemExit(f"video_id {video_id!r} not found in {REGISTRY_PATH}")


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    """Load *path* and return (mono float tensor of shape (T,), 16 kHz)."""
    signal, sr = torchaudio.load(str(path))
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        signal = torchaudio.transforms.Resample(sr, TARGET_SR)(signal)
    return signal.squeeze(0), TARGET_SR


def load_encoder():
    from speechbrain.inference.speaker import EncoderClassifier
    cache_dir = PROJECT_ROOT / ".cache" / "spkrec_ecapa"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading SpeechBrain ECAPA-TDNN encoder (CPU)…")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache_dir),
        run_opts={"device": "cpu"},
    )


def embed(encoder, signal: torch.Tensor) -> np.ndarray:
    """Return a unit-normalized 192-dim embedding for a 1-D 16 kHz waveform."""
    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    with torch.inference_mode():
        emb = encoder.encode_batch(signal).squeeze().detach().cpu().numpy()
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-8 else emb


def slice_signal(audio: torch.Tensor, sr: int, start_s: float, end_s: float) -> torch.Tensor:
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    return audio[s:e]


def relabel(
    audio_path: Path,
    diar_segments: list[dict],
    references: dict[str, Path],
) -> tuple[list[dict], dict]:
    encoder = load_encoder()

    log.info(f"Loading source audio: {audio_path.name}")
    audio, sr = load_audio(audio_path)
    log.info(f"  duration={len(audio)/sr:.1f}s sr={sr}")

    log.info(f"Embedding {len(references)} reference WAVs…")
    centroids: dict[str, np.ndarray] = {}
    for label, path in sorted(references.items()):
        ref_audio, _ = load_audio(path)
        centroids[label] = embed(encoder, ref_audio)
        log.info(f"  {label}: {path.name} ({len(ref_audio)/sr:.1f}s)")

    # Sanity check: how distinct are the references from each other?
    log.info("\nReference centroid pairwise cosine similarity:")
    labels = sorted(centroids)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            sim = float(np.dot(centroids[a], centroids[b]))
            log.info(f"  {a} vs {b}: {sim:+.3f}")

    log.info(f"\nRelabeling {len(diar_segments)} pyannote intervals…")
    relabeled: list[dict] = []
    changes = Counter()
    skipped_short = 0
    for s in diar_segments:
        old = s["speaker"]
        sig = slice_signal(audio, sr, s["start_s"], s["end_s"])
        if len(sig) < int(MIN_INTERVAL_SECONDS * sr):
            relabeled.append(dict(s))
            skipped_short += 1
            continue
        e = embed(encoder, sig)
        scores = {label: float(np.dot(c, e)) for label, c in centroids.items()}
        new = max(scores, key=scores.get)
        relabeled.append({**s, "speaker": new})
        changes[(old, new)] += 1

    moved = sum(n for (o, n), n_c in changes.items() if o != n for n in [n_c])
    log.info("\n--- Label transitions (old → new) ---")
    for (old, new), n in sorted(changes.items()):
        marker = "" if old == new else "  ←"
        log.info(f"  {old} → {new}: {n}{marker}")
    log.info(f"\nIntervals relabeled: {moved} / {len(diar_segments)}")
    log.info(f"Intervals skipped (<{MIN_INTERVAL_SECONDS}s, kept original label): {skipped_short}")

    summary = {
        "total_intervals": len(diar_segments),
        "intervals_changed": moved,
        "intervals_skipped_short": skipped_short,
        "ref_pairwise_similarity": {
            f"{a}|{b}": float(np.dot(centroids[a], centroids[b]))
            for i, a in enumerate(labels) for b in labels[i + 1:]
        },
    }
    return relabeled, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video_id", help="key in video_registry.yml")
    ap.add_argument("--lang", help="override target language")
    args = ap.parse_args()

    title, registry_lang = resolve_title(args.video_id)
    lang = args.lang or registry_lang

    diar_path = PIPELINE_DATA / "api" / "diarizations" / f"{title}.json"
    audio_path = PIPELINE_DATA / "api" / "diarizations" / f"{title}.wav"
    speakers_dir = PIPELINE_DATA / "speakers" / lang

    if not diar_path.exists():
        raise SystemExit(f"diarization JSON missing: {diar_path}")
    if not audio_path.exists():
        raise SystemExit(f"diarization audio missing: {audio_path}\n"
                          "Diarize must have run successfully first.")

    diar = json.loads(diar_path.read_text())
    diar_segments = diar.get("segments", [])
    if not diar_segments:
        raise SystemExit(f"{diar_path} has no intervals — re-run diarize endpoint first")

    references: dict[str, Path] = {}
    for p in sorted(speakers_dir.glob("SPEAKER_*.wav")):
        m = re.match(r"^(SPEAKER_\d+)\.wav$", p.name)
        if m:
            references[m.group(1)] = p
    if len(references) < 2:
        raise SystemExit(f"need at least 2 SPEAKER_NN.wav references in {speakers_dir}; "
                          f"found {len(references)}")

    relabeled_segments, summary = relabel(audio_path, diar_segments, references)

    speakers = sorted({s["speaker"] for s in relabeled_segments})
    out = {"speakers": speakers, "segments": relabeled_segments, "_relabeled": summary}
    diar_path.write_text(json.dumps(out))
    log.info(f"\nWrote relabeled diarization to: {diar_path.name}")
    log.info("Next: re-hit POST /api/diarize/<video_id> to merge into translation cache, "
             "then re-run TTS+stitch.")


if __name__ == "__main__":
    main()
