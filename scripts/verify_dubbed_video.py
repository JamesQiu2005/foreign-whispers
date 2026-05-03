#!/usr/bin/env python3
"""Objective verification of a dubbed video.

Two checks, both quantitative — nothing requires a human to listen.

  1. Audio delay
     Extract audio from the original and dubbed MP4s, run silero-VAD on
     each, and report the difference in first-speech-onset.

  2. Speaker attribution in the dubbed audio
     For every segment in the translated JSON, slice the corresponding
     window from the dubbed audio, embed it with ECAPA-TDNN, and check
     whether the nearest reference centroid matches the segment's
     diarization label.

Usage:
    uv run python scripts/verify_dubbed_video.py <video_id> --config c-XXXXXXX
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torchaudio
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "pipeline_data"
REGISTRY = PROJECT_ROOT / "video_registry.yml"
TARGET_SR = 16000
MIN_SEG_S = 0.5


def resolve_title(video_id: str) -> tuple[str, str]:
    reg = yaml.safe_load(REGISTRY.read_text())
    for v in reg.get("videos", []):
        if v["id"] == video_id:
            return v["title"], v["target_language"]
    sys.exit(f"video_id {video_id!r} not in registry")


def extract_wav(src_mp4: Path, dst_wav: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp4),
         "-vn", "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1",
         str(dst_wav)],
        check=True,
    )


def load_mono(path: Path) -> tuple[torch.Tensor, int]:
    sig, sr = torchaudio.load(str(path))
    if sig.shape[0] > 1:
        sig = sig.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        sig = torchaudio.transforms.Resample(sr, TARGET_SR)(sig)
    return sig.squeeze(0), TARGET_SR


def first_speech_onset_s(wav_path: Path) -> float | None:
    """Return the start time (s) of the first speech region per silero-VAD."""
    try:
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError:
        return None
    model = load_silero_vad()
    wav = read_audio(str(wav_path))
    ts = get_speech_timestamps(wav, model, return_seconds=True)
    return ts[0]["start"] if ts else None


def load_ecapa():
    from speechbrain.inference.speaker import EncoderClassifier
    cache = PROJECT_ROOT / ".cache" / "spkrec_ecapa"
    cache.mkdir(parents=True, exist_ok=True)
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(cache),
        run_opts={"device": "cpu"},
    )


def embed(encoder, sig: torch.Tensor) -> np.ndarray:
    if sig.dim() == 1:
        sig = sig.unsqueeze(0)
    with torch.inference_mode():
        e = encoder.encode_batch(sig).squeeze().detach().cpu().numpy()
    n = np.linalg.norm(e)
    return e / n if n > 1e-8 else e


def slice_sig(audio: torch.Tensor, sr: int, start_s: float, end_s: float) -> torch.Tensor:
    s = max(0, int(start_s * sr))
    e = min(len(audio), int(end_s * sr))
    return audio[s:e]


def check_delay(orig_mp4: Path, dub_mp4: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        orig_wav = td / "orig.wav"
        dub_wav = td / "dub.wav"
        extract_wav(orig_mp4, orig_wav)
        extract_wav(dub_mp4, dub_wav)
        o = first_speech_onset_s(orig_wav)
        d = first_speech_onset_s(dub_wav)
    out = {"original_first_speech_s": o, "dubbed_first_speech_s": d,
           "delta_s": (d - o) if (o is not None and d is not None) else None}
    return out


def check_speakers(dub_mp4: Path, translations_path: Path, speakers_dir: Path) -> dict:
    refs = {}
    for p in sorted(speakers_dir.glob("SPEAKER_*.wav")):
        m = re.match(r"^(SPEAKER_\d+)\.wav$", p.name)
        if m:
            refs[m.group(1)] = p
    if len(refs) < 2:
        return {"error": f"need ≥2 SPEAKER_NN.wav refs in {speakers_dir}; found {len(refs)}"}

    encoder = load_ecapa()
    centroids = {label: embed(encoder, load_mono(p)[0]) for label, p in refs.items()}

    pairs = {}
    labels = sorted(centroids)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pairs[f"{a}|{b}"] = float(np.dot(centroids[a], centroids[b]))

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "dub.wav"
        extract_wav(dub_mp4, wav)
        audio, sr = load_mono(wav)

    segs = json.loads(translations_path.read_text()).get("segments", [])
    if not segs:
        return {"error": f"no segments in {translations_path}"}

    confusion: Counter = Counter()
    per_speaker: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0,
                                                        "mean_top1_sim": 0.0,
                                                        "mean_margin": 0.0})
    correct = 0
    skipped = 0
    detail = []
    for s in segs:
        labeled = s.get("speaker")
        if not labeled or labeled not in centroids:
            skipped += 1
            continue
        sig = slice_sig(audio, sr, s["start"], s["end"])
        if len(sig) < int(MIN_SEG_S * sr):
            skipped += 1
            continue
        e = embed(encoder, sig)
        scores = {label: float(np.dot(c, e)) for label, c in centroids.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top1, top1_sim = ranked[0]
        top2_sim = ranked[1][1] if len(ranked) > 1 else 0.0
        is_correct = (top1 == labeled)
        if is_correct:
            correct += 1
        confusion[(labeled, top1)] += 1
        ps = per_speaker[labeled]
        ps["n"] += 1
        ps["correct"] += int(is_correct)
        ps["mean_top1_sim"] += top1_sim
        ps["mean_margin"] += (top1_sim - top2_sim)
        detail.append({"start": round(s["start"], 2), "end": round(s["end"], 2),
                       "labeled": labeled, "predicted": top1,
                       "top1_sim": round(top1_sim, 3),
                       "margin": round(top1_sim - top2_sim, 3)})

    for ps in per_speaker.values():
        if ps["n"]:
            ps["mean_top1_sim"] /= ps["n"]
            ps["mean_margin"] /= ps["n"]

    n_eval = sum(ps["n"] for ps in per_speaker.values())
    return {
        "n_segments_total": len(segs),
        "n_segments_evaluated": n_eval,
        "n_segments_skipped": skipped,
        "overall_accuracy": (correct / n_eval) if n_eval else None,
        "per_speaker": dict(per_speaker),
        "confusion_labeled_to_predicted": {f"{a}->{b}": n for (a, b), n in confusion.items()},
        "reference_pairwise_cosine": pairs,
        "per_segment": detail,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video_id", help="key in video_registry.yml")
    ap.add_argument("--config", required=True, help="dubbed_videos/<config> dir name (e.g. c-1eafa11)")
    ap.add_argument("--lang", help="override target language")
    ap.add_argument("--out", help="write JSON report to this path")
    args = ap.parse_args()

    title, reg_lang = resolve_title(args.video_id)
    lang = args.lang or reg_lang

    orig_mp4 = PIPELINE / "api" / "videos" / f"{title}.mp4"
    dub_mp4 = PIPELINE / "api" / "dubbed_videos" / args.config / f"{title}.mp4"
    trans = PIPELINE / "api" / "translations" / "argos" / f"{title}.json"
    speakers_dir = PIPELINE / "speakers" / lang

    for p in (orig_mp4, dub_mp4, trans, speakers_dir):
        if not p.exists():
            sys.exit(f"missing: {p}")

    print(f"[1/2] Audio-delay check ({title})")
    delay = check_delay(orig_mp4, dub_mp4)
    print(json.dumps(delay, indent=2))

    print(f"\n[2/3] Speaker-attribution check on DUBBED audio (config={args.config})")
    print("      (warns if local Coqui TTS ignored voice_map → all segs collapse to one voice)")
    spk_dub = check_speakers(dub_mp4, trans, speakers_dir)
    summary = {k: v for k, v in spk_dub.items() if k != "per_segment"}
    print(json.dumps(summary, indent=2, default=str))

    print(f"\n[3/3] Speaker-attribution check on ORIGINAL audio (label correctness)")
    print("      tests whether diarization+resegmentation labelled each segment with the right speaker")
    spk_orig = check_speakers(orig_mp4, trans, speakers_dir)
    summary = {k: v for k, v in spk_orig.items() if k != "per_segment"}
    print(json.dumps(summary, indent=2, default=str))

    report = {"video_id": args.video_id, "title": title, "config": args.config,
              "lang": lang, "delay": delay,
              "speakers_dubbed": spk_dub, "speakers_original": spk_orig}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull report (incl. per-segment) → {args.out}")


if __name__ == "__main__":
    main()
