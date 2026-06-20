"""One-click archive export.

For a set of clips we copy the WAV files into an export folder, write a
per-clip sidecar ``.json`` (emotion, features, transcript, milestones) and a
flat ``index.csv`` so the folder is self-describing and shareable.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Iterable

import config
from src.storage.database import Clip, Database


def export_clips(
    clips: Iterable[Clip],
    db: Database,
    dest: Path = config.EXPORT_ROOT,
    include_audio: bool = True,
) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    audio_dir = dest / "audio"
    if include_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    index_path = dest / "index.csv"
    clips = list(clips)
    with open(index_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "clip_id", "recorded_at", "tag", "emotion", "emotion_zh",
            "confidence", "f0_hz", "energy_db", "duration_s",
            "transcript", "milestone", "audio",
        ])
        for clip in clips:
            audio_name = ""
            if include_audio:
                src = config.AUDIO_ROOT / clip.audio_path
                if src.exists():
                    audio_name = src.name
                    shutil.copy2(src, audio_dir / audio_name)
            sidecar = {
                **clip.to_dict(),
                "milestones": [
                    {"word": m.word, "timestamp_s": m.timestamp, "confidence": m.confidence}
                    for m in db.list_milestones() if m.clip_id == clip.id
                ],
            }
            with open(dest / f"{Path(clip.audio_path).stem}.json", "w", encoding="utf-8") as sj:
                json.dump(sidecar, sj, ensure_ascii=False, indent=2)
            writer.writerow([
                clip.id, clip.recorded_at.isoformat(sep=" ", timespec="seconds"),
                clip.tag, clip.emotion, sidecar["emotion_zh"],
                round(clip.emotion_confidence, 3), round(clip.f0_mean, 1),
                round(clip.energy_db, 2), round(clip.duration_s, 2),
                clip.transcript, "Milestone" if clip.is_milestone else "", audio_name,
            ])
    return dest
