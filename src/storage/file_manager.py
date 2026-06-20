"""File-system layout for raw audio.

Layout rule (per the spec)::

    <AUDIO_ROOT>/<year>/<month>/<YYYYMMDD>_<tag>.wav

e.g. ``data/audio/2026/06/20260620_抱抱.wav``.

Tags are sanitised so the filename is safe on Windows while still being
human-readable.  Collisions get a ``_2``, ``_3`` ... suffix.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import config


_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_tag(tag: str) -> str:
    tag = (tag or "").strip() or config.SAFE_TAG_DEFAULT
    tag = _WIN_ILLEGAL.sub("_", tag)
    tag = tag.replace(" ", "_")
    if len(tag) > config.TAG_MAX_LEN:
        tag = tag[: config.TAG_MAX_LEN]
    return tag or config.SAFE_TAG_DEFAULT


class FileManager:
    def __init__(self, root: Path = config.AUDIO_ROOT):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def relative_path(self, recorded_at: datetime, tag: str) -> Path:
        tag = sanitize_tag(tag)
        year = f"{recorded_at.year:04d}"
        month = f"{recorded_at.month:02d}"
        stem = f"{recorded_at.strftime('%Y%m%d')}_{tag}"
        rel = Path(year) / month / f"{stem}.wav"
        return rel

    def absolute_path(self, recorded_at: datetime, tag: str) -> Path:
        return self.root / self.relative_path(recorded_at, tag)

    def ensure_unique(self, recorded_at: datetime, tag: str) -> Path:
        """Return an absolute path that does not yet exist on disk."""
        base = self.absolute_path(recorded_at, tag)
        if not base.exists():
            return base
        stem, ext = base.stem, base.suffix
        i = 2
        while True:
            cand = base.with_name(f"{stem}_{i}{ext}")
            if not cand.exists():
                return cand
            i += 1

    # ------------------------------------------------------------------ #
    def save(self, samples, recorded_at: datetime, tag: str, sr: int = config.SAMPLE_RATE) -> Path:
        from src.audio.recorder import save_wav

        path = self.ensure_unique(recorded_at, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_wav(path, samples, sr)
        return path

    def resolve(self, relative: str | Path) -> Path:
        return self.root / relative

    def move_for_tag(self, current_abs: Path, recorded_at: datetime, new_tag: str) -> Path:
        """Re-file a clip when its tag changes (used by tag drag-drop edit)."""
        new_abs = self.ensure_unique(recorded_at, new_tag)
        new_abs.parent.mkdir(parents=True, exist_ok=True)
        if current_abs.resolve() != new_abs.resolve():
            current_abs.rename(new_abs)
        return new_abs
