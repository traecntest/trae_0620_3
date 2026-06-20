"""SQLite metadata store for BabyVoice Museum.

Tables
------
clips       one row per recorded segment (tag, date, emotion, acoustic
            features, transcript, relative audio path, extra tags JSON)
milestones  first-occurrence events -> the special ``Milestone`` tag
clip_tags   additional human tags (drag-drop editable)

The store is thread-safe (GUI worker threads) via a re-entrant lock and
``check_same_thread=False``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


@dataclass
class Clip:
    id: Optional[int]
    recorded_at: datetime
    tag: str
    emotion: str
    emotion_confidence: float
    f0_mean: float
    energy_db: float
    duration_s: float
    transcript: str
    audio_path: str                       # relative to AUDIO_ROOT
    tags: list[str] = field(default_factory=list)
    is_milestone: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "recorded_at": self.recorded_at.isoformat(sep=" ", timespec="seconds"),
            "tag": self.tag,
            "emotion": self.emotion,
            "emotion_zh": _EMOTION_ZH.get(self.emotion, self.emotion),
            "emotion_confidence": round(self.emotion_confidence, 3),
            "f0_mean": round(self.f0_mean, 1),
            "energy_db": round(self.energy_db, 2),
            "duration_s": round(self.duration_s, 2),
            "transcript": self.transcript,
            "audio_path": self.audio_path,
            "tags": list(self.tags),
            "is_milestone": self.is_milestone,
        }


@dataclass
class Milestone:
    id: Optional[int]
    clip_id: int
    word: str
    timestamp: float
    confidence: float
    created_at: datetime


_EMOTION_ZH = {
    "crying": "哭泣", "laughing": "大笑", "babbling": "咿呀学语",
    "talking": "说话", "neutral": "安静",
}


class Database:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS clips (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at        TEXT    NOT NULL,
        tag                TEXT    NOT NULL,
        emotion            TEXT    NOT NULL,
        emotion_confidence REAL    NOT NULL DEFAULT 0,
        f0_mean            REAL    NOT NULL DEFAULT 0,
        energy_db          REAL    NOT NULL DEFAULT 0,
        duration_s         REAL    NOT NULL DEFAULT 0,
        transcript         TEXT    NOT NULL DEFAULT '',
        audio_path         TEXT    NOT NULL,
        tags               TEXT    NOT NULL DEFAULT '[]',
        is_milestone       INTEGER NOT NULL DEFAULT 0,
        created_at         TEXT    NOT NULL
    );
    CREATE TABLE IF NOT EXISTS milestones (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        clip_id     INTEGER NOT NULL,
        word        TEXT    NOT NULL,
        timestamp   REAL    NOT NULL,
        confidence  REAL    NOT NULL DEFAULT 0,
        created_at  TEXT    NOT NULL,
        FOREIGN KEY (clip_id) REFERENCES clips(id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_clips_recorded ON clips(recorded_at);
    CREATE INDEX IF NOT EXISTS idx_clips_tag ON clips(tag);
    CREATE INDEX IF NOT EXISTS idx_clips_emotion ON clips(emotion);
    CREATE INDEX IF NOT EXISTS idx_milestones_word ON milestones(word);
    """

    def __init__(self, path: Path = config.DB_PATH):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # clips
    # ------------------------------------------------------------------ #
    def add_clip(self, clip: Clip) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO clips
                   (recorded_at, tag, emotion, emotion_confidence, f0_mean,
                    energy_db, duration_s, transcript, audio_path, tags,
                    is_milestone, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    clip.recorded_at.isoformat(sep=" ", timespec="seconds"),
                    clip.tag, clip.emotion, clip.emotion_confidence,
                    clip.f0_mean, clip.energy_db, clip.duration_s,
                    clip.transcript, clip.audio_path,
                    json.dumps(clip.tags, ensure_ascii=False),
                    int(clip.is_milestone),
                    datetime.now().isoformat(sep=" ", timespec="seconds"),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_clip_tag(self, clip_id: int, new_tag: str, new_audio_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE clips SET tag=?, audio_path=? WHERE id=?",
                (new_tag, new_audio_path, clip_id),
            )
            self._conn.commit()

    def update_clip_extra(self, clip_id: int, **fields) -> None:
        allowed = {"emotion", "emotion_confidence", "transcript", "tags",
                   "is_milestone", "f0_mean", "energy_db"}
        cols = [f"{k}=?" for k in fields if k in allowed]
        vals = [v for k, v in fields.items() if k in allowed]
        if not cols:
            return
        if "tags" in fields:
            vals[cols.index("tags=?")] = json.dumps(fields["tags"], ensure_ascii=False)
        vals.append(clip_id)
        with self._lock:
            self._conn.execute(f"UPDATE clips SET {', '.join(cols)} WHERE id=?", vals)
            self._conn.commit()

    def get_clip(self, clip_id: int) -> Optional[Clip]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM clips WHERE id=?", (clip_id,)
            ).fetchone()
        return self._row_to_clip(row) if row else None

    def list_clips(self, limit: int = 500, offset: int = 0) -> list[Clip]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM clips ORDER BY recorded_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_clip(r) for r in rows]

    def search_clips(
        self,
        tag: Optional[str] = None,
        emotion: Optional[str] = None,
        keyword: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        milestone_only: bool = False,
    ) -> list[Clip]:
        q = "SELECT * FROM clips WHERE 1=1"
        params: list = []
        if tag:
            q += " AND tag=?"; params.append(tag)
        if emotion:
            q += " AND emotion=?"; params.append(emotion)
        if keyword:
            q += " AND transcript LIKE ?"; params.append(f"%{keyword}%")
        if date_from:
            q += " AND recorded_at>=?"; params.append(date_from.isoformat(sep=" ", timespec="seconds"))
        if date_to:
            q += " AND recorded_at<=?"; params.append(date_to.isoformat(sep=" ", timespec="seconds"))
        if milestone_only:
            q += " AND is_milestone=1"
        q += " ORDER BY recorded_at DESC"
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_clip(r) for r in rows]

    def delete_clip(self, clip_id: int) -> Optional[str]:
        """Delete a clip row; return its relative audio path (caller deletes file)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT audio_path FROM clips WHERE id=?", (clip_id,)
            ).fetchone()
            self._conn.execute("DELETE FROM clips WHERE id=?", (clip_id,))
            self._conn.commit()
        return row["audio_path"] if row else None

    # ------------------------------------------------------------------ #
    # milestones
    # ------------------------------------------------------------------ #
    def add_milestone(self, clip_id: int, word: str, timestamp: float, confidence: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO milestones (clip_id, word, timestamp, confidence, created_at)
                   VALUES (?,?,?,?,?)""",
                (clip_id, word, timestamp, confidence,
                 datetime.now().isoformat(sep=" ", timespec="seconds")),
            )
            # mark the clip itself with the special Milestone tag
            self._conn.execute(
                "UPDATE clips SET is_milestone=1 WHERE id=?", (clip_id,)
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def has_milestone_word(self, word: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM milestones WHERE word=? LIMIT 1", (word,)
            ).fetchone()
        return row is not None

    def list_milestones(self) -> list[Milestone]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM milestones ORDER BY created_at DESC"
            ).fetchall()
        return [Milestone(
            id=r["id"], clip_id=r["clip_id"], word=r["word"],
            timestamp=r["timestamp"], confidence=r["confidence"],
            created_at=datetime.fromisoformat(r["created_at"]),
        ) for r in rows]

    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_clip(row: sqlite3.Row) -> Clip:
        try:
            tags = json.loads(row["tags"]) if row["tags"] else []
        except (TypeError, ValueError):
            tags = []
        return Clip(
            id=row["id"],
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            tag=row["tag"],
            emotion=row["emotion"],
            emotion_confidence=row["emotion_confidence"],
            f0_mean=row["f0_mean"],
            energy_db=row["energy_db"],
            duration_s=row["duration_s"],
            transcript=row["transcript"],
            audio_path=row["audio_path"],
            tags=tags,
            is_milestone=bool(row["is_milestone"]),
        )
