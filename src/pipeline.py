"""Orchestration pipeline.

Ties the audio / ASR / emotion / storage subsystems together so the GUI and
CLI only deal with a single :class:`ClipProcessor`.

A processed clip produces:
* a WAV file under ``year/month/date_tag.wav``
* a SQLite metadata row (emotion, acoustic features, transcript)
* milestone rows + the special ``Milestone`` flag when target words are
  heard for the first time
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

import config
from src.asr.keyword import KeywordHit, KeywordSpotter
from src.asr.recognizer import ASRResult, VoskRecognizer
from src.audio.features import AcousticFeatures, extract_features
from src.emotion.analyzer import EmotionAnalyzer, EmotionResult
from src.emotion.milestone import MilestoneJudge, MilestoneRecord
from src.storage.database import Clip, Database
from src.storage.file_manager import FileManager


@dataclass
class ProcessOutcome:
    clip: Clip
    features: AcousticFeatures
    emotion: EmotionResult
    asr: Optional[ASRResult]
    keywords: list[KeywordHit]
    milestones: list[MilestoneRecord]


class ClipProcessor:
    """Stateful service that owns the recogniser, emotion + milestone logic."""

    def __init__(
        self,
        db: Database,
        files: FileManager,
        recognizer: Optional[VoskRecognizer] = None,
        analyzer: Optional[EmotionAnalyzer] = None,
        spotter: Optional[KeywordSpotter] = None,
        judge: Optional[MilestoneJudge] = None,
    ):
        self.db = db
        self.files = files
        self.analyzer = analyzer or EmotionAnalyzer()
        self.spotter = spotter or KeywordSpotter()
        self.judge = judge or MilestoneJudge(self.spotter)
        self._recognizer = recognizer  # lazily created on first use

    # ------------------------------------------------------------------ #
    @property
    def recognizer(self) -> VoskRecognizer:
        if self._recognizer is None:
            self._recognizer = VoskRecognizer()
        return self._recognizer

    def asr_available(self) -> bool:
        try:
            _ = self.recognizer  # noqa: F841
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    def process(
        self,
        samples: np.ndarray,
        recorded_at: Optional[datetime] = None,
        tag: str = config.SAFE_TAG_DEFAULT,
        run_asr: bool = True,
    ) -> ProcessOutcome:
        recorded_at = recorded_at or datetime.now()
        samples = np.asarray(samples, dtype=np.float32)

        # 1. acoustic features
        features = extract_features(samples, config.SAMPLE_RATE)

        # 2. ASR + keyword spotting (optional)
        asr: Optional[ASRResult] = None
        keywords: list[KeywordHit] = []
        transcript = ""
        if run_asr and samples.size > config.SAMPLE_RATE * 0.3:
            try:
                asr = self.recognizer.transcribe(samples)
                transcript = asr.text
                keywords = self.spotter.find(asr)
            except Exception:
                asr = None

        # 3. emotion
        has_speech = bool(keywords) or bool(transcript and transcript.replace(" ", ""))
        emotion = self.analyzer.analyze(features, has_speech=has_speech)

        # 4. persist audio + metadata
        chosen_tag = tag if tag and tag != config.SAFE_TAG_DEFAULT else self._auto_tag(emotion, keywords)
        audio_path = self.files.save(samples, recorded_at, chosen_tag)
        clip = Clip(
            id=None,
            recorded_at=recorded_at,
            tag=chosen_tag,
            emotion=emotion.label,
            emotion_confidence=emotion.confidence,
            f0_mean=features.f0_mean,
            energy_db=features.energy_mean_db,
            duration_s=features.duration,
            transcript=transcript,
            audio_path=str(audio_path.relative_to(self.files.root)).replace("\\", "/"),
        )
        clip_id = self.db.add_clip(clip)
        clip.id = clip_id

        # 5. milestones -> special Milestone tag in DB
        milestones: list[MilestoneRecord] = []
        if asr is not None:
            milestones = self.judge.evaluate(asr, self.db.has_milestone_word)
            for m in milestones:
                self.db.add_milestone(clip_id, m.word, m.timestamp, m.confidence)
                clip.is_milestone = True
                if m.word not in clip.tags:
                    clip.tags.append(m.word)

        if milestones:
            self.db.update_clip_extra(clip_id, tags=clip.tags, is_milestone=True)

        return ProcessOutcome(
            clip=clip,
            features=features,
            emotion=emotion,
            asr=asr,
            keywords=keywords,
            milestones=milestones,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _auto_tag(emotion: EmotionResult, keywords: list[KeywordHit]) -> str:
        if keywords:
            return keywords[0].word
        return {
            "crying": "哭泣", "laughing": "大笑",
            "babbling": "咿呀", "talking": "说话",
        }.get(emotion.label, config.SAFE_TAG_DEFAULT)

    # ------------------------------------------------------------------ #
    def re_tag(self, clip: Clip, new_tag: str) -> str:
        """Apply a drag-dropped new tag: move the file + update DB + path."""
        from src.storage.file_manager import sanitize_tag
        new_tag = sanitize_tag(new_tag)
        current_abs = self.files.resolve(clip.audio_path)
        new_abs = self.files.move_for_tag(current_abs, clip.recorded_at, new_tag)
        rel = str(new_abs.relative_to(self.files.root)).replace("\\", "/")
        self.db.update_clip_tag(clip.id, new_tag, rel)
        clip.tag = new_tag
        clip.audio_path = rel
        return rel
