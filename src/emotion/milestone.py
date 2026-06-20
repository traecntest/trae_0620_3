"""Milestone detection.

A *milestone* fires the very first time the baby produces a target word
(e.g. the first ``抱抱``).  The judge is decoupled from storage: it takes a
``seen_check`` callable (``word -> bool``) so it can be unit-tested without a
database, and returns a list of :class:`MilestoneRecord` for the caller to
persist (which writes the special ``Milestone`` tag in SQLite).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import config
from src.asr.keyword import KeywordHit, KeywordSpotter
from src.asr.recognizer import ASRResult


@dataclass
class MilestoneRecord:
    word: str
    timestamp: float          # seconds within the clip
    confidence: float


class MilestoneJudge:
    def __init__(
        self,
        spotter: KeywordSpotter | None = None,
        words: Iterable[str] | None = None,
    ):
        self.words = set(words if words is not None else config.MILESTONE_WORDS)
        self.spotter = spotter or KeywordSpotter()

    def evaluate(
        self,
        result: ASRResult,
        seen_check: Callable[[str], bool],
    ) -> list[MilestoneRecord]:
        """Return milestones for words spoken *for the first time* here."""
        hits: list[KeywordHit] = self.spotter.find(result)
        milestones: list[MilestoneRecord] = []
        seen_now: set[str] = set()
        for hit in hits:
            if hit.word not in self.words:
                continue
            # a word repeated within the same clip is still one milestone
            if hit.word in seen_now:
                continue
            if not seen_check(hit.word):
                seen_now.add(hit.word)
                milestones.append(MilestoneRecord(
                    word=hit.word,
                    timestamp=hit.start,
                    confidence=hit.conf,
                ))
        milestones.sort(key=lambda m: m.timestamp)
        return milestones
