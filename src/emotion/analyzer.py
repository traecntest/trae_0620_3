"""Rule-engine emotion classification.

Given :class:`AcousticFeatures` (F0, MFCC, short-time energy, laugh
modulation) plus an optional ASR transcription, classify the vocal state
into one of::

    crying | laughing | babbling | talking | neutral

The rules are explicit and tunable through ``config`` so a paediatric
expert can re-calibrate thresholds without touching logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import config
from src.audio.features import AcousticFeatures

LABELS = ("crying", "laughing", "babbling", "talking", "neutral")


@dataclass
class EmotionResult:
    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            self.label = "neutral"


class EmotionAnalyzer:
    """Map acoustic features + (optional) recognised words to a label."""

    def analyze(
        self,
        features: AcousticFeatures,
        has_speech: bool = False,
    ) -> EmotionResult:
        scores = self._score(features, has_speech)
        label, conf = max(scores.items(), key=lambda kv: kv[1])
        # normalise confidence to 0..1 over the top-2 for a softer signal
        total = sum(scores.values())
        conf = conf / total if total > 0 else 0.0
        return EmotionResult(label=label, confidence=conf, scores=scores)

    # ------------------------------------------------------------------ #
    def _score(self, f: AcousticFeatures, has_speech: bool) -> dict[str, float]:
        s = {lbl: 0.0 for lbl in LABELS}

        # --- crying: sustained high pitch + loud + voiced -------------- #
        if f.f0_mean >= config.CRY_F0_MIN and f.energy_mean_db >= config.CRY_ENERGY_DB:
            s["crying"] += 0.5
        if f.f0_mean >= config.CRY_F0_MIN * 1.1 and f.voiced_ratio > 0.6:
            s["crying"] += 0.3
        # high pitch with low variability = distress cry (plateau wail)
        if f.f0_mean >= config.CRY_F0_MIN and f.f0_std < 40:
            s["crying"] += 0.2

        # --- laughing: rhythmic ~5-6 Hz amplitude modulation ----------- #
        if f.laugh_modulation > 0.25:
            s["laughing"] += 0.4 + 0.4 * f.laugh_modulation
        if 0.3 < f.voiced_ratio < 0.7 and f.energy_mean_db > config.CRY_ENERGY_DB:
            s["laughing"] += 0.1

        # --- babbling: intermittent voicing, moderate pitch ----------- #
        if 0.15 < f.voiced_ratio <= config.BABBLE_VOICED_RATIO_MAX \
                and 150 < f.f0_mean < config.CRY_F0_MIN:
            s["babbling"] += 0.4
        if f.f0_std > 60 and f.voiced_ratio < 0.6:
            s["babbling"] += 0.2

        # --- talking: real words recognised --------------------------- #
        if has_speech:
            s["talking"] += 0.6
        if has_speech and f.voiced_ratio > 0.5:
            s["talking"] += 0.2

        # --- neutral: silence / low energy ---------------------------- #
        if f.energy_mean_db < config.CRY_ENERGY_DB - 12 or f.voiced_ratio < 0.1:
            s["neutral"] += 0.6
        if not has_speech and f.f0_mean == 0:
            s["neutral"] += 0.3

        return {k: round(v, 4) for k, v in s.items()}

    # ------------------------------------------------------------------ #
    @staticmethod
    def label_zh(label: str) -> str:
        return {
            "crying": "哭泣",
            "laughing": "大笑",
            "babbling": "咿呀学语",
            "talking": "说话",
            "neutral": "安静",
        }.get(label, label)
