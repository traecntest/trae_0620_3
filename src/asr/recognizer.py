"""Offline Chinese ASR with word-level timestamps (Vosk).

Vosk ships small CN models that run fully offline and emit per-token
start/end times.  The small-cn model tokenises Chinese as single characters,
so :class:`KeywordSpotter` joins consecutive tokens back into words before
matching ``抱抱`` etc.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config


@dataclass
class Token:
    text: str
    start: float
    end: float
    conf: float


@dataclass
class ASRResult:
    text: str
    tokens: list[Token] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.tokens[-1].end if self.tokens else 0.0


class VoskRecognizer:
    """Offline recognizer wrapping a single shared :class:`vosk.Model`.

    A process-wide model cache avoids reloading the 40 MB+ model on every
    clip.
    """

    _model_cache: dict[str, "object"] = {}

    def __init__(self, model_path: Optional[str] = None, sr: int = config.SAMPLE_RATE):
        from vosk import Model, SetLogLevel

        self.sr = sr
        path = str(model_path or config.VOSK_MODEL_PATH)
        if path not in self._model_cache:
            if not config.VOSK_MODEL_PATH.exists():
                raise FileNotFoundError(
                    "Vosk model not found at %s. Download "
                    "'vosk-model-small-cn-0.22' and extract it there."
                    % config.VOSK_MODEL_PATH
                )
            SetLogLevel(-1)
            self._model_cache[path] = Model(path)
        self._model = self._model_cache[path]

    # ------------------------------------------------------------------ #
    # offline full-clip transcription
    # ------------------------------------------------------------------ #
    def transcribe(self, samples: np.ndarray) -> ASRResult:
        from vosk import KaldiRecognizer

        samples = np.asarray(samples, dtype=np.float32)
        rec = KaldiRecognizer(self._model, self.sr)
        rec.SetWords(True)
        rec.SetMaxAlternatives(0)
        step = self.sr  # 1 s chunks keep memory bounded
        for i in range(0, max(samples.size, 1), step):
            rec.AcceptWaveform((samples[i:i + step] * 32768).astype(np.int16).tobytes())
        final = json.loads(rec.FinalResult())
        return self._parse(final)

    # ------------------------------------------------------------------ #
    # streaming helpers (used by the live pipeline)
    # ------------------------------------------------------------------ #
    def new_stream(self):
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, self.sr)
        rec.SetWords(True)
        rec.SetMaxAlternatives(0)
        return rec

    @staticmethod
    def feed(rec, block: np.ndarray):
        return json.loads(rec.AcceptWaveform(
            (np.asarray(block, dtype=np.float32) * 32768).astype(np.int16).tobytes()
        ))

    @staticmethod
    def finalize(rec) -> ASRResult:
        return VoskRecognizer._parse(json.loads(rec.FinalResult()))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse(payload: dict) -> ASRResult:
        tokens: list[Token] = []
        for w in payload.get("result", []):
            tokens.append(Token(
                text=str(w.get("word", "")).strip(),
                start=float(w.get("start", 0.0)),
                end=float(w.get("end", 0.0)),
                conf=float(w.get("conf", 0.0)),
            ))
        return ASRResult(text=payload.get("text", "").strip(), tokens=tokens)
