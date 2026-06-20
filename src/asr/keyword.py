"""Keyword / wake-word spotting over a Vosk transcription.

The small CN Vosk model emits one token per Chinese character, each with its
own timestamp.  To recognise a multi-character target such as ``抱抱`` we
rejoin consecutive characters into a string, remember which character came
from which token, and search for the target substring.  The hit's timestamp
is then ``start`` of the first token and ``end`` of the last token covered by
the match -> precise, frame-aligned timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass

import config
from src.asr.recognizer import ASRResult, Token


@dataclass
class KeywordHit:
    word: str
    start: float        # seconds, absolute within the clip
    end: float
    conf: float


class KeywordSpotter:
    def __init__(self, words: list[str] | None = None):
        self.words = [w for w in (words or config.TARGET_WORDS) if w]

    # ------------------------------------------------------------------ #
    def find(self, result: ASRResult) -> list[KeywordHit]:
        """Return all keyword occurrences with timestamps."""
        if not result.tokens:
            return []
        hits: list[KeywordHit] = []
        # build joined text and char->token index map
        joined = ""
        char_token: list[int] = []
        token_chars: list[tuple[int, int]] = []  # (char_start, char_end) per token
        for idx, tok in enumerate(result.tokens):
            t = tok.text
            s = len(joined)
            joined += t
            token_chars.append((s, len(joined)))
            for _ in t:
                char_token.append(idx)

        for word in self.words:
            if not word:
                continue
            start = 0
            while True:
                pos = joined.find(word, start)
                if pos < 0:
                    break
                first_tok = char_token[pos] if pos < len(char_token) else 0
                last_char = pos + len(word) - 1
                last_tok = char_token[last_char] if last_char < len(char_token) else result.tokens[-1]  # noqa: E501
                ft: Token = result.tokens[first_tok]
                lt: Token = result.tokens[last_tok]
                hits.append(KeywordHit(
                    word=word,
                    start=ft.start,
                    end=max(lt.end, lt.start + 0.05),
                    conf=min(ft.conf, lt.conf) if (ft.conf and lt.conf) else max(ft.conf, lt.conf),
                ))
                start = pos + len(word)
        hits.sort(key=lambda h: h.start)
        return hits

    # ------------------------------------------------------------------ #
    def first_occurrence(self, result: ASRResult) -> KeywordHit | None:
        hits = self.find(result)
        return hits[0] if hits else None

    def contains(self, result: ASRResult, word: str) -> KeywordHit | None:
        for h in self.find(result):
            if h.word == word:
                return h
        return None
