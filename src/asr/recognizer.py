"""Offline Chinese ASR with word-level timestamps (Vosk).

Vosk ships small CN models that run fully offline and emit per-token
start/end times.  The small-cn model tokenises Chinese as single characters,
so :class:`KeywordSpotter` joins consecutive tokens back into words before
matching ``抱抱`` etc.

If the model is missing the :func:`ensure_vosk_model` helper can either
download it straight from Alpha Cephei's official mirror, or point the user
to the exact URL.
"""
from __future__ import annotations

import io
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import config


class VoskModelMissingError(FileNotFoundError):
    """Raised when the Vosk model directory cannot be located."""

    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        urls = "\n  * ".join(config.VOSK_MODEL_DOWNLOAD_URLS)
        super().__init__(
            "未找到 Vosk 中文离线模型。\n"
            f"期望路径: {self.model_path}\n"
            f"一键下载命令:\n"
            f"  python -c \"import sys; sys.path.insert(0,'.'); "
            f"from src.asr.recognizer import ensure_vosk_model; "
            f"ensure_vosk_model(force=False)\"\n"
            f"手动下载地址:\n  * {urls}\n"
            f"模型列表: {config.VOSK_MODEL_HOME_URL}\n"
            f"请将压缩包解压后放置到: {config.VOSK_MODEL_PATH.parent}"
        )


def ensure_vosk_model(
    force: bool = False,
    model_path: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    timeout_s: int = 60,
) -> Path:
    """Download and extract the Chinese Vosk model if not already present.

    Parameters
    ----------
    force:
        Re-download even if the model directory already exists.
    model_path:
        Target extraction directory (defaults to ``config.VOSK_MODEL_PATH``).
    progress_cb:
        ``callback(downloaded_bytes, total_bytes)`` invoked periodically.
        ``total_bytes`` is ``-1`` when the server does not send Content-Length.
    timeout_s:
        Per-request HTTP timeout.

    Returns
    -------
    Path
        The absolute model directory (valid for ``vosk.Model(str(path))``).
    """
    target = Path(model_path or config.VOSK_MODEL_PATH)
    manifest = target / "README"
    if not force and (manifest.exists() or (target / "conf").exists()):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Optional[Exception] = None
    for url in config.VOSK_MODEL_DOWNLOAD_URLS:
        try:
            return _download_and_extract(url, target, progress_cb, timeout_s)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    raise RuntimeError(
        f"Vosk 模型下载失败（已尝试 {len(config.VOSK_MODEL_DOWNLOAD_URLS)} 个镜像），"
        f"请手动下载: {config.VOSK_MODEL_DOWNLOAD_URLS[0]}\n"
        f"最后一次错误: {last_exc!r}"
    )


def _download_and_extract(
    url: str,
    target: Path,
    progress_cb: Optional[Callable[[int, int], None]],
    timeout_s: int,
) -> Path:
    request = urllib.request.Request(url, headers={"User-Agent": "BabyVoiceMuseum/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as resp:
        total = int(resp.headers.get("Content-Length", "-1"))
        data = bytearray()
        chunk = 64 * 1024
        downloaded = 0
        while True:
            block = resp.read(chunk)
            if not block:
                break
            data.extend(block)
            downloaded += len(block)
            if progress_cb is not None:
                try:
                    progress_cb(downloaded, total)
                except Exception:
                    pass

    tmp_dir = target.with_name(target.name + ".downloading")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    with zipfile.ZipFile(io.BytesIO(bytes(data))) as zf:
        # The zip wraps a single top-level folder, e.g. vosk-model-small-cn-0.22/.
        # Extract to tmp_dir, then move that subfolder to target.
        zf.extractall(tmp_dir)
    candidates = [p for p in tmp_dir.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        source = candidates[0]
    else:
        # fallback: flat extract means tmp_dir itself is the model dir
        source = tmp_dir

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), str(target))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return target


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

    @staticmethod
    def is_available() -> bool:
        """True if both the `vosk` Python package *and* the model directory
        exist (no loading yet, so very cheap)."""
        try:
            import vosk  # noqa: F401
        except Exception:
            return False
        return bool(config.VOSK_MODEL_PATH.exists())

    def __init__(self, model_path: Optional[str] = None, sr: int = config.SAMPLE_RATE):
        from vosk import Model, SetLogLevel

        self.sr = sr
        target_dir = Path(model_path) if model_path else config.VOSK_MODEL_PATH
        path = str(target_dir)
        if path not in self._model_cache:
            if not target_dir.exists():
                raise VoskModelMissingError(target_dir)
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
