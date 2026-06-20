"""Real-time audio capture with inline denoising.

:class:`AudioRecorder` wraps a :mod:`sounddevice` input stream, denoises each
incoming block on the fly and exposes:
* the growing denoised buffer (for waveform visualisation), and
* a per-block callback (for live VU / partial ASR feeding).
"""
from __future__ import annotations

import threading
from typing import Callable, Optional

import numpy as np

import config
from src.audio.denoiser import Denoiser

BlockCallback = Callable[[np.ndarray], None]


class AudioRecorder:
    def __init__(
        self,
        denoiser: Optional[Denoiser] = None,
        sr: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        blocksize: int = config.BLOCK_FRAMES,
        on_block: Optional[BlockCallback] = None,
    ):
        import sounddevice as sd  # imported lazily so the GUI can start w/o audio hw

        self._sd = sd
        self.sr = sr
        self.channels = channels
        self.blocksize = blocksize
        self.denoiser = denoiser or Denoiser(sr=sr)
        self.on_block = on_block

        self._lock = threading.Lock()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._raw = np.zeros(0, dtype=np.float32)
        self._stream = None
        self._running = False

    # ------------------------------------------------------------------ #
    # stream lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._running:
            return
        self.denoiser.reset()
        self._buffer = np.zeros(0, dtype=np.float32)
        self._raw = np.zeros(0, dtype=np.float32)
        self._stream = self._sd.InputStream(
            samplerate=self.sr,
            channels=self.channels,
            blocksize=self.blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self) -> np.ndarray:
        if not self._running:
            with self._lock:
                return self._buffer.copy()
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._running = False
        with self._lock:
            return self._buffer.copy()

    @property
    def running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    # buffer access
    # ------------------------------------------------------------------ #
    def get_buffer(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()

    def clear(self) -> None:
        with self._lock:
            self._buffer = np.zeros(0, dtype=np.float32)
            self._raw = np.zeros(0, dtype=np.float32)
        self.denoiser.reset()

    # ------------------------------------------------------------------ #
    # sounddevice callback
    # ------------------------------------------------------------------ #
    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        block = indata[:, 0].copy() if self.channels > 1 else indata.copy()
        block = block.astype(np.float32, copy=False)
        try:
            clean = self.denoiser.denoise_block(block)
        except Exception:
            clean = block
        with self._lock:
            self._raw = np.concatenate([self._raw, block])
            self._buffer = np.concatenate([self._buffer, clean])
        if self.on_block is not None:
            try:
                self.on_block(clean)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# WAV helpers
# --------------------------------------------------------------------------- #
def save_wav(path, samples: np.ndarray, sr: int = config.SAMPLE_RATE) -> str:
    import soundfile as sf

    samples = np.asarray(samples, dtype=np.float32)
    sf.write(str(path), samples, sr, subtype="PCM_16")
    return str(path)


def load_wav(path, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if file_sr != sr and data.size:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32, copy=False)
