"""Real-time audio denoising for BabyVoice Museum.

Design goals
------------
1. *Lightweight deep-learning* denoising via RNNoise when the compiled
   binding is available (``pip install rnnoise``).
2. A *self-contained* high-frequency-preserving spectral-gating fallback so
   the app always works on a fresh Windows install.
3. **Preserve baby's high-frequency overtones** (the >4 kHz harmonic content
   that carries infant timbre).  The HF band gain is protected so that
   spectral gating never mutes the very cues the emotion/milestone engine
   relies on.

Both backends expose the same :class:`Denoiser` facade::

    denoiser = Denoiser(sr=16000)
    clean = denoiser.denoise(noisy_float32)
    # or block-by-block for streaming:
    clean_block = denoiser.denoise_block(block_float32)
"""
from __future__ import annotations

import numpy as np

import config

# --------------------------------------------------------------------------- #
# Optional RNNoise deep-learning backend
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on a compiled native binding
    from rnnoise import RNNoise as _RNNoise  # type: ignore

    _HAS_RNNOISE = True
except Exception:  # pragma: no cover
    _RNNoise = None  # type: ignore
    _HAS_RNNOISE = False


class RNNoiseBackend:
    """Thin wrapper around the RNNoise recurrent-network denoiser.

    RNNoise operates on 10 ms frames (480 samples at 48 kHz).  We resample
    16 kHz <-> 48 kHz so the rest of the pipeline can stay at Vosk's rate.
    """

    RNNOISE_SR = 48000
    FRAME = 480  # 10 ms @ 48 kHz

    def __init__(self, sr: int = config.SAMPLE_RATE):
        if not _HAS_RNNOISE:
            raise RuntimeError("rnnoise binding not available")
        self._sr = sr
        self._den = _RNNoise()

    def denoise(self, x: np.ndarray) -> np.ndarray:
        if x.size == 0:
            return x.astype(np.float32, copy=False)
        x = np.asarray(x, dtype=np.float32)
        # resample 16k -> 48k (linear is fine for denoising)
        ratio = self.RNNOISE_SR / self._sr
        n48 = int(round(len(x) * ratio))
        x48 = _resample_linear(x, n48)
        # pad to a multiple of FRAME
        pad = (-x48.size) % self.FRAME
        if pad:
            x48 = np.concatenate([x48, np.zeros(pad, dtype=np.float32)])
        # process frame by frame
        out48 = np.empty_like(x48)
        for i in range(0, x48.size, self.FRAME):
            chunk = x48[i:i + self.FRAME]
            out48[i:i + self.FRAME] = self._den.filter(chunk)
        # back to original length / rate
        out = _resample_linear(out48[: n48 - pad if pad else n48], len(x))
        return out.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# HF-preserving spectral gating fallback (always available)
# --------------------------------------------------------------------------- #
class SpectralDenoiser:
    """STFT spectral subtraction that protects the high-frequency band.

    Parameters
    ----------
    sr:
        Sample rate.
    n_fft / hop:
        STFT windowing.
    noise_frames:
        Number of leading frames used to estimate the stationary noise
        floor (assumed to be room tone before the baby vocalises).
    hf_protect_hz:
        Lower edge of the protected HF band.  Bins above this keep a larger
        fraction of their original energy so infant overtones survive.
    hf_keep:
        Minimum retained gain in the protected band (0..1).  ``0.75`` keeps
        75% of the HF energy even when the gating logic would suppress it.
    """

    def __init__(
        self,
        sr: int = config.SAMPLE_RATE,
        n_fft: int = config.N_FFT,
        hop: int = config.HOP_LENGTH,
        noise_frames: int = 16,
        hf_protect_hz: float = config.HF_PROTECT_HZ,
        hf_keep: float = 0.75,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.hf_protect_hz = hf_protect_hz
        self.hf_keep = hf_keep
        self._win = np.hanning(n_fft + 1)[:-1].astype(np.float32)
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)
        self._hf_mask = (freqs >= hf_protect_hz).astype(np.float32)
        self._noise_ps: np.ndarray | None = None
        self._noise_frames = noise_frames

    # -- internals -------------------------------------------------------- #
    def _stft(self, x: np.ndarray) -> np.ndarray:
        n = self.n_fft
        hop = self.hop
        if x.size < n:
            x = np.pad(x, (0, n - x.size))
        n_frames = 1 + (x.size - n) // hop
        frames = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, n),
            strides=(x.strides[0] * hop, x.strides[0]),
        )
        return np.fft.rfft(frames * self._win, axis=1).astype(np.complex64)

    def _istft(self, spec: np.ndarray, length: int) -> np.ndarray:
        n_frames, _n_bins = spec.shape
        n = self.n_fft
        hop = self.hop
        out = np.zeros(length, dtype=np.float32)
        norm = np.zeros(length, dtype=np.float32)
        frames = np.fft.irfft(spec, n=n, axis=1).astype(np.float32) * self._win
        for i in range(n_frames):
            start = i * hop
            stop = min(start + n, length)
            k = stop - start
            out[start:stop] += frames[i, :k]
            norm[start:stop] += (self._win ** 2)[:k]
        norm[norm == 0] = 1.0
        return out / norm

    def _estimate_noise(self, spec: np.ndarray) -> np.ndarray:
        mag = np.abs(spec) ** 2
        if spec.shape[0] >= self._noise_frames:
            return mag[: self._noise_frames].mean(axis=0)
        # not enough frames -> use a low percentile of the whole clip
        return np.percentile(mag, 20, axis=0)

    def _gain(self, sig_ps: np.ndarray, noise_ps: np.ndarray) -> np.ndarray:
        # Wiener-style gain with a floor, computed in the power domain.
        snr = sig_ps / (noise_ps + 1e-10)
        gain = np.clip(snr / (snr + 1.0), 1e-3, 1.0)
        # protect the HF band -> keep at least ``hf_keep`` of the energy there
        gain = gain * (1 - self._hf_mask) + np.maximum(gain, self.hf_keep) * self._hf_mask
        return np.sqrt(gain).astype(np.float32)

    # -- public API ------------------------------------------------------- #
    def denoise(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.size < self.n_fft:
            return x
        spec = self._stft(x)
        if self._noise_ps is None:
            self._noise_ps = self._estimate_noise(spec)
        gain = self._gain(np.abs(spec) ** 2, self._noise_ps)
        clean = spec * gain
        out = self._istft(clean, len(x))[: x.size].astype(np.float32)
        # OLA / windowing gains can distort amplitude; rescale to match input
        # RMS so downstream feature extraction (energy / MFCC) stays correct.
        in_rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        out_rms = float(np.sqrt(np.mean(out ** 2) + 1e-12))
        if out_rms > 1e-6:
            out = out * (in_rms / out_rms)
        return out

    def feed_noise_profile(self, x: np.ndarray) -> None:
        """Prime the noise floor from a silent-room sample."""
        x = np.asarray(x, dtype=np.float32)
        if x.size < self.n_fft:
            return
        self._noise_ps = self._estimate_noise(self._stft(x))

    def reset(self) -> None:
        self._noise_ps = None


# --------------------------------------------------------------------------- #
# Public facade
# --------------------------------------------------------------------------- #
class Denoiser:
    """Pick the best available backend and expose a uniform API.

    ``backend="auto"`` prefers RNNoise, falling back to the HF-preserving
    spectral denoiser when the native binding is missing.
    """

    def __init__(self, sr: int = config.SAMPLE_RATE, backend: str = "auto"):
        self.sr = sr
        self.backend_name: str
        if backend in ("auto", "rnnoise") and _HAS_RNNOISE:
            try:
                self._impl = RNNoiseBackend(sr)
                self.backend_name = "rnnoise"
            except Exception:
                self._impl = SpectralDenoiser(sr)
                self.backend_name = "spectral"
        elif backend == "rnnoise" and not _HAS_RNNOISE:
            raise RuntimeError(
                "RNNoise backend requested but the 'rnnoise' package is not "
                "installed. Install it or use backend='spectral'."
            )
        else:
            self._impl = SpectralDenoiser(sr)
            self.backend_name = "spectral"

    def denoise(self, x: np.ndarray) -> np.ndarray:
        return self._impl.denoise(x)

    def denoise_block(self, block: np.ndarray) -> np.ndarray:
        """Denoise a single streaming block (real-time safe, O(block))."""
        return self.denoise(block)

    def feed_noise_profile(self, x: np.ndarray) -> None:
        if isinstance(self._impl, SpectralDenoiser):
            self._impl.feed_noise_profile(x)

    def reset(self) -> None:
        if isinstance(self._impl, SpectralDenoiser):
            self._impl.reset()

    @staticmethod
    def available_backends() -> list[str]:
        names = ["spectral"]
        if _HAS_RNNOISE:
            names.append("rnnoise")
        return names


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _resample_linear(x: np.ndarray, n_out: int) -> np.ndarray:
    if n_out <= 1:
        return np.zeros(max(n_out, 0), dtype=np.float32)
    if x.size == 0:
        return np.zeros(n_out, dtype=np.float32)
    idx = np.linspace(0, x.size - 1, n_out, dtype=np.float32)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x.size - 1)
    frac = (idx - i0).astype(np.float32)
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)
