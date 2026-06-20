"""Acoustic feature extraction: F0, MFCC and short-time energy.

These features feed the emotion rule engine and the milestone judge.
Everything is computed offline over a full clip for accuracy, but the
function is cheap enough (~tens of ms for a few seconds of audio) to call
once per recorded segment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class AcousticFeatures:
    """Container for the features consumed by the emotion rule engine."""
    f0_mean: float          # mean fundamental frequency over voiced frames
    f0_std: float           # pitch variability
    voiced_ratio: float     # fraction of frames that are voiced
    energy_mean_db: float   # mean short-time energy in dB
    energy_std_db: float    # energy variability
    mfcc_mean: np.ndarray = field(default_factory=lambda: np.zeros(config.N_MFCC))
    # rhythmic modulation peak around the laugh rate (0..1 confidence)
    laugh_modulation: float = 0.0
    duration: float = 0.0   # seconds


def extract_features(samples: np.ndarray, sr: int = config.SAMPLE_RATE) -> AcousticFeatures:
    """Extract F0 / MFCC / energy + laugh-modulation from a mono clip."""
    import librosa

    samples = np.asarray(samples, dtype=np.float32)
    n = samples.size
    duration = n / sr if sr else 0.0

    if n < config.N_FFT:
        return AcousticFeatures(
            f0_mean=0.0, f0_std=0.0, voiced_ratio=0.0,
            energy_mean_db=-80.0, energy_std_db=0.0, duration=duration,
        )

    # --- F0 via probabilistic YIN ------------------------------------- #
    f0, voiced_flag, _ = librosa.pyin(
        samples,
        fmin=config.F0_FMIN,
        fmax=config.F0_FMAX,
        sr=sr,
        frame_length=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        fill_na=np.nan,
    )
    voiced = voiced_flag.astype(bool)
    f0_voiced = f0[voiced]
    f0_mean = float(np.nanmean(f0_voiced)) if f0_voiced.size else 0.0
    f0_std = float(np.nanstd(f0_voiced)) if f0_voiced.size else 0.0
    voiced_ratio = float(voiced.mean())

    # --- short-time energy (RMS -> dB) ------------------------------- #
    rms = librosa.feature.rms(
        y=samples, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH
    )[0]
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    energy_mean_db = float(np.mean(rms_db))
    energy_std_db = float(np.std(rms_db))

    # --- MFCC -------------------------------------------------------- #
    mfcc = librosa.feature.mfcc(
        y=samples, sr=sr, n_mfcc=config.N_MFCC,
        hop_length=config.HOP_LENGTH, n_fft=config.N_FFT,
    )
    mfcc_mean = mfcc.mean(axis=1).astype(np.float32)

    # --- laugh modulation: peak of the amplitude envelope spectrum --- #
    laugh_mod = _laugh_modulation(rms, sr)

    return AcousticFeatures(
        f0_mean=f0_mean,
        f0_std=f0_std,
        voiced_ratio=voiced_ratio,
        energy_mean_db=energy_mean_db,
        energy_std_db=energy_std_db,
        mfcc_mean=mfcc_mean,
        laugh_modulation=laugh_mod,
        duration=duration,
    )


def _laugh_modulation(rms: np.ndarray, sr: int) -> float:
    """Confidence that the RMS envelope pulses at the laugh rate (~5-6 Hz).

    Laughs produce quasi-periodic amplitude modulation; we look for spectral
    energy near ``1/LAUGH_PERIOD_S`` Hz and normalise by total power.
    """
    if rms.size < 8:
        return 0.0
    env = rms - rms.mean()
    if env.max() == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(env.size, d=config.HOP_LENGTH / sr)
    target = 1.0 / config.LAUGH_PERIOD_S
    tol = 1.0 / config.LAUGH_PERIOD_TOL
    band = (freqs >= target - tol) & (freqs <= target + tol)
    band_power = spec[band].sum()
    total = spec.sum() + 1e-10
    return float(np.clip(band_power / total, 0.0, 1.0))
