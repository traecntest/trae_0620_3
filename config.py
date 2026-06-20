"""Central configuration for BabyVoice Museum.

Every tunable constant lives here so that audio / ASR / emotion / storage /
GUI modules share a single source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("BVM_DATA_ROOT", APP_ROOT / "data"))
AUDIO_ROOT = DATA_ROOT / "audio"
DB_PATH = DATA_ROOT / "babyvoice.db"
EXPORT_ROOT = DATA_ROOT / "exports"

# Vosk model directory (offline Chinese model).  The user downloads
# `vosk-model-small-cn-0.22` (or a larger one) and extracts it here.
VOSK_MODEL_NAME = os.environ.get("BVM_VOSK_MODEL_NAME", "vosk-model-small-cn-0.22")
VOSK_MODEL_PATH = Path(os.environ.get(
    "BVM_VOSK_MODEL",
    DATA_ROOT / "models" / VOSK_MODEL_NAME,
))

# Vosk model download URLs (in priority order; fall back on failure).
# NOTE: Only verified-valid URLs are listed here.  More mirrors can be found
# at VOSK_MODEL_HOME_URL (the official Vosk models page).
VOSK_MODEL_DOWNLOAD_URLS: list[str] = [
    "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip",
]
VOSK_MODEL_HOME_URL = "https://alphacephei.com/vosk/models"

for _p in (DATA_ROOT, AUDIO_ROOT, EXPORT_ROOT):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Audio / DSP
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 16000          # 16 kHz matches Vosk & RNNoise frame expectations
CHANNELS = 1
BLOCK_MS = 30               # ~30 ms blocks -> realtime denoising granularity
BLOCK_FRAMES = int(SAMPLE_RATE * BLOCK_MS / 1000)   # 480 frames @16 kHz

# Denoiser: baby vowels carry strong harmonics above 4 kHz that we must keep.
# High-frequency preservation band (Hz) whose gain is protected during gating.
HF_PROTECT_HZ = 4000.0
NOISE_PROFILE_FRAMES = int(SAMPLE_RATE * 0.5)  # 0.5 s noise estimate

# Acoustic feature extraction
F0_FMIN = 80.0
F0_FMAX = 600.0             # baby F0 commonly reaches 500+ Hz
N_MFCC = 13
HOP_LENGTH = 512
N_FFT = 2048

# --------------------------------------------------------------------------- #
# Emotion rule engine thresholds (tunable, derived from child-voice stats)
# --------------------------------------------------------------------------- #
CRY_F0_MIN = 350.0           # crying -> sustained high pitch
CRY_ENERGY_DB = -22.0        # RMS energy in dB above which it is "loud"
LAUGH_PERIOD_S = 0.18       # rhythmic ~5-6 Hz amplitude modulation
LAUGH_PERIOD_TOL = 0.06
BABBLE_VOICED_RATIO_MAX = 0.45   # babbling is intermittently voiced

# --------------------------------------------------------------------------- #
# Keyword / milestone vocabulary (Chinese)
# --------------------------------------------------------------------------- #
# Recognised target words; first occurrence of any -> Milestone tag.
TARGET_WORDS = [
    "抱抱", "妈妈", "爸爸", "奶奶", "爷爷",
    "吃", "喝", "要", "不", "好",
    "狗狗", "猫猫", "车车", "球", "拜拜",
]

# Words that should trigger a "Milestone" the very first time they are heard.
MILESTONE_WORDS = TARGET_WORDS

# --------------------------------------------------------------------------- #
# Storage layout
# --------------------------------------------------------------------------- #
AUDIO_FILENAME_FMT = "{date}_{tag}.wav"   # within year/month folders
TAG_MAX_LEN = 24
SAFE_TAG_DEFAULT = "untagged"
