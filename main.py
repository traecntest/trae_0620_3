"""BabyVoice Museum - application entry point.

Launches the PySide6 desktop GUI that wires together:

* audio  : real-time denoising (RNNoise / HF-preserving spectral) + recording
* asr    : offline Chinese Vosk recognition with word-level timestamps
* emotion: rule-engine crying/laughing/babbling classification + milestones
* storage: SQLite metadata + ``year/month/date_tag.wav`` file layout
* gui    : waveform visualisation, tag drag-drop editing, one-click export

Run::

    pip install -r requirements.txt
    python main.py
"""
from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication

    from src.gui.main_window import MainWindow
    from src.pipeline import ClipProcessor
    from src.storage.database import Database
    from src.storage.file_manager import FileManager

    app = QApplication(sys.argv)
    app.setApplicationName("BabyVoice Museum")

    db = Database()
    files = FileManager()
    processor = ClipProcessor(db=db, files=files)

    window = MainWindow(db=db, files=files, processor=processor)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
