"""Main application window (PySide6).

Layout
------
* toolbar  : record / stop / import / export / denoiser backend / ASR toggle
* left     : search + clips table (drop a tag onto a row to re-tag it)
* center   : waveform + emotion / feature / transcript read-outs
* right    : draggable tag palette + milestone log

Recording is captured on the sounddevice thread; a 30 ms QTimer polls the
recorder buffer for the live waveform.  Heavy work (features + ASR + DB)
runs on a QThread so the UI never stalls.
"""
from __future__ import annotations

from datetime import datetime
import threading
from typing import Optional

import numpy as np

import config
from src.audio.recorder import AudioRecorder, load_wav
from src.audio.denoiser import Denoiser
from src.gui.waveform_widget import WaveformWidget
from src.pipeline import ClipProcessor, ProcessOutcome
from src.storage.database import Database
from src.storage.exporter import export_clips
from src.storage.file_manager import FileManager, sanitize_tag
from src.emotion.analyzer import EmotionAnalyzer

TAG_MIME = "application/x-bvm-tag"
_EMOTION_COLORS = {
    "crying": "#f72585", "laughing": "#ffd60a", "babbling": "#4cc9f0",
    "talking": "#06d6a0", "neutral": "#8d99ae",
}
_EMOTION_ZH = {
    "crying": "哭泣", "laughing": "大笑", "babbling": "咿呀学语",
    "talking": "说话", "neutral": "安静",
}


# --------------------------------------------------------------------------- #
# processing worker
# --------------------------------------------------------------------------- #
from PySide6.QtCore import Qt, QThread, Signal, QObject, QTimer  # noqa: E402


class ProcessWorker(QObject):
    finished = Signal(object)   # ProcessOutcome or Exception

    def __init__(self, processor: ClipProcessor, samples: np.ndarray,
                 tag: str, run_asr: bool):
        super().__init__()
        self.processor = processor
        self.samples = samples
        self.tag = tag
        self.run_asr = run_asr

    def run(self) -> None:
        try:
            outcome = self.processor.process(
                self.samples, datetime.now(), self.tag, run_asr=self.run_asr
            )
            self.finished.emit(outcome)
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(exc)


# --------------------------------------------------------------------------- #
# model download worker (background thread so GUI stays alive)
# --------------------------------------------------------------------------- #
class ModelDownloadWorker(QObject):
    progress = Signal(int, int)     # downloaded, total (total=-1 when unknown)
    finished_ok = Signal(object)    # resulting Path
    failed = Signal(str)

    def __init__(self, force: bool = False):
        super().__init__()
        self.force = force

    def run(self) -> None:
        from src.asr.recognizer import ensure_vosk_model
        try:
            path = ensure_vosk_model(
                force=self.force,
                progress_cb=lambda done, total: self.progress.emit(int(done), int(total)),
                timeout_s=120,
            )
            self.finished_ok.emit(path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# clips table with tag drag-drop
# --------------------------------------------------------------------------- #
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSizePolicy, QSlider, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QWidgetAction,
)


class ClipsTable(QTableWidget):
    """Table whose rows accept dropped tags to re-tag a clip."""

    def __init__(self, on_drop_tag):
        super().__init__(0, 7)
        self.on_drop_tag = on_drop_tag
        self.setHorizontalHeaderLabels(
            ["日期", "标签", "情绪", "时长(s)", "F0(Hz)", "转写", "里程碑"]
        )
        self.verticalHeader().setVisible(False)
        self.setAcceptDrops(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.DoubleClicked
                             | QAbstractItemView.EditKeyPressed)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

    # -- drag & drop ----------------------------------------------------- #
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat(TAG_MIME) or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(TAG_MIME) or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        mime = event.mimeData()
        if mime.hasFormat(TAG_MIME):
            tag = bytes(mime.data(TAG_MIME)).decode("utf-8")
        elif mime.hasText():
            tag = mime.text()
        else:
            event.ignore()
            return
        row = self.indexAt(event.position().toPoint() if hasattr(event, "position") else event.pos()).row()
        if row < 0:
            event.ignore()
            return
        clip_id_item = self.item(row, 0)
        clip_id = clip_id_item.data(Qt.UserRole) if clip_id_item else None
        if clip_id is not None:
            self.on_drop_tag(clip_id, tag)
        event.acceptProposedAction()

    # -- inline tag rename ---------------------------------------------- #
    def commit_tag_edit(self, row: int, new_text: str) -> None:
        clip_id = self.item(row, 0).data(Qt.UserRole)
        self.on_drop_tag(clip_id, new_text)


# --------------------------------------------------------------------------- #
# tag palette (drag source)
# --------------------------------------------------------------------------- #
class TagPalette(QListWidget):
    def __init__(self, tags):
        super().__init__()
        self.setDragEnabled(True)
        self.setDefaultDropAction(Qt.CopyAction)
        for t in tags:
            self.add_tag(t)

    def add_tag(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setData(Qt.UserRole, text)
        self.addItem(item)

    def startDrag(self, supportedActions):
        from PySide6.QtCore import QMimeData, QByteArray
        from PySide6.QtGui import QDrag

        item = self.currentItem()
        if item is None:
            return
        tag = item.data(Qt.UserRole)
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(TAG_MIME, QByteArray(tag.encode("utf-8")))
        mime.setText(tag)
        drag.setMimeData(mime)
        drag.exec_(Qt.CopyAction)


# --------------------------------------------------------------------------- #
# main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self, db: Database, files: FileManager, processor: ClipProcessor):
        super().__init__()
        self.db = db
        self.files = files
        self.processor = processor
        self.recorder: Optional[AudioRecorder] = None
        self.denoiser = Denoiser(backend="auto")
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[ProcessWorker] = None
        self._run_asr = processor.asr_available()

        self.setWindowTitle("BabyVoice Museum — 婴儿声音成长档案")
        self.resize(1280, 820)

        self._build_toolbar()
        self._build_central()
        self._build_status()
        self._refresh_clips()
        self._refresh_milestones()

        # playback state
        self._play_samples: np.ndarray | None = None
        self._play_sr: int = config.SAMPLE_RATE
        self._play_stream = None
        self._play_pos: int = 0          # frames
        self._play_lock = threading.Lock()
        self._play_timer: QTimer | None = None

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_toolbar(self) -> None:
        tb = self.addToolBar("主工具栏")
        tb.setMovable(False)

        self.act_record = QPushButton("● 开始录音")
        self.act_record.clicked.connect(self.toggle_recording)
        tb.addWidget(self.act_record)

        act_import = QPushButton("导入 WAV")
        act_import.clicked.connect(self.import_wav)
        tb.addWidget(act_import)

        act_export_sel = QPushButton("导出选中")
        act_export_sel.clicked.connect(lambda: self.export(selected=True))
        tb.addWidget(act_export_sel)

        act_export_all = QPushButton("导出全部")
        act_export_all.clicked.connect(lambda: self.export(selected=False))
        tb.addWidget(act_export_all)

        tb.addSeparator()
        from PySide6.QtWidgets import QCheckBox, QComboBox, QLabel as QL
        self.cb_asr = QCheckBox("启用语音识别")
        self.cb_asr.setChecked(self._run_asr)
        self.cb_asr.toggled.connect(self._on_asr_toggle)
        tb.addWidget(self.cb_asr)

        tb.addWidget(QL("  降噪后端:"))
        self.cb_denoise = QComboBox()
        self.cb_denoise.addItems(Denoiser.available_backends())
        self.cb_denoise.setCurrentText(self.denoiser.backend_name)
        self.cb_denoise.currentTextChanged.connect(self._on_denoise_backend)
        tb.addWidget(self.cb_denoise)

        tb.addSeparator()
        info = QLabel(f"  音频: {config.SAMPLE_RATE}Hz | F0 {config.F0_FMIN}-{config.F0_FMAX}Hz | HF保护>{int(config.HF_PROTECT_HZ)}Hz")
        tb.addWidget(info)

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Horizontal)

        # ---- left: search + clips table ----
        left = QWidget()
        llay = QVBoxLayout(left)
        from PySide6.QtWidgets import QLineEdit, QPushButton
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索标签/关键词…")
        self.search.textChanged.connect(self._refresh_clips)
        llay.addWidget(self.search)
        filter_row = QHBoxLayout()
        self.btn_milestone_only = QPushButton("仅看里程碑")
        self.btn_milestone_only.setCheckable(True)
        self.btn_milestone_only.toggled.connect(self._refresh_clips)
        filter_row.addWidget(self.btn_milestone_only)
        filter_row.addStretch(1)
        llay.addLayout(filter_row)
        self.table = ClipsTable(self._apply_tag_to_clip)
        self.table.itemSelectionChanged.connect(self._on_clip_selected)
        llay.addWidget(self.table)
        splitter.addWidget(left)

        # ---- center: playback controls + waveform + panels ----
        center = QWidget()
        clay = QVBoxLayout(center)

        # playback controls
        self.play_bar = QHBoxLayout()
        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.clicked.connect(self._play_toggle)
        self.btn_play.setEnabled(False)
        self.play_bar.addWidget(self.btn_play)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.clicked.connect(self._play_stop)
        self.btn_stop.setEnabled(False)
        self.play_bar.addWidget(self.btn_stop)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.play_bar.addWidget(self.lbl_time)

        self.play_slider = QSlider(Qt.Horizontal)
        self.play_slider.setRange(0, 1000)
        self.play_slider.setValue(0)
        self.play_slider.setEnabled(False)
        self.play_slider.sliderMoved.connect(self._play_seek)
        self.play_bar.addWidget(self.play_slider, 1)

        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(80)
        self.vol_slider.setToolTip("音量")
        self.vol_slider.setFixedWidth(90)
        self.play_bar.addWidget(self.vol_slider)

        clay.addLayout(self.play_bar)

        self.wave = WaveformWidget()
        clay.addWidget(self.wave.widget(), 3)
        self.lbl_emotion = QLabel("情绪: —")
        self.lbl_emotion.setStyleSheet("font-size:15pt;font-weight:bold;")
        self.lbl_features = QLabel("特征: —")
        self.lbl_transcript = QLabel("转写: —")
        self.lbl_transcript.setWordWrap(True)
        self.lbl_keywords = QLabel("关键词时间戳: —")
        self.lbl_keywords.setWordWrap(True)
        for w in (self.lbl_emotion, self.lbl_features, self.lbl_transcript, self.lbl_keywords):
            clay.addWidget(w)
        clay.addStretch(1)
        splitter.addWidget(center)

        # ---- right: tag palette + milestones ----
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addWidget(QLabel("标签面板 (拖到左侧行可改标签)"))
        self.palette = TagPalette(list(config.TARGET_WORDS) + ["哭泣", "大笑", "咿呀", "说话", "未标记"])
        rlay.addWidget(self.palette)
        from PySide6.QtWidgets import QPushButton
        btn_add_tag = QPushButton("+ 新建标签")
        btn_add_tag.clicked.connect(self._add_custom_tag)
        rlay.addWidget(btn_add_tag)
        rlay.addWidget(QLabel("里程碑记录"))
        self.milestones = QListWidget()
        rlay.addWidget(self.milestones, 1)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

    def _build_status(self) -> None:
        self.status = self.statusBar()
        self.status.showMessage("就绪。按“开始录音”或导入 WAV。")

    # ------------------------------------------------------------------ #
    # recording
    # ------------------------------------------------------------------ #
    def toggle_recording(self) -> None:
        if self.recorder and self.recorder.running:
            self._stop_recording()
            self.act_record.setText("● 开始录音")
        else:
            self._start_recording()
            self.act_record.setText("■ 停止录音")

    def _start_recording(self) -> None:
        self.denoiser = Denoiser(backend=self.cb_denoise.currentText() or "auto")
        self.recorder = AudioRecorder(denoiser=self.denoiser, on_block=self._on_live_block)
        self.recorder.start()
        self.status.showMessage("录音中… (实时降噪 + HF 保护)")
        from PySide6.QtCore import QTimer
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._refresh_live_wave)
        self._live_timer.start(60)

    def _stop_recording(self) -> None:
        if self._live_timer:
            self._live_timer.stop()
            self._live_timer = None
        samples = self.recorder.stop()
        self.status.showMessage(f"录音结束 ({len(samples)/config.SAMPLE_RATE:.1f}s)，处理中…")
        self._process(samples, tag=config.SAFE_TAG_DEFAULT)

    def _on_live_block(self, block: np.ndarray) -> None:
        # cheap: only store latest for the timer to draw
        self._latest_block = block

    def _refresh_live_wave(self) -> None:
        if self.recorder:
            self.wave.set_samples(self.recorder.get_buffer(), config.SAMPLE_RATE)

    # ------------------------------------------------------------------ #
    # import / processing
    # ------------------------------------------------------------------ #
    def import_wav(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频", str(config.DATA_ROOT), "音频 (*.wav *.mp3 *.flac)"
        )
        if not path:
            return
        try:
            samples = load_wav(path, config.SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self.wave.set_samples(samples, config.SAMPLE_RATE)
        self._set_play_samples(samples, config.SAMPLE_RATE)
        self.status.showMessage(f"已导入 {path}，处理中…")
        self._process(samples, tag=config.SAFE_TAG_DEFAULT)

    def _process(self, samples: np.ndarray, tag: str) -> None:
        if self._worker_thread is not None and self._worker_thread.isRunning():
            QMessageBox.information(self, "忙", "上一次处理尚未完成，请稍候。")
            return
        self._worker = ProcessWorker(self.processor, samples, tag, self.cb_asr.isChecked())
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker.finished.connect(self._on_process_done)
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def _on_process_done(self, outcome) -> None:
        if self._worker_thread:
            self._worker_thread.quit()
            self._worker_thread = None
            self._worker = None
        if isinstance(outcome, Exception):
            QMessageBox.critical(self, "处理失败", str(outcome))
            self.status.showMessage("处理失败。")
            return
        self._show_outcome(outcome)
        self._refresh_clips()
        self._refresh_milestones()
        self.status.showMessage(
            f"已归档: {outcome.clip.tag} | {outcome.emotion.label} | 里程碑 +{len(outcome.milestones)}"
        )

    def _show_outcome(self, o: ProcessOutcome) -> None:
        from src.audio.recorder import load_wav
        try:
            samples = load_wav(self.files.resolve(o.clip.audio_path), config.SAMPLE_RATE)
            self.wave.set_samples(samples, config.SAMPLE_RATE)
            self._set_play_samples(samples, config.SAMPLE_RATE)
        except Exception:
            self.wave.set_samples(np.zeros(0, dtype=np.float32), config.SAMPLE_RATE)
            self._set_play_samples(None)
        self.wave.set_keywords(o.keywords)
        self.wave.set_milestones([m.timestamp for m in o.milestones])
        f = o.features
        self.lbl_emotion.setText(
            f"情绪: {_EMOTION_ZH.get(o.emotion.label, o.emotion.label)} "
            f"({o.emotion.confidence*100:.0f}%)"
        )
        self.lbl_emotion.setStyleSheet(
            f"font-size:15pt;font-weight:bold;color:{_EMOTION_COLORS.get(o.emotion.label,'#fff')};"
        )
        self.lbl_features.setText(
            f"特征: F0均值 {f.f0_mean:.0f}Hz (σ {f.f0_std:.0f}) | "
            f"能量 {f.energy_mean_db:.1f}dB | 浊音率 {f.voiced_ratio*100:.0f}% | "
            f"笑调节奏 {f.laugh_modulation:.2f} | 时长 {f.duration:.1f}s"
        )
        self.lbl_transcript.setText(f"转写: {o.clip.transcript or '(无)'}")
        if o.keywords:
            ks = "  ".join(f"{h.word}@{h.start:.2f}-{h.end:.2f}s" for h in o.keywords)
            self.lbl_keywords.setText(f"关键词时间戳: {ks}")
        else:
            self.lbl_keywords.setText("关键词时间戳: (无)")

    # ------------------------------------------------------------------ #
    # playback controls
    # ------------------------------------------------------------------ #
    @staticmethod
    def _fmt_t(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"

    def _set_play_samples(self, samples: np.ndarray, sr: int = config.SAMPLE_RATE) -> None:
        """Set the playback buffer and enable/disable controls accordingly."""
        self._play_stop()
        if samples is None or samples.size == 0:
            self._play_samples = None
            self._play_sr = sr
            self.btn_play.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.play_slider.setEnabled(False)
            self.play_slider.setValue(0)
            self.lbl_time.setText("00:00 / 00:00")
            return
        self._play_samples = np.ascontiguousarray(samples, dtype=np.float32)
        self._play_sr = sr
        self._play_pos = 0
        self.btn_play.setEnabled(True)
        self.btn_play.setText("▶ 播放")
        self.btn_stop.setEnabled(False)
        self.play_slider.setEnabled(True)
        self.play_slider.setRange(0, max(1, int(self._play_samples.size / self._play_sr * 1000)))
        self.play_slider.setValue(0)
        self.lbl_time.setText(f"00:00 / {self._fmt_t(self._play_samples.size / self._play_sr)}")

    def _play_toggle(self) -> None:
        if self._play_samples is None:
            return
        if self._play_stream is not None and self._play_stream.active:
            self._pause_stream()
            return
        # start or resume
        if self._play_pos >= self._play_samples.size:
            self._play_pos = 0
        import sounddevice as sd

        stream = sd.OutputStream(
            samplerate=self._play_sr,
            channels=1,
            blocksize=512,
            dtype="float32",
            callback=self._play_callback,
            finished_callback=self._play_on_finished,
        )
        self._play_stream = stream
        stream.start()
        self.btn_play.setText("⏸ 暂停")
        self.btn_stop.setEnabled(True)

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_update_ui)
        self._play_timer.start(100)

    def _play_stop(self) -> None:
        self._pause_stream()
        self._play_pos = 0
        self.wave.hide_playback_cursor()
        if self._play_samples is not None:
            self.play_slider.setValue(0)
            self.lbl_time.setText(f"00:00 / {self._fmt_t(self._play_samples.size / self._play_sr)}")

    def _play_seek(self, slider_pos: int) -> None:
        if self._play_samples is None:
            return
        t = slider_pos / 1000.0
        self._play_pos = int(t * self._play_sr)
        self.wave.set_playback_position(t)
        self.lbl_time.setText(
            f"{self._fmt_t(t)} / {self._fmt_t(self._play_samples.size / self._play_sr)}"
        )

    def _pause_stream(self) -> None:
        if self._play_stream is not None and self._play_stream.active:
            try:
                self._play_stream.stop()
            except Exception:
                pass
        self._play_stream = None
        if self._play_timer is not None:
            self._play_timer.stop()
            self._play_timer = None
        self.btn_play.setText("▶ 播放")
        self.btn_stop.setEnabled(False)

    def _play_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        if self._play_samples is None:
            outdata.fill(0)
            return
        with self._play_lock:
            pos = self._play_pos
            end = min(pos + frames, self._play_samples.size)
            chunk_size = end - pos
            vol = max(0.0, min(1.0, self.vol_slider.value() / 100.0))
            if chunk_size > 0:
                chunk = self._play_samples[pos:end]
                # apply volume
                if vol != 1.0:
                    chunk = chunk * vol
                outdata[:chunk_size, 0] = chunk
                outdata[chunk_size:, 0] = 0.0
                self._play_pos = end
            else:
                outdata.fill(0)

    def _play_on_finished(self) -> None:
        # called from the audio thread; just schedule UI update via Qt event loop
        pass

    def _play_update_ui(self) -> None:
        if self._play_samples is None:
            return
        with self._play_lock:
            pos = self._play_pos
        t = pos / self._play_sr
        dur = self._play_samples.size / self._play_sr
        self.play_slider.blockSignals(True)
        self.play_slider.setValue(int(t * 1000))
        self.play_slider.blockSignals(False)
        self.lbl_time.setText(f"{self._fmt_t(t)} / {self._fmt_t(dur)}")
        self.wave.set_playback_position(t)
        if pos >= self._play_samples.size:
            self._pause_stream()
            self._play_pos = 0
            self.wave.hide_playback_cursor()

    # ------------------------------------------------------------------ #
    # clips table population / interaction
    # ------------------------------------------------------------------ #
    def _refresh_clips(self) -> None:
        text = self.search.text().strip()
        milestone_only = self.btn_milestone_only.isChecked()
        if text or milestone_only:
            clips = self.db.search_clips(
                keyword=text or None, milestone_only=milestone_only
            )
        else:
            clips = self.db.list_clips(limit=1000)
        self.table.setRowCount(0)
        for c in clips:
            r = self.table.rowCount()
            self.table.insertRow(r)
            date_item = QTableWidgetItem(c.recorded_at.strftime("%Y-%m-%d %H:%M"))
            date_item.setData(Qt.UserRole, c.id)
            tag_item = QTableWidgetItem(c.tag)
            emo_item = QTableWidgetItem(_EMOTION_ZH.get(c.emotion, c.emotion))
            emo_item.setForeground(_qcolor(_EMOTION_COLORS.get(c.emotion, "#cccccc")))
            dur_item = QTableWidgetItem(f"{c.duration_s:.1f}")
            f0_item = QTableWidgetItem(f"{c.f0_mean:.0f}")
            tr_item = QTableWidgetItem(c.transcript[:30] + ("…" if len(c.transcript) > 30 else ""))
            ms_item = QTableWidgetItem("★" if c.is_milestone else "")
            ms_item.setTextAlignment(Qt.AlignCenter)
            for col, it in enumerate(
                [date_item, tag_item, emo_item, dur_item, f0_item, tr_item, ms_item]
            ):
                it.setFlags(it.flags() | Qt.ItemIsEditable if col == 1 else it.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(r, col, it)

    def _on_clip_selected(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        clip_id = self.table.item(row, 0).data(Qt.UserRole)
        clip = self.db.get_clip(clip_id)
        if not clip:
            return
        from src.audio.recorder import load_wav
        try:
            samples = load_wav(self.files.resolve(clip.audio_path), config.SAMPLE_RATE)
            self.wave.set_samples(samples, config.SAMPLE_RATE)
            self._set_play_samples(samples, config.SAMPLE_RATE)
        except Exception:
            self.wave.clear()
            self._set_play_samples(None)
            samples = None
        from src.asr.recognizer import VoskRecognizer
        from src.asr.keyword import KeywordSpotter
        try:
            asr = VoskRecognizer().transcribe(samples)
            hits = KeywordSpotter().find(asr)
        except Exception:
            asr = None
            hits = []
        self.wave.set_keywords(hits)
        ms = [m for m in self.db.list_milestones() if m.clip_id == clip_id]
        self.wave.set_milestones([m.timestamp for m in ms])
        self.lbl_emotion.setText(
            f"情绪: {_EMOTION_ZH.get(clip.emotion, clip.emotion)} "
            f"({clip.emotion_confidence*100:.0f}%)"
        )
        self.lbl_emotion.setStyleSheet(
            f"font-size:15pt;font-weight:bold;color:{_EMOTION_COLORS.get(clip.emotion,'#fff')};"
        )
        self.lbl_features.setText(
            f"特征: F0均值 {clip.f0_mean:.0f}Hz | 能量 {clip.energy_db:.1f}dB | "
            f"时长 {clip.duration_s:.1f}s | 标签 {', '.join(clip.tags) or '-'}"
        )
        self.lbl_transcript.setText(f"转写: {clip.transcript or '(无)'}")
        ks = "  ".join(f"{h.word}@{h.start:.2f}s" for h in hits)
        self.lbl_keywords.setText(f"关键词时间戳: {ks or '(无)'}")

    def _apply_tag_to_clip(self, clip_id: int, new_tag: str) -> None:
        clip = self.db.get_clip(clip_id)
        if not clip:
            return
        new_tag = sanitize_tag(new_tag)
        if new_tag == clip.tag:
            return
        self.processor.re_tag(clip, new_tag)
        self.status.showMessage(f"已将片段 #{clip_id} 重命名为 “{new_tag}”")
        self._refresh_clips()

    # ------------------------------------------------------------------ #
    # export
    # ------------------------------------------------------------------ #
    def export(self, selected: bool) -> None:
        if selected:
            row = self.table.currentRow()
            if row < 0:
                QMessageBox.information(self, "导出", "请先在左侧选中一个片段。")
                return
            clip_id = self.table.item(row, 0).data(Qt.UserRole)
            clips = [self.db.get_clip(clip_id)]
        else:
            clips = self.db.list_clips(limit=100000)
        if not clips:
            QMessageBox.information(self, "导出", "没有可导出的片段。")
            return
        dest = QFileDialog.getExistingDirectory(self, "选择导出目录", str(config.EXPORT_ROOT))
        if not dest:
            return
        from pathlib import Path
        out = export_clips(clips, self.db, Path(dest))
        QMessageBox.information(self, "导出完成", f"已导出 {len(clips)} 个片段到:\n{out}")

    # ------------------------------------------------------------------ #
    # milestones / palette
    # ------------------------------------------------------------------ #
    def _refresh_milestones(self) -> None:
        self.milestones.clear()
        for m in self.db.list_milestones():
            clip = self.db.get_clip(m.clip_id)
            when = clip.recorded_at.strftime("%Y-%m-%d %H:%M") if clip else "?"
            self.milestones.addItem(f"★ {m.word}  @ {when}  ({m.timestamp:.1f}s)")

    def _add_custom_tag(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(self, "新建标签", "标签名称:")
        if ok and text.strip():
            self.palette.add_tag(sanitize_tag(text))

    # ------------------------------------------------------------------ #
    # toolbar handlers
    # ------------------------------------------------------------------ #
    def _on_asr_toggle(self, on: bool) -> None:
        if not on:
            self._run_asr = False
            return
        if self.processor.asr_available():
            self._run_asr = True
            return
        # Not available -> prompt user with download links + one-click option
        self._prompt_download_model_and_enable()
        self.cb_asr.blockSignals(True)
        self.cb_asr.setChecked(bool(self._run_asr))
        self.cb_asr.blockSignals(False)

    def _prompt_download_model_and_enable(self) -> None:
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        from src.asr.recognizer import VoskModelMissingError

        err = VoskModelMissingError(config.VOSK_MODEL_PATH)
        mb = QMessageBox(self)
        mb.setIcon(QMessageBox.Warning)
        mb.setWindowTitle("启用语音识别缺少模型")
        mb.setTextFormat(Qt.RichText)
        mb.setText(
            f"<p><b>未找到 Vosk 中文离线识别模型。</b></p>"
            f"<p>期望路径：<br><code>{config.VOSK_MODEL_PATH}</code></p>"
            f"<p>可选项：</p>"
            f"<ol>"
            f"  <li>点击下方 “<b>一键自动下载</b>” 按钮（推荐，官方源约 40 MB）。</li>"
            f"  <li>手动下载并解压：</li>"
            f"    <ul style='list-style:none'>"
            f"      <li>· <a href='{config.VOSK_MODEL_DOWNLOAD_URLS[0]}'>alphacephei.com 官方镜像</a></li>"
            f"      <li>· <a href='{config.VOSK_MODEL_DOWNLOAD_URLS[1]}'>GitHub Releases 镜像</a></li>"
            f"      <li>· 全部模型：<a href='{config.VOSK_MODEL_HOME_URL}'>{config.VOSK_MODEL_HOME_URL}</a></li>"
            f"    </ul>"
            f"    解压后放入：<code>{config.VOSK_MODEL_PATH.parent}</code></ol>"
        )
        mb.setTextInteractionFlags(Qt.TextBrowserInteraction | Qt.LinksAccessibleByMouse)
        auto_btn = mb.addButton("一键自动下载", QMessageBox.AcceptRole)
        open_dir_btn = mb.addButton("打开目标文件夹", QMessageBox.ActionRole)
        home_btn = mb.addButton("在浏览器打开下载页", QMessageBox.ActionRole)
        cancel_btn = mb.addButton("取消", QMessageBox.RejectRole)
        mb.exec()
        clicked = mb.clickedButton()

        if clicked is open_dir_btn:
            from pathlib import Path
            import os
            Path(config.VOSK_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            os.startfile(str(Path(config.VOSK_MODEL_PATH).parent))  # type: ignore[attr-defined]
            return
        if clicked is home_btn:
            QDesktopServices.openUrl(QUrl(config.VOSK_MODEL_HOME_URL))
            return
        if clicked is not auto_btn:
            self._run_asr = False
            return

        # ---- run the download in a background thread with progress ----
        prog = QProgressDialog("正在下载 Vosk 中文模型…", "取消", 0, 100, self)
        prog.setWindowTitle("下载离线语音识别模型")
        prog.setMinimumDuration(0)
        prog.setWindowModality(Qt.WindowModal)
        prog.setAutoReset(False)
        prog.setAutoClose(False)
        prog.setValue(0)

        worker = ModelDownloadWorker(force=False)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.failed.connect(lambda msg: (
            prog.cancel(),
            QMessageBox.critical(self, "下载失败", msg),
        ))

        def _on_progress(done: int, total: int) -> None:
            if prog.wasCanceled():
                return
            if total > 0:
                pct = int(done * 100 / total)
                prog.setMaximum(100)
                prog.setValue(pct)
                prog.setLabelText(
                    f"下载中 {done/1024/1024:.1f} / {total/1024/1024:.1f} MB ({pct}%)"
                )
            else:
                prog.setMaximum(0)  # busy
                prog.setLabelText(f"下载中 {done/1024/1024:.1f} MB …")

        worker.progress.connect(_on_progress)

        def _on_ok(path) -> None:
            prog.reset()
            prog.close()
            self.status.showMessage(f"模型下载完成: {path}")
            self._run_asr = True
            self.cb_asr.blockSignals(True)
            self.cb_asr.setChecked(True)
            self.cb_asr.blockSignals(False)
            QMessageBox.information(
                self, "下载完成",
                f"Vosk 中文模型已就绪：\n{path}\n现在可以使用语音识别了。",
            )

        worker.finished_ok.connect(_on_ok)
        # thread lifecycle cleanup
        def _cleanup():
            thread.quit()
            thread.wait(2000)
        worker.finished_ok.connect(_cleanup)
        worker.failed.connect(_cleanup)
        prog.canceled.connect(lambda: (
            _cleanup(),
            setattr(self, '_run_asr', False),
        ))
        thread.start()

    def _on_denoise_backend(self, name: str) -> None:
        try:
            self.denoiser = Denoiser(backend=name)
            self.status.showMessage(f"降噪后端: {self.denoiser.backend_name}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "降噪后端", str(exc))

    # ------------------------------------------------------------------ #
    def closeEvent(self, event):
        if self.recorder and self.recorder.running:
            self.recorder.stop()
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
        self._pause_stream()
        event.accept()


def _qcolor(hexstr: str):
    from PySide6.QtGui import QColor
    return QColor(hexstr)
