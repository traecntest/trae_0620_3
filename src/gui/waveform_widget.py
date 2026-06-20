"""Waveform visualisation widget (pyqtgraph).

Draws a downsampled amplitude envelope plus vertical markers for recognised
keyword hits and star markers for milestones.  Designed to be cheap to
update from a live recording timer (envelope computed once per refresh).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

import config
from src.asr.keyword import KeywordHit


class WaveformWidget:
    """Thin facade around a ``pg.PlotWidget`` so callers don't import pyqtgraph."""

    def __init__(self, parent=None):
        import pyqtgraph as pg

        pg.setConfigOptions(antialias=False, useOpenGL=False)
        self.plot = pg.PlotWidget(parent=parent)
        self.plot.setLabel("bottom", "时间", units="s")
        self.plot.setLabel("left", "幅度")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setBackground("#101418")
        self.plot.setMouseEnabled(x=True, y=False)
        self._curve_upper = self.plot.plot(pen=pg.mkPen("#4cc9f0", width=1))
        self._curve_lower = self.plot.plot(pen=pg.mkPen("#4cc9f0", width=1))
        self._fill = pg.FillBetweenItem(
            self._curve_upper, self._curve_lower, brush=pg.mkBrush("#4cc9f033")
        )
        self.plot.addItem(self._fill)
        self._keyword_lines: list = []
        self._milestone_items: list = []
        self._sr = config.SAMPLE_RATE

    # ------------------------------------------------------------------ #
    def widget(self):
        return self.plot

    def set_samples(self, samples: np.ndarray, sr: int = config.SAMPLE_RATE) -> None:
        self._sr = sr
        samples = np.asarray(samples, dtype=np.float32)
        if samples.size == 0:
            self._curve_upper.setData([], [])
            self._curve_lower.setData([], [])
            return
        # downsample to ~2000 points via min/max envelope per bucket
        width = 2000
        if samples.size > width:
            trimmed = samples[: (samples.size // width) * width]
            reshaped = trimmed.reshape(width, -1)
            upper = reshaped.max(axis=1)
            lower = reshaped.min(axis=1)
        else:
            upper = samples
            lower = samples
        t = np.linspace(0, samples.size / sr, upper.size, endpoint=False)
        self._curve_upper.setData(t, upper)
        self._curve_lower.setData(t, lower)
        self.plot.setXRange(0, max(samples.size / sr, 0.1), padding=0.02)

    def set_keywords(self, hits: Sequence[KeywordHit]) -> None:
        import pyqtgraph as pg

        for it in self._keyword_lines:
            self.plot.removeItem(it)
        self._keyword_lines = []
        for h in hits:
            line = pg.InfiniteLine(
                pos=h.start, angle=90,
                pen=pg.mkPen("#f72585", width=2),
                label=f"{h.word} {h.start:.2f}s",
                labelOpts={"position": 0.85, "color": "#f72585"},
            )
            self.plot.addItem(line)
            self._keyword_lines.append(line)

    def set_milestones(self, ts_list: Sequence[float]) -> None:
        import pyqtgraph as pg

        for it in self._milestone_items:
            self.plot.removeItem(it)
        self._milestone_items = []
        upper = self.plot.getAxis("left").range[1] if self.plot.getAxis("left").range else 1.0
        for ts in ts_list:
            star = pg.ScatterPlotItem(
                x=[ts], y=[upper * 0.9], size=18,
                symbol="star", brush=pg.mkBrush("#ffd60a"),
                pen=pg.mkPen("#ffd60a"),
            )
            self.plot.addItem(star)
            self._milestone_items.append(star)

    def clear(self) -> None:
        self.set_samples(np.zeros(0, dtype=np.float32))
        self.set_keywords([])
        self.set_milestones([])
