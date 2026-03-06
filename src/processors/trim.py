"""
trim.py — TrimProcessor: trim leading silence from each sample.

Uses onset detection (5 ms RMS windows) to find the first frame where
the signal exceeds `meta.onset_threshold_db`, then slices the audio
array from that point forward.  Works for mono (1-D) and multichannel
(2-D, frames × channels) arrays.
"""

from __future__ import annotations

import numpy as np

from ..base import ProcessingObject
from ..data import InstrumentData


def _detect_onset(audio: np.ndarray, sr: int, threshold_db: float) -> int:
    """
    Return the sample index of the first onset.

    Scans in 5 ms windows; once a window exceeds the RMS threshold the
    exact sample within that window is found.  Returns 0 if no silent
    leading region is detected.
    """
    window = max(1, int(sr * 0.005))  # 5 ms windows
    threshold_linear = 10 ** (threshold_db / 20.0)

    # Use max across channels so any channel triggering counts
    mono = np.max(np.abs(audio), axis=1) if audio.ndim > 1 else np.abs(audio)
    n = len(mono)

    for i in range(0, n - window, window):
        rms = float(np.sqrt(np.mean(mono[i : i + window] ** 2)))
        if rms >= threshold_linear:
            # Refine: find first sample in this window that crosses threshold
            for j in range(i, min(i + window, n)):
                if mono[j] >= threshold_linear:
                    return j
            return i
    return 0


class TrimProcessor(ProcessingObject):
    """
    Trim leading silence from every sample (sustain and release).

    The threshold is read from `data.meta.onset_threshold_db`.
    """

    def process(self, data: InstrumentData) -> InstrumentData:
        threshold = data.meta.onset_threshold_db
        for s in data.all_samples():
            onset = _detect_onset(s.audio, s.sr, threshold)
            if onset > 0:
                s.audio = s.audio[onset:]
        return data
