"""
trim.py — TrimProcessor: cut a fixed number of milliseconds from the front of each sample.

`pre_trim_ms` (from metadata) is applied to every sustain and release sample.
Set to 0.0 to skip trimming entirely.
"""

from __future__ import annotations

from ..base import ProcessingObject
from ..data import InstrumentData


class TrimProcessor(ProcessingObject):
    """
    Cut `meta.pre_trim_ms` milliseconds from the front of every sample.

    Works for mono (1-D) and multichannel (2-D, frames × channels) arrays.
    """

    def process(self, data: InstrumentData) -> InstrumentData:
        for s in data.all_samples():
            cut = int(data.meta.pre_trim_ms / 1000.0 * s.sr)
            if cut > 0:
                s.audio = s.audio[cut:]
        return data
