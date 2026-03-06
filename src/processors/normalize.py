"""
normalize.py — NormalizeProcessor: DC removal + loudness normalization.

Processing order:
  1. DC offset removal    — always applied to all samples (sustain + release)
  2. Group normalization  — only when meta.normalize is True

Normalization modes (meta.normalize_mode):
  'lufs'     — ITU-R BS.1770-4 integrated loudness; falls back to RMS if
               signal is too short/silent.  One scale factor per velocity
               layer (group mean → target, relative timbral diffs preserved).
  'rms'      — Same grouping logic, using RMS energy in dBFS.
  'velocity' — Per-note linear gain curve: scales all velocity layers of
               each note so their peak levels align with the loudest layer,
               then stores the computed total_range_db in
               meta.computed_dynamic_range_db for SFZ output.

After group normalization a peak ceiling (meta.peak_ceiling_db) is applied
per group so no sample exceeds that level.  The ceiling only ever attenuates,
never boosts.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import pyloudnorm as pyln

from ..base import ProcessingObject
from ..data import InstrumentData, Sample


# ---------------------------------------------------------------------------
# Low-level DSP helpers
# ---------------------------------------------------------------------------

def _remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """Subtract per-channel mean; returns float64 copy."""
    out = audio.astype(np.float64)
    if out.ndim == 1:
        out -= out.mean()
    else:
        out -= out.mean(axis=0)
    return out


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def _peak(audio: np.ndarray) -> float:
    return float(np.max(np.abs(audio.astype(np.float64))))


def _measure_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Measure integrated loudness via pyloudnorm.  Returns None on failure."""
    meter = pyln.Meter(sr)
    arr = audio.astype(np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    try:
        loudness = meter.integrated_loudness(arr)
    except Exception:
        return None
    return float(loudness) if np.isfinite(loudness) else None


def _measure_level(audio: np.ndarray, sr: int, mode: str) -> float:
    """Return signal level in dB.  Returns -100 sentinel for silence."""
    if mode == "lufs":
        lufs = _measure_lufs(audio, sr)
        if lufs is not None:
            return lufs
    r = _rms(audio)
    return 20.0 * np.log10(r) if r > 0 else -100.0


# ---------------------------------------------------------------------------
# Group normalization (lufs / rms modes)
# ---------------------------------------------------------------------------

def _apply_peak_ceiling(samples: List[Sample], peak_db: float) -> None:
    """Apply peak ceiling in-place across a group; never amplifies."""
    if not samples:
        return
    group_peak = max(_peak(s.audio) for s in samples)
    if group_peak == 0.0:
        return
    target_linear = 10.0 ** (peak_db / 20.0)
    scale = target_linear / group_peak
    if scale < 1.0:
        for s in samples:
            s.audio = s.audio * scale


def _normalize_group(
    samples: List[Sample],
    target: float,
    mode: str,
    peak_db: float,
) -> None:
    """
    Normalize a list of samples (same velocity layer) in-place.

    Computes the group mean level, applies one scale factor to all samples
    (preserving relative timbral differences), then enforces the peak ceiling.
    """
    if not samples:
        return
    sr = samples[0].sr
    levels = [_measure_level(s.audio, sr, mode) for s in samples]
    mean_level = float(np.mean(levels))

    if np.isfinite(mean_level) and mean_level > -100.0:
        gain_db = target - mean_level
        scale = 10.0 ** (gain_db / 20.0)
        for s in samples:
            s.audio = s.audio.astype(np.float64) * scale

    _apply_peak_ceiling(samples, peak_db)


def _normalize_by_velocity_layer(
    samples: List[Sample],
    target: float,
    mode: str,
    peak_db: float,
) -> None:
    """Group samples by velocity value, normalize each group independently."""
    by_vel: dict[int, List[Sample]] = defaultdict(list)
    for s in samples:
        by_vel[s.velocity].append(s)
    for group in by_vel.values():
        _normalize_group(group, target, mode, peak_db)


# ---------------------------------------------------------------------------
# Velocity mode (linear gain curve per note)
# ---------------------------------------------------------------------------

def _apply_velocity_mode(data: InstrumentData, peak_db: float) -> None:
    """
    For each MIDI note in sustain, apply a linear gain curve so that
    velocity layers align with the peak level of the highest-velocity sample.

    The gain applied to velocity v is:
        gain_db(v) = total_range_db * (127 - v) / 127

    This means vel=127 gets 0 dB gain and lower velocities get progressively
    boosted, making all layers peak at the same level as the loudest sample.

    The average total_range_db across all notes is stored in
    data.meta.computed_dynamic_range_db for the SFZ output module to use.

    After the per-note gain curve, a global peak ceiling is applied across
    all sustain samples.  Release samples are normalized per-velocity-layer
    using RMS (velocity curve is not meaningful for release tails).
    """
    by_note: dict[int, List[Sample]] = defaultdict(list)
    for s in data.sustain:
        by_note[s.midi_note].append(s)

    total_range_values: List[float] = []

    for note_samples in by_note.values():
        velocities = [s.velocity for s in note_samples]
        audios = [s.audio.astype(np.float64) for s in note_samples]

        peaks_db: List[float] = []
        for a in audios:
            p = _peak(a)
            peaks_db.append(20.0 * np.log10(p) if p > 0 else -100.0)

        vels = np.array(velocities, dtype=float)
        peaks = np.array(peaks_db, dtype=float)

        min_idx = int(np.argmin(vels))
        max_idx = int(np.argmax(vels))

        vel_min = float(vels[min_idx])
        vel_max = float(vels[max_idx])
        peak_min = float(peaks[min_idx])
        peak_max = float(peaks[max_idx])

        if vel_max <= vel_min or peak_min <= -99.0 or peak_max <= -99.0:
            total_range_values.append(0.0)
            continue

        slope = (peak_max - peak_min) / (vel_max - vel_min)
        total_range_db = float(np.clip(slope * 127.0, 0.0, 60.0))
        total_range_values.append(total_range_db)

        for s, v in zip(note_samples, velocities):
            gain_db = total_range_db * (127.0 - float(v)) / 127.0
            scale = 10.0 ** (gain_db / 20.0)
            s.audio = s.audio.astype(np.float64) * scale

    if total_range_values:
        data.meta.computed_dynamic_range_db = float(np.mean(total_range_values))

    # Peak ceiling across all sustain samples
    _apply_peak_ceiling(data.sustain, peak_db)

    # Release: use RMS group normalization per velocity layer
    if data.release:
        _normalize_by_velocity_layer(data.release, -18.0, "rms", peak_db)


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class NormalizeProcessor(ProcessingObject):
    """
    DC removal + optional loudness normalization for all samples.

    DC removal is unconditional.  Normalization is skipped when
    meta.normalize is False.
    """

    def process(self, data: InstrumentData) -> InstrumentData:
        # Always apply DC removal
        for s in data.all_samples():
            s.audio = _remove_dc_offset(s.audio)

        if not data.meta.normalize:
            return data

        mode = data.meta.normalize_mode
        target = data.meta.normalize_target_lufs
        peak_db = data.meta.peak_ceiling_db

        if mode in ("lufs", "rms"):
            _normalize_by_velocity_layer(data.sustain, target, mode, peak_db)
            _normalize_by_velocity_layer(data.release, target, mode, peak_db)
        elif mode == "velocity":
            _apply_velocity_mode(data, peak_db)

        return data
