"""
data.py — Core data structures passed between all processing stages.

InstrumentData is the single object that flows through the pipeline:
    InputModule -> Processor1 -> Processor2 -> ... -> OutputModule

Each processor receives an InstrumentData and returns a (possibly modified)
InstrumentData. Audio is stored in-memory as numpy float64 arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Sample:
    """Represents one individual note+velocity slice of audio."""

    midi_note: int          # MIDI note number (0–127)
    velocity: int           # MIDI velocity value (1–127)
    audio: np.ndarray       # shape: (n_samples,) mono float64, or (n_samples, n_ch)
    sr: int                 # sample rate in Hz

    # Set by LoopFinderProcessor; None until then
    loop_start: Optional[int] = None   # sample index into audio
    loop_end: Optional[int] = None     # sample index into audio


@dataclass
class InstrumentMeta:
    """All fields parsed from metadata.txt."""

    # --- Recording parameters (must match generate_midi.py settings) ---
    velocity_layers: int          # X: number of velocity layers recorded
    semitone_interval: int        # N: semitone steps between recorded notes
    hold_time: float              # H: sustain.wav — how long each note was held
    release_time: float           # R: sustain.wav — silence captured after note-off
    start_note: int               # lowest MIDI note in the recording (default 21)
    end_note: int                 # highest MIDI note in the recording (default 108)

    # --- Output sample selection ---
    min_note: int                 # lowest MIDI note to include in output
    max_note: int                 # highest MIDI note to include in output
    note_percentage: float        # 1–100: fraction of notes in range to keep
    velocity_layers_out: int      # how many velocity layers to include in output

    # --- Crossfade ---
    crossfade_percent: float      # 0–100: % of velocity zone width that overlaps

    # --- Normalization ---
    normalize: bool
    normalize_mode: str           # 'lufs', 'rms', or 'velocity'
    normalize_target_lufs: float  # target LUFS (or dB RMS) for lufs/rms modes
    peak_ceiling_db: float        # hard peak ceiling after normalization
    velocity_dynamic_range_db: float  # amp_velcurve_1 range for lufs/rms modes

    # --- SFZ / Playback ---
    instrument_name: str
    ampeg_release: float          # seconds; also used as ampeg_attack in release SFZ
    loop_crossfade_ms: float      # loop_crossfade opcode value in ms

    # --- Advanced ---
    onset_threshold_db: float     # leading silence trim threshold in dBFS

    # --- Optional: release.wav recording overrides ---
    # sustain.wav and release.wav may have been recorded with different H and R values.
    # If these are None, the input module falls back to hold_time / release_time.
    release_hold_time: Optional[float] = None    # H used when recording release.wav
    release_release_time: Optional[float] = None # R (tail capture) used when recording release.wav

    # --- Computed fields (set during processing, not from metadata.txt) ---
    computed_dynamic_range_db: Optional[float] = None  # set by NormalizeProcessor (velocity mode)
    collapse_to_mono: bool = False                      # mix stereo to mono on input

    def effective_release_hold_time(self) -> float:
        """Return release_hold_time, falling back to hold_time if not set."""
        return self.release_hold_time if self.release_hold_time is not None else self.hold_time

    def effective_release_release_time(self) -> float:
        """Return release_release_time, falling back to release_time if not set."""
        return self.release_release_time if self.release_release_time is not None else self.release_time


@dataclass
class InstrumentData:
    """
    Central data structure that flows through the entire processing pipeline.

    sustain: one Sample per (note, velocity) combination from sustain.wav
    release: same structure from release.wav; empty list if no release.wav
    meta:    all configuration/metadata for this instrument
    """

    sustain: List[Sample]
    release: List[Sample]
    meta: InstrumentMeta

    def all_samples(self) -> List[Sample]:
        """Convenience: iterate all sustain + release samples."""
        return self.sustain + self.release

    def sustain_notes(self) -> List[int]:
        """Sorted unique MIDI notes present in sustain samples."""
        return sorted(set(s.midi_note for s in self.sustain))

    def sustain_velocities(self) -> List[int]:
        """Sorted unique velocity values present in sustain samples."""
        return sorted(set(s.velocity for s in self.sustain))

    def get_sustain(self, midi_note: int, velocity: int) -> Optional[Sample]:
        """Look up a specific sustain sample by note and velocity."""
        for s in self.sustain:
            if s.midi_note == midi_note and s.velocity == velocity:
                return s
        return None
