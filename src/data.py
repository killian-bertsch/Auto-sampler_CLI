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
class InputSampleDef:
    """
    One entry in input.toml — describes a single event in samples.flac.

    Events are laid out sequentially in samples.flac:
      event_i starts at sum(def_j.hold_s + def_j.tail_s  for j < i) seconds

    For is_release=False (sustain): extract [event_start, event_start + hold_s]
    For is_release=True  (release): extract [event_start + hold_s, event_start + hold_s + tail_s]
    """
    note: int        # MIDI note number (0–127)
    velocity: int    # minimum velocity of this sample's zone (1–127); zone extends to next_velocity - 1
    hold_s: float    # seconds the note was held
    tail_s: float    # seconds of decay/silence captured after note-off
    is_release: bool # True → extract tail; False → extract hold portion


@dataclass
class Sample:
    """Represents one individual note+velocity slice of audio."""

    midi_note: int          # MIDI note number (0–127)
    velocity: int           # minimum velocity of this sample's zone (1–127); zone extends to next_velocity - 1
    audio: np.ndarray       # shape: (n_samples,) mono float64, or (n_samples, n_ch)
    sr: int                 # sample rate in Hz

    # Set by LoopFinderProcessor; None until then
    loop_start: Optional[int] = None   # sample index into audio
    loop_end: Optional[int] = None     # sample index into audio


@dataclass
class InstrumentMeta:
    """All fields parsed from output.toml."""

    # --- Output sample selection ---
    min_note: int = 21                # lowest MIDI note to include in output
    max_note: int = 108               # highest MIDI note to include in output
    note_percentage: float = 100.0   # 1–100: fraction of notes in range to keep

    # --- Crossfade ---
    crossfade_percent: float = 0.0   # 0–100: % of velocity zone width that overlaps

    # --- Normalization ---
    normalize: bool = True
    normalize_mode: str = "lufs"           # 'lufs', 'rms', or 'velocity'
    normalize_target_lufs: float = -18.0  # target LUFS (or dB RMS) for lufs/rms modes
    peak_ceiling_db: float = -1.0         # hard peak ceiling after normalization
    velocity_dynamic_range_db: float = 40.0  # amp_velcurve_1 range for lufs/rms modes

    # --- SFZ / Playback ---
    instrument_name: str = ""
    ampeg_release: float = 0.5         # seconds; also used as ampeg_attack in release SFZ
    loop_crossfade_ms: float = 20.0    # loop_crossfade opcode value in ms

    # --- Advanced ---
    pre_trim_ms: float = 0.0           # ms to cut from the front of every sample

    # --- Transient shaper ---
    enable_transient_shaper: bool = False
    transient_attack_db: float = 6.0
    transient_sustain_db: float = 0.0
    transient_speed_ms: float = 10.0

    # --- Computed fields (set during processing, not from output.toml) ---
    computed_dynamic_range_db: Optional[float] = None  # set by NormalizeProcessor (velocity mode)


@dataclass
class InstrumentData:
    """
    Central data structure that flows through the entire processing pipeline.

    sustain: one Sample per (note, velocity) combination — the held portion
    release: same structure for release tails; empty list if none recorded
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
