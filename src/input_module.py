"""
input_module.py — Load an instrument folder into InstrumentData.

Each instrument folder must contain:
    input.toml    — ordered sample manifest (one entry per event in samples.flac)
    output.toml   — processing / SFZ configuration
    samples.flac  — all samples in sequence (sustain + release interleaved as recorded)

The samples.flac contains all events back-to-back.  input.toml describes each
event in order: its MIDI note, velocity, hold length, tail length, and whether
it is a release sample.  This module slices the long file into individual
per-(note, velocity) Sample objects in memory — no intermediate files written.

Public API:
    load_instrument(folder_path: str | Path) -> InstrumentData
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

from .data import InstrumentData, InstrumentMeta, InputSampleDef, Sample
from .metadata_parser import parse_input_meta, parse_output_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Mix a (n_frames, n_channels) or (n_frames,) array down to (n_frames,)."""
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _slice_samples(
    audio: np.ndarray,
    sr: int,
    sample_defs: List[InputSampleDef],
    min_note: int,
    max_note: int,
    collapse_to_mono: bool,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Slice a long samples.flac array into sustain and release Sample objects.

    Event layout in the audio file:
        event_i starts at offset = sum(def_j.hold_s + def_j.tail_s  for j < i)

    Extraction:
        is_release=False → [event_start,          event_start + hold_s]
        is_release=True  → [event_start + hold_s,  event_start + hold_s + tail_s]

    Notes outside [min_note, max_note] are skipped but their time offsets are
    still accumulated so slicing of later events remains correct.

    Returns:
        (sustain_samples, release_samples) — each a list of Sample objects.
    """
    total_frames = audio.shape[0]
    sustain: List[Sample] = []
    release: List[Sample] = []

    current_offset_s = 0.0

    for defn in sample_defs:
        event_start_s = current_offset_s
        current_offset_s += defn.hold_s + defn.tail_s  # advance regardless of filter

        if not (min_note <= defn.note <= max_note):
            continue

        if defn.is_release:
            slice_start = int(round((event_start_s + defn.hold_s) * sr))
            slice_len   = int(round(defn.tail_s * sr))
        else:
            slice_start = int(round(event_start_s * sr))
            slice_len   = int(round(defn.hold_s * sr))

        slice_end = slice_start + slice_len

        if slice_start >= total_frames:
            chunk = np.zeros(max(slice_len, 1), dtype=np.float64)
        else:
            actual_end = min(slice_end, total_frames)
            chunk = audio[slice_start:actual_end].astype(np.float64)
            if chunk.shape[0] < slice_len:
                pad = slice_len - chunk.shape[0]
                if chunk.ndim == 1:
                    chunk = np.pad(chunk, (0, pad))
                else:
                    chunk = np.pad(chunk, ((0, pad), (0, 0)))

        if collapse_to_mono:
            chunk = _to_mono(chunk)
        elif chunk.ndim == 2 and chunk.shape[1] == 1:
            chunk = chunk[:, 0]

        sample = Sample(
            midi_note=defn.note,
            velocity=defn.velocity,
            audio=chunk,
            sr=sr,
        )

        if defn.is_release:
            release.append(sample)
        else:
            sustain.append(sample)

    return sustain, release


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_instrument(folder_path: str | Path) -> InstrumentData:
    """
    Load an instrument folder into an InstrumentData ready for the pipeline.

    Steps:
      1. Parse input.toml  → sample definitions + collapse_to_mono flag
      2. Parse output.toml → InstrumentMeta
      3. Load samples.flac (or .wav fallback)
      4. Slice into sustain / release Sample lists using the sample definitions
      5. Return InstrumentData(sustain, release, meta)

    Args:
        folder_path: Path to the instrument folder containing input.toml,
                     output.toml, and samples.flac.

    Returns:
        InstrumentData populated with all sliced samples.

    Raises:
        FileNotFoundError:  if input.toml, output.toml, or samples.flac is missing.
        MetadataError:      if either TOML file is invalid.
        RuntimeError:       if the audio file cannot be read.
    """
    folder = Path(folder_path)

    samples_path = next(
        (folder / f"samples{ext}" for ext in (".flac", ".wav")
         if (folder / f"samples{ext}").exists()),
        None,
    )
    if samples_path is None:
        raise FileNotFoundError(f"samples.flac/.wav not found in: {folder}")

    sample_defs, collapse_to_mono = parse_input_meta(folder)
    meta: InstrumentMeta = parse_output_meta(folder)
    meta.instrument_name = folder.name

    audio, sr = sf.read(str(samples_path), dtype="float64", always_2d=False)

    sustain, release = _slice_samples(
        audio=audio,
        sr=sr,
        sample_defs=sample_defs,
        min_note=meta.min_note,
        max_note=meta.max_note,
        collapse_to_mono=collapse_to_mono,
    )

    return InstrumentData(sustain=sustain, release=release, meta=meta)
