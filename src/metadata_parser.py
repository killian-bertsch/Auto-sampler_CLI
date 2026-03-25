"""
metadata_parser.py — Parse instrument metadata from two TOML files.

  input.toml  — ordered sample manifest (what's in samples.flac)
  output.toml — processing / SFZ configuration

Public API:
    parse_input_meta(path)  -> (List[InputSampleDef], collapse_to_mono: bool)
    parse_output_meta(path) -> InstrumentMeta

Both functions accept either a directory path (they look for the file inside)
or a direct path to the file.

Raises:
    FileNotFoundError  — file not found
    MetadataError      — malformed or semantically invalid content
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .data import InputSampleDef, InstrumentMeta


class MetadataError(ValueError):
    """Raised when a metadata file is missing, malformed, or has invalid values."""


# ---------------------------------------------------------------------------
# output.toml field specification
# (name, python_type, required, default)
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS: list[tuple[str, type, bool, Any]] = [
    # Output selection
    ("min_note",                  int,   False, 21),
    ("max_note",                  int,   False, 108),
    ("note_percentage",           float, False, 100.0),
    # Crossfade
    ("crossfade_percent",         float, False, 0.0),
    # Normalization
    ("normalize",                 bool,  False, True),
    ("normalize_mode",            str,   False, "lufs"),
    ("normalize_target_lufs",     float, False, -18.0),
    ("peak_ceiling_db",           float, False, -1.0),
    ("velocity_dynamic_range_db", float, False, 40.0),
    # SFZ / Playback
    ("instrument_name",           str,   False, ""),
    ("ampeg_release",             float, False, 0.5),
    ("loop_crossfade_ms",         float, False, 20.0),
    # Advanced
    ("pre_trim_ms",               float, False, 0.0),
    # Transient shaper
    ("enable_transient_shaper",   bool,  False, False),
    ("transient_attack_db",       float, False, 6.0),
    ("transient_sustain_db",      float, False, 0.0),
    ("transient_speed_ms",        float, False, 10.0),
]


# ---------------------------------------------------------------------------
# input.toml parser
# ---------------------------------------------------------------------------

def parse_input_meta(path: str | Path) -> Tuple[List[InputSampleDef], bool]:
    """
    Parse input.toml and return (sample_defs, collapse_to_mono).

    Entries in [[samples]] must appear in the same order as events in
    samples.flac.  Each entry requires: note, velocity, hold_s, tail_s,
    is_release.

    Raises FileNotFoundError or MetadataError.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "input.toml"
    if not path.exists():
        raise FileNotFoundError(f"input.toml not found: {path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise MetadataError(f"TOML parse error in {path.name}: {exc}") from exc

    collapse_to_mono = bool(data.get("collapse_to_mono", False))

    raw_samples = data.get("samples", [])
    if not raw_samples:
        raise MetadataError("input.toml: no [[samples]] entries found")

    defs: List[InputSampleDef] = []
    for i, entry in enumerate(raw_samples):
        try:
            note       = int(entry["note"])
            velocity   = int(entry["velocity"])
            hold_s     = float(entry["hold_s"])
            tail_s     = float(entry["tail_s"])
            is_release = bool(entry["is_release"])
        except KeyError as exc:
            raise MetadataError(
                f"input.toml: samples[{i}] missing required field {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise MetadataError(
                f"input.toml: samples[{i}] invalid value: {exc}"
            ) from exc

        if not (0 <= note <= 127):
            raise MetadataError(
                f"input.toml: samples[{i}] note={note} out of range 0–127"
            )
        if not (1 <= velocity <= 127):
            raise MetadataError(
                f"input.toml: samples[{i}] velocity={velocity} out of range 1–127"
            )
        if hold_s <= 0:
            raise MetadataError(
                f"input.toml: samples[{i}] hold_s must be > 0"
            )
        if tail_s < 0:
            raise MetadataError(
                f"input.toml: samples[{i}] tail_s must be >= 0"
            )

        defs.append(InputSampleDef(
            note=note,
            velocity=velocity,
            hold_s=hold_s,
            tail_s=tail_s,
            is_release=is_release,
        ))

    return defs, collapse_to_mono


# ---------------------------------------------------------------------------
# output.toml parser
# ---------------------------------------------------------------------------

def parse_output_meta(path: str | Path) -> InstrumentMeta:
    """
    Parse output.toml and return an InstrumentMeta.

    All fields are optional and have sensible defaults.

    Raises FileNotFoundError or MetadataError.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "output.toml"
    if not path.exists():
        raise FileNotFoundError(f"output.toml not found: {path}")

    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise MetadataError(f"TOML parse error in {path.name}: {exc}") from exc

    parsed: Dict[str, Any] = {}
    for name, typ, _required, default in _OUTPUT_FIELDS:
        if name in data:
            val = data[name]
            # TOML gives int where float is expected — coerce
            if typ is float and isinstance(val, int):
                val = float(val)
            if not isinstance(val, typ):
                raise MetadataError(
                    f"output.toml: '{name}' expected {typ.__name__}, "
                    f"got {type(val).__name__}"
                )
            parsed[name] = val
        else:
            parsed[name] = default

    meta = InstrumentMeta(**parsed)
    _validate_output_meta(meta)
    return meta


def _validate_output_meta(meta: InstrumentMeta) -> None:
    """Raise MetadataError for semantically invalid combinations."""
    if not (0 <= meta.min_note <= 127):
        raise MetadataError("min_note must be 0–127")
    if not (0 <= meta.max_note <= 127):
        raise MetadataError("max_note must be 0–127")
    if meta.min_note > meta.max_note:
        raise MetadataError("min_note must be <= max_note")
    if not (1.0 <= meta.note_percentage <= 100.0):
        raise MetadataError("note_percentage must be 1–100")
    if not (0.0 <= meta.crossfade_percent <= 100.0):
        raise MetadataError("crossfade_percent must be 0–100")
    if meta.normalize_mode not in ("lufs", "rms", "velocity"):
        raise MetadataError("normalize_mode must be 'lufs', 'rms', or 'velocity'")
    if meta.ampeg_release < 0:
        raise MetadataError("ampeg_release must be >= 0")
    if meta.loop_crossfade_ms < 0:
        raise MetadataError("loop_crossfade_ms must be >= 0")
    if not (-12.0 <= meta.transient_attack_db <= 12.0):
        raise MetadataError("transient_attack_db must be -12 … +12 dB")
    if not (-12.0 <= meta.transient_sustain_db <= 12.0):
        raise MetadataError("transient_sustain_db must be -12 … +12 dB")
    if not (1.0 <= meta.transient_speed_ms <= 50.0):
        raise MetadataError("transient_speed_ms must be 1 … 50 ms")
