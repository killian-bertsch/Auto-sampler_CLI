"""
metadata_parser.py — Parse an instrument's metadata.txt into InstrumentMeta.

Format rules:
  - Lines starting with # (after stripping whitespace) are comments, ignored.
  - Blank lines are ignored.
  - Each data line is:  key = value
  - Keys are case-insensitive; stored as lowercase internally.
  - Values are stripped of surrounding whitespace.
  - Booleans: 'true'/'false' (case-insensitive).
  - Numbers: parsed as int or float based on field type.
  - Strings: kept as-is.

Raises MetadataError with a descriptive message on any parse/validation failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .data import InstrumentMeta


class MetadataError(ValueError):
    """Raised when metadata.txt is missing, malformed, or has invalid values."""


# ---------------------------------------------------------------------------
# Field specification table
# Each entry: (key_name, python_type, required, default_or_None)
#
# Optional float fields whose default is None are allowed to be absent from
# the file; the input module resolves them at load time via the
# effective_*() helpers on InstrumentMeta.
# ---------------------------------------------------------------------------
_FIELDS: list[tuple[str, type, bool, Any]] = [
    # Recording parameters — sustain.wav
    ("velocity_layers",          int,   True,  None),
    ("semitone_interval",        int,   True,  None),
    ("hold_time",                float, True,  None),
    ("release_time",             float, True,  None),
    ("start_note",               int,   False, 21),
    ("end_note",                 int,   False, 108),
    # Recording parameters — release.wav overrides (optional)
    # If absent, input module falls back to hold_time / release_time
    ("release_hold_time",        float, False, None),
    ("release_release_time",     float, False, None),
    # Output selection
    ("min_note",                 int,   False, 21),
    ("max_note",                 int,   False, 108),
    ("note_percentage",          float, False, 100.0),
    ("velocity_layers_out",      int,   True,  None),
    # Crossfade
    ("crossfade_percent",        float, False, 0.0),
    # Normalization
    ("normalize",                bool,  False, True),
    ("normalize_mode",           str,   False, "lufs"),
    ("normalize_target_lufs",    float, False, -18.0),
    ("peak_ceiling_db",          float, False, -1.0),
    ("velocity_dynamic_range_db",float, False, 40.0),
    # SFZ / Playback
    ("instrument_name",          str,   True,  None),
    ("ampeg_release",            float, False, 0.5),
    ("loop_crossfade_ms",        float, False, 20.0),
    # Advanced
    ("onset_threshold_db",       float, False, -40.0),
    ("collapse_to_mono",         bool,  False, False),
]

_FIELD_TYPES: Dict[str, type] = {name: typ for name, typ, _, _ in _FIELDS}
_FIELD_REQUIRED: Dict[str, bool] = {name: req for name, _, req, _ in _FIELDS}
_FIELD_DEFAULTS: Dict[str, Any] = {name: default for name, _, _, default in _FIELDS}


def _coerce(key: str, raw_value: str) -> Any:
    """Convert a raw string value to the correct Python type for `key`."""
    typ = _FIELD_TYPES[key]
    val = raw_value.strip()
    try:
        if typ is bool:
            if val.lower() == "true":
                return True
            elif val.lower() == "false":
                return False
            else:
                raise MetadataError(
                    f"'{key}': expected true/false, got '{val}'"
                )
        elif typ is int:
            return int(val)
        elif typ is float:
            return float(val)
        else:  # str
            return val
    except (ValueError, TypeError) as exc:
        raise MetadataError(
            f"'{key}': cannot convert '{val}' to {typ.__name__}: {exc}"
        ) from exc


def parse_metadata(path: str | Path) -> InstrumentMeta:
    """
    Parse a metadata.txt file and return an InstrumentMeta.

    Args:
        path: Path to the metadata.txt file.

    Returns:
        InstrumentMeta with all fields populated.

    Raises:
        MetadataError: if the file is missing, unreadable, or contains
                       invalid values.
        FileNotFoundError: if path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metadata.txt not found: {path}")

    raw: Dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise MetadataError(
                    f"Line {lineno}: expected 'key = value', got: {line!r}"
                )
            key, _, value = line.partition("=")
            key = key.strip().lower()
            value = value.strip()
            if key not in _FIELD_TYPES:
                # Unknown keys are silently ignored (forward-compatibility)
                continue
            raw[key] = value

    # Check required fields
    for name in _FIELD_TYPES:
        if _FIELD_REQUIRED[name] and name not in raw:
            raise MetadataError(f"Required field '{name}' is missing from metadata.txt")

    # Build parsed dict, applying defaults for optional missing fields
    parsed: Dict[str, Any] = {}
    for name in _FIELD_TYPES:
        if name in raw:
            parsed[name] = _coerce(name, raw[name])
        else:
            parsed[name] = _FIELD_DEFAULTS[name]

    # Semantic validation
    meta = InstrumentMeta(**parsed)
    _validate(meta)
    return meta


def _validate(meta: InstrumentMeta) -> None:
    """Raise MetadataError for semantically invalid combinations."""
    if meta.velocity_layers < 1:
        raise MetadataError("velocity_layers must be >= 1")
    if meta.semitone_interval < 1:
        raise MetadataError("semitone_interval must be >= 1")
    if meta.hold_time <= 0:
        raise MetadataError("hold_time must be > 0")
    if meta.release_time < 0:
        raise MetadataError("release_time must be >= 0")
    if meta.release_hold_time is not None and meta.release_hold_time <= 0:
        raise MetadataError("release_hold_time must be > 0")
    if meta.release_release_time is not None and meta.release_release_time < 0:
        raise MetadataError("release_release_time must be >= 0")
    if not (0 <= meta.start_note <= 127):
        raise MetadataError("start_note must be 0–127")
    if not (0 <= meta.end_note <= 127):
        raise MetadataError("end_note must be 0–127")
    if meta.start_note > meta.end_note:
        raise MetadataError("start_note must be <= end_note")
    if not (0 <= meta.min_note <= 127):
        raise MetadataError("min_note must be 0–127")
    if not (0 <= meta.max_note <= 127):
        raise MetadataError("max_note must be 0–127")
    if meta.min_note > meta.max_note:
        raise MetadataError("min_note must be <= max_note")
    if not (1.0 <= meta.note_percentage <= 100.0):
        raise MetadataError("note_percentage must be 1–100")
    if meta.velocity_layers_out < 1:
        raise MetadataError("velocity_layers_out must be >= 1")
    if meta.velocity_layers_out > meta.velocity_layers:
        raise MetadataError(
            f"velocity_layers_out ({meta.velocity_layers_out}) cannot exceed "
            f"velocity_layers ({meta.velocity_layers})"
        )
    if not (0.0 <= meta.crossfade_percent <= 100.0):
        raise MetadataError("crossfade_percent must be 0–100")
    if meta.normalize_mode not in ("lufs", "rms", "velocity"):
        raise MetadataError("normalize_mode must be 'lufs', 'rms', or 'velocity'")
    if meta.ampeg_release < 0:
        raise MetadataError("ampeg_release must be >= 0")
    if meta.loop_crossfade_ms < 0:
        raise MetadataError("loop_crossfade_ms must be >= 0")
