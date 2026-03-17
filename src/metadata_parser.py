"""
metadata_parser.py — Parse an instrument's metadata.toml (or legacy metadata.txt)
into InstrumentMeta.

Preferred format: TOML (metadata.toml)
Legacy format:    key = value lines with # comments (metadata.txt)

Raises MetadataError with a descriptive message on any parse/validation failure.
"""

from __future__ import annotations

import tomllib
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
    ("instrument_name",          str,   False, ""),
    ("ampeg_release",            float, False, 0.5),
    ("loop_crossfade_ms",        float, False, 20.0),
    # Advanced
    ("pre_trim_ms",              float, False, 0.0),
    ("collapse_to_mono",         bool,  False, False),
    # Transient shaper
    ("enable_transient_shaper",  bool,  False, False),
    ("transient_attack_db",      float, False, 6.0),
    ("transient_sustain_db",     float, False, 0.0),
    ("transient_speed_ms",       float, False, 10.0),
    # Optional velocity map — overrides velocity_layers_out when present
    # Format: "lo-hi:n, lo-hi:n, ..."  e.g. "0-63:1, 64-127:5"
    ("velocity_map",             str,   False, None),
]

_FIELD_TYPES: Dict[str, type] = {name: typ for name, typ, _, _ in _FIELDS}
_FIELD_REQUIRED: Dict[str, bool] = {name: req for name, _, req, _ in _FIELDS}
_FIELD_DEFAULTS: Dict[str, Any] = {name: default for name, _, _, default in _FIELDS}


def _coerce(key: str, raw_value: Any) -> Any:
    """Convert a value (string from .txt or native type from .toml) to the correct type."""
    typ = _FIELD_TYPES[key]

    # TOML already gives us the right Python type — validate and return directly
    if not isinstance(raw_value, str):
        if typ is float and isinstance(raw_value, int):
            return float(raw_value)  # TOML integer where float is expected
        if isinstance(raw_value, typ):
            return raw_value
        raise MetadataError(
            f"'{key}': expected {typ.__name__}, got {type(raw_value).__name__}"
        )

    # Legacy .txt path — parse from string
    val = raw_value.strip()
    try:
        if typ is bool:
            if val.lower() == "true":
                return True
            elif val.lower() == "false":
                return False
            else:
                raise MetadataError(f"'{key}': expected true/false, got '{val}'")
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
    Parse a metadata.toml or legacy metadata.txt file and return an InstrumentMeta.

    Lookup order when a directory is given:
      1. metadata.toml
      2. metadata.txt  (legacy)

    Raises:
        FileNotFoundError: if neither file exists.
        MetadataError: if the file is malformed or contains invalid values.
    """
    path = Path(path)

    # If a directory is passed, resolve to the metadata file inside it
    if path.is_dir():
        toml_path = path / "metadata.toml"
        txt_path  = path / "metadata.txt"
        if toml_path.exists():
            path = toml_path
        elif txt_path.exists():
            path = txt_path
        else:
            raise FileNotFoundError(f"No metadata.toml or metadata.txt in: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Metadata file not found: {path}")

    if path.suffix == ".toml":
        raw = _parse_toml(path)
    else:
        raw = _parse_txt(path)

    # Check required fields
    for name in _FIELD_TYPES:
        if _FIELD_REQUIRED[name] and name not in raw:
            raise MetadataError(f"Required field '{name}' is missing")

    # Build parsed dict, applying defaults for optional missing fields
    parsed: Dict[str, Any] = {}
    for name in _FIELD_TYPES:
        if name in raw:
            parsed[name] = _coerce(name, raw[name])
        else:
            parsed[name] = _FIELD_DEFAULTS[name]

    meta = InstrumentMeta(**parsed)
    _validate(meta)
    return meta


def _parse_toml(path: Path) -> Dict[str, Any]:
    """Load a TOML metadata file; returns a flat dict of raw (already-typed) values."""
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise MetadataError(f"TOML parse error in {path.name}: {exc}") from exc

    # TOML already gives us typed values — convert to strings so _coerce handles them
    # uniformly, or just return as-is and let _coerce skip re-parsing native types.
    flat: Dict[str, Any] = {}
    for key, val in data.items():
        key = key.lower()
        if key not in _FIELD_TYPES:
            continue  # unknown keys silently ignored
        flat[key] = val
    return flat


def _parse_txt(path: Path) -> Dict[str, str]:
    """Load a legacy key = value metadata.txt; returns a flat dict of raw strings."""
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
                continue
            raw[key] = value
    return raw


def parse_velocity_map(raw: str) -> list[tuple[int, int, int]]:
    """
    Parse a velocity_map string into a list of (lo, hi, n) tuples.

    Format: "lo-hi:n, lo-hi:n, ..."
      lo  — minimum velocity of the segment (0–127)
      hi  — maximum velocity of the segment (0–127)
      n   — number of layers to select from recorded velocities in [lo, hi]

    Returns list of (lo, hi, n) sorted by lo.

    Raises MetadataError on any syntax or value error.
    """
    segments = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            range_part, _, n_str = part.partition(":")
            lo_str, _, hi_str = range_part.partition("-")
            lo, hi, n = int(lo_str.strip()), int(hi_str.strip()), int(n_str.strip())
        except ValueError:
            raise MetadataError(
                f"velocity_map: cannot parse segment '{part}' — "
                "expected format 'lo-hi:n' e.g. '0-63:1'"
            )
        if not (0 <= lo <= 127 and 0 <= hi <= 127):
            raise MetadataError(
                f"velocity_map: segment '{part}' — lo/hi must be 0–127"
            )
        if lo > hi:
            raise MetadataError(
                f"velocity_map: segment '{part}' — lo must be <= hi"
            )
        if n < 1:
            raise MetadataError(
                f"velocity_map: segment '{part}' — layer count must be >= 1"
            )
        segments.append((lo, hi, n))
    if not segments:
        raise MetadataError("velocity_map: must contain at least one segment")
    segments.sort(key=lambda s: s[0])
    return segments


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
    if not (-12.0 <= meta.transient_attack_db <= 12.0):
        raise MetadataError("transient_attack_db must be -12 … +12 dB")
    if not (-12.0 <= meta.transient_sustain_db <= 12.0):
        raise MetadataError("transient_sustain_db must be -12 … +12 dB")
    if not (1.0 <= meta.transient_speed_ms <= 50.0):
        raise MetadataError("transient_speed_ms must be 1 … 50 ms")
    if meta.velocity_map is not None:
        # Validates syntax; raises MetadataError on bad format
        parse_velocity_map(meta.velocity_map)
