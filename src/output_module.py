"""
output_module.py — Write a fully-processed InstrumentData to disk.

Output structure:
    {output_root}/{instrument_name}/
        {instrument_name}_sustain.sfz
        {instrument_name}_release.sfz   (if release samples exist)
        {instrument_name}-sustain/
            {instrument_name}_{note_name}_v{velocity}.flac
        {instrument_name}-release/
            {instrument_name}_{note_name}_v{velocity}_rel.flac

Velocity zones are built per-note from whatever velocities were actually
recorded for that note — fully supporting non-uniform velocity layers.

Entry point: write_instrument(data, output_root) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf

from .data import InstrumentData, InstrumentMeta, Sample


# ---------------------------------------------------------------------------
# MIDI note name helper
# ---------------------------------------------------------------------------

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI note number to note name string, e.g. 60 -> 'C4'."""
    octave = (midi // 12) - 1
    return f"{_NOTE_NAMES[midi % 12]}{octave}"


# ---------------------------------------------------------------------------
# Note subset selection
# ---------------------------------------------------------------------------

def select_notes(notes: List[int], note_percentage: float) -> List[int]:
    """
    Return an evenly-spaced subset of `notes` covering note_percentage% of them.

    `notes` must be sorted. Always includes the first and last notes when
    the target count >= 2. For a target of 1 the middle note is returned.
    """
    n = len(notes)
    if n == 0:
        return []
    target = max(1, round(note_percentage / 100.0 * n))
    if target >= n:
        return list(notes)
    if target == 1:
        return [notes[n // 2]]
    indices = {round(i * (n - 1) / (target - 1)) for i in range(target)}
    return [notes[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Key / velocity zone computation
# ---------------------------------------------------------------------------

def compute_note_ranges(
    selected_notes: List[int], min_note: int, max_note: int
) -> List[dict]:
    """
    For a sorted list of selected MIDI notes, compute lokey/hikey for each.

    Boundaries sit at midpoints between adjacent notes. Outer edges clamp to
    min_note / max_note so the full requested range is covered.

    Returns list of dicts: {midi_note, lokey, hikey}.
    """
    notes = sorted(selected_notes)
    n = len(notes)
    result = []
    for i, note in enumerate(notes):
        lo = min_note if i == 0 else (notes[i - 1] + note) // 2 + 1
        hi = max_note if i == n - 1 else (note + notes[i + 1]) // 2
        result.append({"midi_note": note, "lokey": lo, "hikey": hi})
    return result


def compute_velocity_ranges(
    velocities: List[int], crossfade_percent: float
) -> List[dict]:
    """
    For a sorted list of velocities, compute SFZ velocity zone boundaries
    with optional crossfade overlap.

    Each velocity value is the MINIMUM (floor) of its zone:
        zone[i] = [vels[i], vels[i+1] - 1]   (last zone extends to 127)

    crossfade_percent (0–100) is the fraction of each zone's natural width
    that extends into adjacent zones, enabling xfin/xfout crossfading.

    Returns list of dicts with: velocity, lovel, hivel,
    xfin_lovel, xfin_hivel, xfout_lovel, xfout_hivel,
    has_xfin (bool), has_xfout (bool).
    """
    vels = sorted(velocities)
    n = len(vels)
    result = []
    for i, vc in enumerate(vels):
        lo_nat = vc
        hi_nat = 127 if i == n - 1 else vels[i + 1] - 1

        zone_width = hi_nat - lo_nat + 1
        xfade = int(crossfade_percent / 100.0 * zone_width)

        lovel = max(1, lo_nat - xfade) if i > 0 else 1
        hivel = min(127, hi_nat + xfade) if i < n - 1 else 127

        result.append({
            "velocity": vc,
            "lovel": lovel,
            "hivel": hivel,
            "xfin_lovel":  lovel,
            "xfin_hivel":  lo_nat,
            "xfout_lovel": hi_nat,
            "xfout_hivel": hivel,
            "has_xfin":  i > 0 and xfade > 0,
            "has_xfout": i < n - 1 and xfade > 0,
        })
    return result


# ---------------------------------------------------------------------------
# rt_decay estimation (for release SFZ)
# ---------------------------------------------------------------------------

def estimate_rt_decay(audio: np.ndarray, sr: int) -> float:
    """
    Estimate amplitude decay rate of a release sample in dB/second.

    Uses 20 ms RMS windows, fits a linear slope on the first 80% of windows,
    and clamps the result to [1.0, 24.0] dB/s.
    """
    mono = audio if audio.ndim == 1 else audio.mean(axis=1)
    window = max(1, int(0.02 * sr))
    n_windows = len(mono) // window
    if n_windows < 2:
        return 6.0

    rms_db = []
    for i in range(n_windows):
        chunk = mono[i * window:(i + 1) * window]
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        rms_db.append(20.0 * np.log10(max(rms, 1e-9)))

    times = np.arange(n_windows) * (window / sr)
    db_arr = np.array(rms_db)
    n_use = max(2, int(n_windows * 0.8))
    slope, _ = np.polyfit(times[:n_use], db_arr[:n_use], 1)
    return float(np.clip(-slope, 1.0, 24.0))


# ---------------------------------------------------------------------------
# amp_velcurve_1 helper
# ---------------------------------------------------------------------------

def _amp_velcurve_1(meta: InstrumentMeta) -> float:
    """
    Compute the SFZ amp_velcurve_1 value (gain at velocity=1 relative to 127).

    In velocity mode the dynamics are already baked into the audio by
    NormalizeProcessor, so amp_velcurve_1 is set to 1.0 (flat SFZ curve).
    In lufs/rms modes, velocity_dynamic_range_db drives the curve.
    """
    if meta.normalize_mode == "velocity":
        return 1.0
    range_db = meta.velocity_dynamic_range_db
    return 10.0 ** (-range_db / 20.0)


# ---------------------------------------------------------------------------
# SFZ text generation
# ---------------------------------------------------------------------------

def generate_sustain_sfz(regions: List[dict], meta: InstrumentMeta) -> str:
    """
    Generate sustain SFZ 1.0 text.

    Each region dict must have: filename, lokey, hikey, pitch_keycenter,
    lovel, hivel, has_xfin, has_xfout, xfin_lovel, xfin_hivel,
    xfout_lovel, xfout_hivel.
    Optional: loop_start, loop_end (integers or None).
    """
    amp_vel1 = _amp_velcurve_1(meta)
    lines = [
        "<control>",
        f"default_path={meta.instrument_name}-sustain/",
        "",
        "<group>",
        f"ampeg_release={meta.ampeg_release:.3f}",
        f"amp_velcurve_1={amp_vel1:.6f}",
        "",
    ]
    for r in regions:
        lines.append("<region>")
        lines.append(f"sample={r['filename']}")
        lines.append(
            f"lokey={r['lokey']}  hikey={r['hikey']}  pitch_keycenter={r['pitch_keycenter']}"
        )
        lines.append(f"lovel={r['lovel']}  hivel={r['hivel']}")
        if r.get("has_xfin"):
            lines.append(f"xfin_lovel={r['xfin_lovel']}  xfin_hivel={r['xfin_hivel']}")
        if r.get("has_xfout"):
            lines.append(f"xfout_lovel={r['xfout_lovel']}  xfout_hivel={r['xfout_hivel']}")
        if r.get("loop_start") is not None:
            lines.append("loop_mode=loop_continuous")
            lines.append(f"loop_start={r['loop_start']}")
            lines.append(f"loop_end={r['loop_end']}")
            if meta.loop_crossfade_ms > 0:
                lines.append(f"loop_crossfade={meta.loop_crossfade_ms / 1000.0:.4f}")
        lines.append("")
    return "\n".join(lines)


def generate_release_sfz(regions: List[dict], meta: InstrumentMeta) -> str:
    """
    Generate release SFZ 1.0 text.

    Each region dict must have: filename, lokey, hikey, pitch_keycenter,
    lovel, hivel, rt_decay.
    """
    amp_vel1 = _amp_velcurve_1(meta)
    lines = [
        "<control>",
        f"default_path={meta.instrument_name}-release/",
        "",
        "<group>",
        "trigger=release",
        f"ampeg_attack={meta.ampeg_release:.3f}",
        "ampeg_sustain=0",
        "ampeg_decay=1.000",
        "ampeg_release=0.300",
        f"amp_velcurve_1={amp_vel1:.6f}",
        "",
    ]
    for r in regions:
        lines.append("<region>")
        lines.append(f"sample={r['filename']}")
        lines.append(
            f"lokey={r['lokey']}  hikey={r['hikey']}  pitch_keycenter={r['pitch_keycenter']}"
        )
        lines.append(f"lovel={r['lovel']}  hivel={r['hivel']}")
        lines.append(f"rt_decay={r['rt_decay']:.2f}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FLAC encoding
# ---------------------------------------------------------------------------

def encode_flac(audio: np.ndarray, sr: int, path: Path) -> None:
    """Write audio to a 24-bit FLAC file at `path`. Clips to [-1, 1] first."""
    data = np.clip(audio, -1.0, 1.0).astype(np.float64)
    sf.write(str(path), data, sr, format="FLAC", subtype="PCM_24")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def write_instrument(data: InstrumentData, output_root: Path) -> dict:
    """
    Write a fully-processed InstrumentData to disk under output_root.

    Steps:
      1. Apply note subset selection (note_percentage within [min_note, max_note])
      2. For each selected note, build velocity zones from that note's recorded
         velocities (fully supports non-uniform layers across notes)
      3. Encode each sustain sample to 24-bit FLAC
      4. Generate and write {instrument_name}_sustain.sfz
      5. If release samples exist: encode + generate {instrument_name}_release.sfz

    Args:
        data:        Fully processed InstrumentData (all processors have run).
        output_root: Root directory; instrument folder is created inside it.

    Returns dict with: instrument_name, output_dir, files_written,
                       sustain_regions, release_regions.
    """
    meta = data.meta
    name = meta.instrument_name

    out_dir     = output_root / name
    samples_dir = out_dir / f"{name}-sustain"
    release_dir = out_dir / f"{name}-release"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ── Note subset selection ─────────────────────────────────────────────
    all_notes = sorted(set(
        s.midi_note for s in data.sustain
        if meta.min_note <= s.midi_note <= meta.max_note
    ))
    selected_notes = select_notes(all_notes, meta.note_percentage)

    note_ranges = compute_note_ranges(selected_notes, meta.min_note, meta.max_note)
    note_to_range: Dict[int, dict] = {nr["midi_note"]: nr for nr in note_ranges}

    # ── Sustain samples ───────────────────────────────────────────────────
    sfz_regions: List[dict] = []
    files_written = 0

    for midi_note in selected_notes:
        nrng = note_to_range[midi_note]
        note_name = midi_to_note_name(midi_note)

        note_samples = sorted(
            [s for s in data.sustain if s.midi_note == midi_note],
            key=lambda s: s.velocity,
        )
        if not note_samples:
            continue

        # Per-note velocity zones — built from this note's actual velocities
        note_vels = [s.velocity for s in note_samples]
        vel_ranges = compute_velocity_ranges(note_vels, meta.crossfade_percent)
        vel_to_range: Dict[int, dict] = {vr["velocity"]: vr for vr in vel_ranges}

        for sample in note_samples:
            vrng = vel_to_range[sample.velocity]
            filename = f"{name}_{note_name}_v{sample.velocity}.flac"
            encode_flac(sample.audio, sample.sr, samples_dir / filename)
            files_written += 1

            sfz_regions.append({
                "filename":        filename,
                "lokey":           nrng["lokey"],
                "hikey":           nrng["hikey"],
                "pitch_keycenter": midi_note,
                "lovel":           vrng["lovel"],
                "hivel":           vrng["hivel"],
                "has_xfin":        vrng["has_xfin"],
                "has_xfout":       vrng["has_xfout"],
                "xfin_lovel":      vrng["xfin_lovel"],
                "xfin_hivel":      vrng["xfin_hivel"],
                "xfout_lovel":     vrng["xfout_lovel"],
                "xfout_hivel":     vrng["xfout_hivel"],
                "loop_start":      sample.loop_start,
                "loop_end":        sample.loop_end,
            })

    sustain_sfz_text = generate_sustain_sfz(sfz_regions, meta)
    (out_dir / f"{name}_sustain.sfz").write_text(sustain_sfz_text, encoding="utf-8")

    # Velocity mode report
    if meta.normalize_mode == "velocity" and meta.computed_dynamic_range_db is not None:
        range_db = meta.computed_dynamic_range_db
        _MAX_DB = 48.0
        capped = range_db > _MAX_DB
        recommended_db = _MAX_DB if capped else range_db
        if capped:
            extra_note = (
                f"Note: measured range ({range_db:.1f} dB) exceeds AudioLayer's\n"
                f"48 dB velocity curve limit. Normalization was capped at 48 dB;\n"
                f"the remaining {range_db - _MAX_DB:.1f} dB of natural variation is\n"
                f"preserved in the audio. Set velocity_dynamic_range to 48 dB.\n"
            )
        else:
            extra_note = (
                f"Set velocity_dynamic_range to {recommended_db:.1f} dB.\n"
            )
        report = (
            f"Velocity Dynamic Range Report — {name}\n"
            f"{'=' * 50}\n\n"
            f"Measured dynamic range : {range_db:.1f} dB\n"
            f"Recommended setting    : {recommended_db:.1f} dB\n\n"
            f"Dynamics are fully baked into audio samples.\n"
            f"SFZ amp_velcurve_1 = 1.0 (flat — no additional scaling).\n\n"
            + extra_note
        )
        (out_dir / "velocity_range.txt").write_text(report, encoding="utf-8")

    # ── Release samples ───────────────────────────────────────────────────
    release_sfz_regions: List[dict] = []

    if data.release:
        release_dir.mkdir(parents=True, exist_ok=True)

        # Note ranges for release — built from release notes actually present
        all_rel_notes = sorted(set(
            s.midi_note for s in data.release
            if meta.min_note <= s.midi_note <= meta.max_note
        ))
        rel_selected_notes = select_notes(all_rel_notes, meta.note_percentage)
        rel_note_ranges = compute_note_ranges(
            rel_selected_notes, meta.min_note, meta.max_note
        )
        rel_note_to_range: Dict[int, dict] = {
            nr["midi_note"]: nr for nr in rel_note_ranges
        }

        for midi_note in rel_selected_notes:
            nrng = rel_note_to_range[midi_note]
            note_name = midi_to_note_name(midi_note)

            rel_samples = sorted(
                [s for s in data.release if s.midi_note == midi_note],
                key=lambda s: s.velocity,
            )
            if not rel_samples:
                continue

            # Per-note velocity zones for release
            rel_vels = [s.velocity for s in rel_samples]
            rel_vel_ranges = compute_velocity_ranges(rel_vels, meta.crossfade_percent)
            rel_vel_to_range: Dict[int, dict] = {
                vr["velocity"]: vr for vr in rel_vel_ranges
            }

            for sample in rel_samples:
                vrng = rel_vel_to_range[sample.velocity]
                filename = f"{name}_{note_name}_v{sample.velocity}_rel.flac"
                encode_flac(sample.audio, sample.sr, release_dir / filename)
                files_written += 1

                rt_decay = estimate_rt_decay(sample.audio, sample.sr)
                release_sfz_regions.append({
                    "filename":        filename,
                    "lokey":           nrng["lokey"],
                    "hikey":           nrng["hikey"],
                    "pitch_keycenter": midi_note,
                    "lovel":           vrng["lovel"],
                    "hivel":           vrng["hivel"],
                    "rt_decay":        rt_decay,
                })

        if release_sfz_regions:
            release_sfz_text = generate_release_sfz(release_sfz_regions, meta)
            (out_dir / f"{name}_release.sfz").write_text(
                release_sfz_text, encoding="utf-8"
            )

    return {
        "instrument_name":  name,
        "output_dir":       str(out_dir),
        "files_written":    files_written,
        "sustain_regions":  len(sfz_regions),
        "release_regions":  len(release_sfz_regions),
    }
