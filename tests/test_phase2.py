"""
test_phase2.py — Unit tests for Phase 2: InputModule (samples.flac slicer).

All tests generate synthetic audio files in a tmp directory so no real
instrument recordings are needed.

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf

from src.data import InputSampleDef
from src.input_module import _to_mono, _slice_samples, load_instrument
from src.metadata_parser import MetadataError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050


def make_input_toml(tmp_path: Path, defs: List[dict], collapse_to_mono: bool = False) -> Path:
    """Write an input.toml matching the given sample definitions."""
    entries = []
    for d in defs:
        is_rel = "true" if d.get("is_release", False) else "false"
        entries.append(
            f'  {{note={d["note"]}, velocity={d["velocity"]}, '
            f'hold_s={d["hold_s"]}, tail_s={d["tail_s"]}, is_release={is_rel}}}'
        )
    content = (
        f"collapse_to_mono = {'true' if collapse_to_mono else 'false'}\n"
        "samples = [\n"
        + ",\n".join(entries) + "\n"
        "]\n"
    )
    p = tmp_path / "input.toml"
    p.write_text(content)
    return p


def make_output_toml(tmp_path: Path, content: str = "") -> Path:
    p = tmp_path / "output.toml"
    p.write_text(content)
    return p


def make_samples_flac(
    tmp_path: Path,
    defs: List[dict],
    sr: int = SR,
    n_channels: int = 1,
) -> Path:
    """
    Build a synthetic samples.flac matching the given definitions.

    Each event (hold + tail) is filled with a DC value unique to
    (note, velocity, is_release) so tests can verify correct slicing.
    """
    # Compute total frames
    total_s = sum(d["hold_s"] + d["tail_s"] for d in defs)
    total_frames = int(round(total_s * sr))

    if n_channels == 1:
        audio = np.zeros(total_frames, dtype=np.float32)
    else:
        audio = np.zeros((total_frames, n_channels), dtype=np.float32)

    offset_s = 0.0
    for d in defs:
        event_start = int(round(offset_s * sr))
        hold_frames = int(round(d["hold_s"] * sr))
        tail_frames = int(round(d["tail_s"] * sr))

        # Unique DC value for the hold portion
        hold_dc = (d["note"] * 1000 + d["velocity"]) / 1_000_000.0
        # Unique DC value for the tail portion
        tail_dc = (d["note"] * 1000 + d["velocity"] + 500) / 1_000_000.0

        hold_start = event_start
        hold_end   = hold_start + hold_frames
        tail_start = hold_end
        tail_end   = tail_start + tail_frames

        if n_channels == 1:
            audio[hold_start:hold_end] = hold_dc
            audio[tail_start:tail_end] = tail_dc
        else:
            audio[hold_start:hold_end, :] = hold_dc
            audio[tail_start:tail_end, :] = tail_dc

        offset_s += d["hold_s"] + d["tail_s"]

    out = tmp_path / "samples.flac"
    sf.write(str(out), audio, sr)
    return out


def _dc(audio: np.ndarray) -> float:
    """Mean absolute value of audio — proxy for DC level of a constant signal."""
    return float(np.mean(np.abs(audio.astype(np.float64))))


# ---------------------------------------------------------------------------
# _to_mono
# ---------------------------------------------------------------------------

class TestToMono:
    def test_already_mono_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        result = _to_mono(a)
        np.testing.assert_array_equal(result, a)

    def test_stereo_averages_channels(self):
        a = np.array([[1.0, 3.0], [2.0, 4.0]])
        result = _to_mono(a)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0])

    def test_output_is_1d(self):
        a = np.ones((100, 2))
        assert _to_mono(a).ndim == 1


# ---------------------------------------------------------------------------
# _slice_samples
# ---------------------------------------------------------------------------

class TestSliceSamples:
    def _make_audio(self, defs, sr=SR):
        total_s = sum(d["hold_s"] + d["tail_s"] for d in defs)
        total_frames = int(round(total_s * sr))
        audio = np.zeros(total_frames, dtype=np.float64)
        offset_s = 0.0
        for d in defs:
            event_start = int(round(offset_s * sr))
            hold_frames = int(round(d["hold_s"] * sr))
            tail_frames = int(round(d["tail_s"] * sr))
            hold_dc = (d["note"] * 1000 + d["velocity"]) / 1_000_000.0
            tail_dc = (d["note"] * 1000 + d["velocity"] + 500) / 1_000_000.0
            audio[event_start:event_start + hold_frames] = hold_dc
            audio[event_start + hold_frames:event_start + hold_frames + tail_frames] = tail_dc
            offset_s += d["hold_s"] + d["tail_s"]
        return audio

    def _make_defs(self, raw):
        return [
            InputSampleDef(
                note=d["note"], velocity=d["velocity"],
                hold_s=d["hold_s"], tail_s=d["tail_s"],
                is_release=d.get("is_release", False),
            )
            for d in raw
        ]

    def test_sustain_sample_goes_to_sustain_list(self):
        raw = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, release = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 1
        assert len(release) == 0

    def test_release_sample_goes_to_release_list(self):
        raw = [{"note": 60, "velocity": 64, "hold_s": 0.3, "tail_s": 1.5, "is_release": True}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, release = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 0
        assert len(release) == 1

    def test_sustain_duration(self):
        hold_s = 2.0
        raw = [{"note": 60, "velocity": 64, "hold_s": hold_s, "tail_s": 0.5, "is_release": False}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        expected = int(round(hold_s * SR))
        assert sustain[0].audio.shape[0] == expected

    def test_release_duration(self):
        tail_s = 1.8
        raw = [{"note": 60, "velocity": 64, "hold_s": 0.5, "tail_s": tail_s, "is_release": True}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        _, release = _slice_samples(audio, SR, defs, 0, 127, False)
        expected = int(round(tail_s * SR))
        assert release[0].audio.shape[0] == expected

    def test_sustain_extracts_hold_portion(self):
        """Sustain slice should contain hold_dc, not tail_dc."""
        raw = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        hold_dc = (60 * 1000 + 64) / 1_000_000.0
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        assert abs(_dc(sustain[0].audio) - hold_dc) < 1e-5

    def test_release_extracts_tail_portion(self):
        """Release slice should contain tail_dc, not hold_dc."""
        raw = [{"note": 60, "velocity": 64, "hold_s": 0.5, "tail_s": 1.0, "is_release": True}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        tail_dc = (60 * 1000 + 64 + 500) / 1_000_000.0
        _, release = _slice_samples(audio, SR, defs, 0, 127, False)
        assert abs(_dc(release[0].audio) - tail_dc) < 1e-5

    def test_multiple_events_sequential_offsets(self):
        """Second event must start immediately after first (hold+tail)."""
        raw = [
            {"note": 60, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 2
        assert sustain[0].velocity == 1
        assert sustain[1].velocity == 127

    def test_note_range_filter(self):
        raw = [
            {"note": 58, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 62, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 60, 60, False)
        assert len(sustain) == 1
        assert sustain[0].midi_note == 60

    def test_filtered_event_still_advances_offset(self):
        """Events outside note range are skipped but time offsets must still be correct."""
        raw = [
            {"note": 50, "velocity": 64, "hold_s": 2.0, "tail_s": 1.0, "is_release": False},  # filtered
            {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},  # kept
        ]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        hold_dc = (60 * 1000 + 64) / 1_000_000.0
        sustain, _ = _slice_samples(audio, SR, defs, 60, 60, False)
        assert len(sustain) == 1
        assert abs(_dc(sustain[0].audio) - hold_dc) < 1e-5

    def test_mixed_sustain_and_release(self):
        raw = [
            {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 64, "hold_s": 0.3, "tail_s": 2.0, "is_release": True},
            {"note": 72, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, release = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 2
        assert len(release) == 1

    def test_non_uniform_velocities_per_note(self):
        """Different notes can have different numbers of velocity layers."""
        raw = [
            {"note": 60, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 64,  "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 5
        n60 = [s for s in sustain if s.midi_note == 60]
        n72 = [s for s in sustain if s.midi_note == 72]
        assert len(n60) == 3
        assert len(n72) == 2

    def test_collapse_to_mono(self):
        n_frames = int(round(1.5 * SR))
        audio = np.ones((n_frames, 2), dtype=np.float64) * 0.5
        raw = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, collapse_to_mono=True)
        assert sustain[0].audio.ndim == 1

    def test_stereo_preserved(self):
        n_frames = int(round(1.5 * SR))
        audio = np.ones((n_frames, 2), dtype=np.float64) * 0.5
        raw = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, collapse_to_mono=False)
        assert sustain[0].audio.ndim == 2
        assert sustain[0].audio.shape[1] == 2

    def test_sr_stored_on_sample(self):
        raw = [{"note": 60, "velocity": 64, "hold_s": 0.5, "tail_s": 0.25, "is_release": False}]
        audio = self._make_audio(raw)
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        assert sustain[0].sr == SR

    def test_bounds_clamped_gracefully(self):
        """If audio is shorter than expected, pad with zeros rather than crash."""
        raw = [{"note": 60, "velocity": 64, "hold_s": 5.0, "tail_s": 0.5, "is_release": False}]
        audio = np.zeros(100, dtype=np.float64)  # way too short
        defs = self._make_defs(raw)
        sustain, _ = _slice_samples(audio, SR, defs, 0, 127, False)
        assert len(sustain) == 1  # did not crash


# ---------------------------------------------------------------------------
# load_instrument (integration)
# ---------------------------------------------------------------------------

class TestLoadInstrument:
    def _setup(
        self,
        tmp_path: Path,
        defs: List[dict],
        output_content: str = "",
        n_channels: int = 1,
    ):
        make_input_toml(tmp_path, defs)
        make_output_toml(tmp_path, output_content)
        make_samples_flac(tmp_path, defs, sr=SR, n_channels=n_channels)

    def test_returns_instrument_data(self, tmp_path):
        from src.data import InstrumentData
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert isinstance(data, InstrumentData)

    def test_instrument_name_from_folder(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert data.meta.instrument_name == tmp_path.name

    def test_sustain_sample_count(self, tmp_path):
        defs = [
            {"note": 60, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert len(data.sustain) == 4

    def test_release_samples_separated(self, tmp_path):
        defs = [
            {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 64, "hold_s": 0.3, "tail_s": 2.0, "is_release": True},
        ]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert len(data.sustain) == 1
        assert len(data.release) == 1

    def test_no_release_gives_empty_list(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert data.release == []

    def test_sustain_duration(self, tmp_path):
        hold_s = 1.5
        defs = [{"note": 60, "velocity": 64, "hold_s": hold_s, "tail_s": 0.5, "is_release": False}]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        expected = int(round(hold_s * SR))
        assert data.sustain[0].audio.shape[0] == expected

    def test_release_duration(self, tmp_path):
        tail_s = 2.0
        defs = [{"note": 60, "velocity": 64, "hold_s": 0.3, "tail_s": tail_s, "is_release": True}]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        expected = int(round(tail_s * SR))
        assert data.release[0].audio.shape[0] == expected

    def test_note_range_filter_from_output_toml(self, tmp_path):
        defs = [
            {"note": 58, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 62, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        self._setup(tmp_path, defs, output_content="min_note=60\nmax_note=60\n")
        data = load_instrument(tmp_path)
        assert data.sustain_notes() == [60]

    def test_non_uniform_velocities_preserved(self, tmp_path):
        defs = [
            {"note": 60, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 40,  "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 80,  "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 1,   "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
            {"note": 72, "velocity": 127, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
        ]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        n60_vels = sorted(s.velocity for s in data.sustain if s.midi_note == 60)
        n72_vels = sorted(s.velocity for s in data.sustain if s.midi_note == 72)
        assert n60_vels == [1, 40, 80, 127]
        assert n72_vels == [1, 127]

    def test_different_hold_tail_per_sample(self, tmp_path):
        """Each sample can have different hold_s and tail_s."""
        defs = [
            {"note": 60, "velocity": 64,  "hold_s": 2.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 5.0, "tail_s": 1.0, "is_release": False},
        ]
        self._setup(tmp_path, defs)
        data = load_instrument(tmp_path)
        assert len(data.sustain) == 2
        s64  = next(s for s in data.sustain if s.velocity == 64)
        s127 = next(s for s in data.sustain if s.velocity == 127)
        assert s64.audio.shape[0]  == int(round(2.0 * SR))
        assert s127.audio.shape[0] == int(round(5.0 * SR))

    def test_collapse_to_mono(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        make_input_toml(tmp_path, defs, collapse_to_mono=True)
        make_output_toml(tmp_path)
        make_samples_flac(tmp_path, defs, sr=SR, n_channels=2)
        data = load_instrument(tmp_path)
        assert data.sustain[0].audio.ndim == 1

    def test_stereo_preserved_without_collapse(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        self._setup(tmp_path, defs, n_channels=2)
        data = load_instrument(tmp_path)
        assert data.sustain[0].audio.ndim == 2

    def test_missing_samples_flac_raises(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        make_input_toml(tmp_path, defs)
        make_output_toml(tmp_path)
        with pytest.raises(FileNotFoundError, match="samples"):
            load_instrument(tmp_path)

    def test_missing_input_toml_raises(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        make_output_toml(tmp_path)
        make_samples_flac(tmp_path, defs)
        with pytest.raises(FileNotFoundError, match="input.toml"):
            load_instrument(tmp_path)

    def test_missing_output_toml_raises(self, tmp_path):
        defs = [{"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False}]
        make_input_toml(tmp_path, defs)
        make_samples_flac(tmp_path, defs)
        with pytest.raises(FileNotFoundError, match="output.toml"):
            load_instrument(tmp_path)
