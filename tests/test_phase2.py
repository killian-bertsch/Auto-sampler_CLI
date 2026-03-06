"""
test_phase2.py — Unit tests for Phase 2: InputModule (WAV slicer).

All tests generate synthetic WAV files in a tmp directory so no real
instrument recordings are needed.

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase2.py -v
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.input_module import (
    _compute_velocities,
    _get_recorded_notes,
    _to_mono,
    _slice_wav,
    load_instrument,
)
from src.metadata_parser import MetadataError


# ---------------------------------------------------------------------------
# Minimal metadata string helpers
# ---------------------------------------------------------------------------

MINIMAL_META = """\
velocity_layers = {x}
semitone_interval = {n}
hold_time = {h}
release_time = {r}
start_note = {start}
end_note = {end}
velocity_layers_out = {x}
instrument_name = TestPiano
"""


def write_meta(tmp_path: Path, **kwargs) -> Path:
    defaults = dict(x=2, n=12, h=1.0, r=0.5, start=60, end=72)
    defaults.update(kwargs)
    content = MINIMAL_META.format(**defaults)
    p = tmp_path / "metadata.txt"
    p.write_text(textwrap.dedent(content))
    return p


def make_long_wav(
    tmp_path: Path,
    filename: str,
    notes: list[int],
    velocities: list[int],
    hold_time: float,
    release_time: float,
    sr: int = 22050,
    n_channels: int = 1,
    extract_release: bool = False,
) -> Path:
    """
    Build a synthetic long WAV that mimics a rendered autosample session.
    Each note-event slot is filled with a DC tone at a unique amplitude so
    tests can verify the correct slice was extracted.
    """
    note_duration = hold_time + release_time
    frames_per_event = int(round(note_duration * sr))
    frames_hold = int(round(hold_time * sr))
    frames_release = int(round(release_time * sr))

    total_events = len(notes) * len(velocities)
    total_frames = total_events * frames_per_event

    if n_channels == 1:
        audio = np.zeros(total_frames, dtype=np.float32)
    else:
        audio = np.zeros((total_frames, n_channels), dtype=np.float32)

    event_index = 0
    for note in notes:
        for vel in velocities:
            # Use a unique DC value so we can assert which slice was picked
            dc_value = (note * 1000 + vel) / 1_000_000.0
            event_start = int(round(event_index * note_duration * sr))

            if extract_release:
                seg_start = event_start + frames_hold
                seg_len = frames_release
            else:
                seg_start = event_start
                seg_len = frames_hold

            seg_end = seg_start + seg_len
            if n_channels == 1:
                audio[seg_start:seg_end] = dc_value
            else:
                audio[seg_start:seg_end, :] = dc_value

            event_index += 1

    out_path = tmp_path / filename
    sf.write(str(out_path), audio, sr)
    return out_path


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestComputeVelocities:
    def test_single_layer(self):
        assert _compute_velocities(1) == [127]

    def test_two_layers(self):
        assert _compute_velocities(2) == [1, 127]

    def test_four_layers_endpoints(self):
        v = _compute_velocities(4)
        assert v[0] == 1
        assert v[-1] == 127
        assert len(v) == 4

    def test_four_layers_monotonic(self):
        v = _compute_velocities(4)
        assert v == sorted(v)

    def test_eight_layers(self):
        v = _compute_velocities(8)
        assert len(v) == 8
        assert v[0] == 1
        assert v[-1] == 127


class TestGetRecordedNotes:
    def test_every_semitone(self):
        notes = _get_recorded_notes(60, 72, 1)
        assert notes == list(range(60, 73))

    def test_every_octave(self):
        notes = _get_recorded_notes(21, 108, 12)
        assert 21 in notes
        assert all((n - 21) % 12 == 0 for n in notes)

    def test_interval_3(self):
        notes = _get_recorded_notes(60, 72, 3)
        assert notes == [60, 63, 66, 69, 72]

    def test_single_note(self):
        notes = _get_recorded_notes(60, 60, 1)
        assert notes == [60]


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
# _slice_wav
# ---------------------------------------------------------------------------

class TestSliceWav:
    SR = 22050

    def _make_audio(self, notes, velocities, hold, release, extract_release=False):
        """Build a synthetic long WAV array matching the given params."""
        note_duration = hold + release
        frames_hold = int(round(hold * self.SR))
        frames_release = int(round(release * self.SR))
        total_frames = len(notes) * len(velocities) * int(round(note_duration * self.SR))
        audio = np.zeros(total_frames, dtype=np.float64)
        idx = 0
        for note in notes:
            for vel in velocities:
                ev_start = int(round(idx * note_duration * self.SR))
                dc = (note * 1000 + vel) / 1_000_000.0
                if extract_release:
                    s = ev_start + frames_hold
                    audio[s:s + frames_release] = dc
                else:
                    audio[ev_start:ev_start + frames_hold] = dc
                idx += 1
        return audio

    def test_correct_note_count(self):
        notes = [60, 72]
        vels = [1, 127]
        audio = self._make_audio(notes, vels, hold=1.0, release=0.5)
        samples = _slice_wav(audio, self.SR, notes, vels, 1.0, 0.5,
                             extract_release=False, min_note=60, max_note=72,
                             collapse_to_mono=False)
        assert len(samples) == 4  # 2 notes × 2 velocities

    def test_min_max_note_filtering(self):
        notes = [60, 64, 68, 72]
        vels = [127]
        audio = self._make_audio(notes, vels, hold=0.5, release=0.25)
        samples = _slice_wav(audio, self.SR, notes, vels, 0.5, 0.25,
                             extract_release=False, min_note=64, max_note=68,
                             collapse_to_mono=False)
        returned_notes = [s.midi_note for s in samples]
        assert returned_notes == [64, 68]

    def test_sustain_slice_duration(self):
        notes = [60]
        vels = [127]
        hold = 2.0
        audio = self._make_audio(notes, vels, hold=hold, release=0.5)
        samples = _slice_wav(audio, self.SR, notes, vels, hold, 0.5,
                             extract_release=False, min_note=60, max_note=60,
                             collapse_to_mono=False)
        expected_frames = int(round(hold * self.SR))
        assert samples[0].audio.shape[0] == expected_frames

    def test_release_slice_duration(self):
        notes = [60]
        vels = [127]
        release = 1.5
        audio = self._make_audio(notes, vels, hold=2.0, release=release,
                                 extract_release=True)
        samples = _slice_wav(audio, self.SR, notes, vels, 2.0, release,
                             extract_release=True, min_note=60, max_note=60,
                             collapse_to_mono=False)
        expected_frames = int(round(release * self.SR))
        assert samples[0].audio.shape[0] == expected_frames

    def test_correct_note_and_velocity_assigned(self):
        notes = [60, 72]
        vels = _compute_velocities(2)
        audio = self._make_audio(notes, vels, hold=1.0, release=0.5)
        samples = _slice_wav(audio, self.SR, notes, vels, 1.0, 0.5,
                             extract_release=False, min_note=60, max_note=72,
                             collapse_to_mono=False)
        note_vel_pairs = [(s.midi_note, s.velocity) for s in samples]
        expected = [(n, v) for n in notes for v in vels]
        assert note_vel_pairs == expected

    def test_collapse_to_mono(self):
        notes = [60]
        vels = [127]
        hold = 0.5
        n_frames = int(round((hold + 0.25) * self.SR))
        audio_stereo = np.ones((n_frames, 2), dtype=np.float64) * 0.5
        samples = _slice_wav(audio_stereo, self.SR, notes, vels, hold, 0.25,
                             extract_release=False, min_note=60, max_note=60,
                             collapse_to_mono=True)
        assert samples[0].audio.ndim == 1

    def test_stereo_preserved_when_not_collapsing(self):
        notes = [60]
        vels = [127]
        hold = 0.5
        n_frames = int(round((hold + 0.25) * self.SR))
        audio_stereo = np.ones((n_frames, 2), dtype=np.float64) * 0.5
        samples = _slice_wav(audio_stereo, self.SR, notes, vels, hold, 0.25,
                             extract_release=False, min_note=60, max_note=60,
                             collapse_to_mono=False)
        assert samples[0].audio.ndim == 2
        assert samples[0].audio.shape[1] == 2

    def test_sr_stored_on_sample(self):
        notes = [60]
        vels = [127]
        audio = self._make_audio(notes, vels, hold=0.5, release=0.25)
        samples = _slice_wav(audio, self.SR, notes, vels, 0.5, 0.25,
                             extract_release=False, min_note=60, max_note=60,
                             collapse_to_mono=False)
        assert samples[0].sr == self.SR


# ---------------------------------------------------------------------------
# load_instrument (integration)
# ---------------------------------------------------------------------------

class TestLoadInstrument:
    SR = 22050

    def _setup_instrument(
        self, tmp_path: Path,
        x: int = 2, n: int = 12,
        h: float = 1.0, r: float = 0.5,
        start: int = 60, end: int = 72,
        with_release: bool = False,
        extra_meta: str = "",
        n_channels: int = 1,
    ):
        notes = _get_recorded_notes(start, end, n)
        velocities = _compute_velocities(x)
        write_meta(tmp_path, x=x, n=n, h=h, r=r, start=start, end=end)
        (tmp_path / "metadata.txt").open("a").write(extra_meta)
        make_long_wav(tmp_path, "sustain.wav", notes, velocities, h, r,
                      sr=self.SR, n_channels=n_channels)
        if with_release:
            make_long_wav(tmp_path, "release.wav", notes, velocities, h, r,
                          sr=self.SR, n_channels=n_channels, extract_release=True)
        return notes, velocities

    def test_returns_instrument_data(self, tmp_path):
        from src.data import InstrumentData
        self._setup_instrument(tmp_path)
        data = load_instrument(tmp_path)
        assert isinstance(data, InstrumentData)

    def test_meta_parsed(self, tmp_path):
        self._setup_instrument(tmp_path, x=2, n=12, h=1.0, r=0.5)
        data = load_instrument(tmp_path)
        assert data.meta.velocity_layers == 2
        assert data.meta.semitone_interval == 12
        assert data.meta.hold_time == 1.0

    def test_sustain_sample_count(self, tmp_path):
        notes, velocities = self._setup_instrument(tmp_path, x=2, n=12,
                                                   start=60, end=72)
        data = load_instrument(tmp_path)
        assert len(data.sustain) == len(notes) * len(velocities)

    def test_no_release_wav_gives_empty_list(self, tmp_path):
        self._setup_instrument(tmp_path, with_release=False)
        data = load_instrument(tmp_path)
        assert data.release == []

    def test_release_wav_loaded(self, tmp_path):
        notes, velocities = self._setup_instrument(tmp_path, x=2, n=12,
                                                   with_release=True)
        data = load_instrument(tmp_path)
        assert len(data.release) == len(notes) * len(velocities)

    def test_sample_duration_matches_hold_time(self, tmp_path):
        h = 1.5
        self._setup_instrument(tmp_path, h=h, r=0.5)
        data = load_instrument(tmp_path)
        expected = int(round(h * self.SR))
        for s in data.sustain:
            assert s.audio.shape[0] == expected

    def test_release_sample_duration_matches_release_time(self, tmp_path):
        r = 1.2
        self._setup_instrument(tmp_path, h=1.0, r=r, with_release=True)
        data = load_instrument(tmp_path)
        expected = int(round(r * self.SR))
        for s in data.release:
            assert s.audio.shape[0] == expected

    def test_note_range_correct(self, tmp_path):
        self._setup_instrument(tmp_path, n=12, start=60, end=72)
        data = load_instrument(tmp_path)
        notes = data.sustain_notes()
        assert min(notes) == 60
        assert max(notes) == 72

    def test_velocity_values_correct(self, tmp_path):
        self._setup_instrument(tmp_path, x=2)
        data = load_instrument(tmp_path)
        assert data.sustain_velocities() == [1, 127]

    def test_min_max_note_filtering(self, tmp_path):
        notes, velocities = self._setup_instrument(
            tmp_path, n=1, start=60, end=65,
            extra_meta="\nmin_note = 62\nmax_note = 64\n"
        )
        data = load_instrument(tmp_path)
        returned_notes = data.sustain_notes()
        assert all(62 <= n <= 64 for n in returned_notes)
        assert 60 not in returned_notes
        assert 65 not in returned_notes

    def test_collapse_to_mono(self, tmp_path):
        self._setup_instrument(tmp_path, n_channels=2,
                               extra_meta="\ncollapse_to_mono = true\n")
        data = load_instrument(tmp_path)
        for s in data.sustain:
            assert s.audio.ndim == 1, "Expected mono after collapse"

    def test_stereo_preserved_without_collapse(self, tmp_path):
        self._setup_instrument(tmp_path, n_channels=2)
        data = load_instrument(tmp_path)
        for s in data.sustain:
            assert s.audio.ndim == 2
            assert s.audio.shape[1] == 2

    def test_missing_sustain_wav_raises(self, tmp_path):
        write_meta(tmp_path)
        with pytest.raises(FileNotFoundError, match="sustain.wav"):
            load_instrument(tmp_path)

    def test_missing_metadata_raises(self, tmp_path):
        notes = _get_recorded_notes(60, 72, 12)
        vels = _compute_velocities(2)
        make_long_wav(tmp_path, "sustain.wav", notes, vels, 1.0, 0.5, sr=self.SR)
        with pytest.raises(FileNotFoundError):
            load_instrument(tmp_path)

    def test_invalid_metadata_raises(self, tmp_path):
        notes = _get_recorded_notes(60, 72, 12)
        vels = _compute_velocities(2)
        make_long_wav(tmp_path, "sustain.wav", notes, vels, 1.0, 0.5, sr=self.SR)
        p = tmp_path / "metadata.txt"
        p.write_text("velocity_layers = 2\n")  # missing required fields
        with pytest.raises(MetadataError):
            load_instrument(tmp_path)

    def test_release_uses_override_timing(self, tmp_path):
        """release.wav sliced with release_hold_time / release_release_time when set."""
        x, n = 2, 12
        h_sustain, r_sustain = 1.0, 0.5
        h_release, r_release = 2.0, 1.5
        start, end = 60, 72
        notes = _get_recorded_notes(start, end, n)
        velocities = _compute_velocities(x)

        write_meta(tmp_path, x=x, n=n, h=h_sustain, r=r_sustain, start=start, end=end)
        meta_path = tmp_path / "metadata.txt"
        with meta_path.open("a") as f:
            f.write(f"\nrelease_hold_time = {h_release}\nrelease_release_time = {r_release}\n")

        make_long_wav(tmp_path, "sustain.wav", notes, velocities, h_sustain, r_sustain,
                      sr=self.SR)
        make_long_wav(tmp_path, "release.wav", notes, velocities, h_release, r_release,
                      sr=self.SR, extract_release=True)

        data = load_instrument(tmp_path)
        expected_release_frames = int(round(r_release * self.SR))
        for s in data.release:
            assert s.audio.shape[0] == expected_release_frames

    def test_four_velocity_layers(self, tmp_path):
        self._setup_instrument(tmp_path, x=4, n=12)
        data = load_instrument(tmp_path)
        assert data.sustain_velocities() == _compute_velocities(4)

    def test_single_velocity_layer(self, tmp_path):
        self._setup_instrument(tmp_path, x=1, n=12)
        data = load_instrument(tmp_path)
        assert data.sustain_velocities() == [127]

    def test_single_note(self, tmp_path):
        self._setup_instrument(tmp_path, x=2, n=12, start=60, end=60)
        data = load_instrument(tmp_path)
        assert data.sustain_notes() == [60]
        assert len(data.sustain) == 2  # 1 note × 2 velocities
