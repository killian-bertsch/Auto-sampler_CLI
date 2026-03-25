"""
Phase 5 tests — OutputModule: note selection, velocity zone computation,
SFZ generation, FLAC encoding, and full write_instrument integration.

Key changes from original:
  - velocity_layers_out / velocity_map removed; per-note velocity zones used
  - velocity mode amp_velcurve_1 is always 1.0 (dynamics baked into audio)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data import InstrumentData, InstrumentMeta, Sample
from src.output_module import (
    compute_note_ranges,
    compute_velocity_ranges,
    encode_flac,
    estimate_rt_decay,
    generate_release_sfz,
    generate_sustain_sfz,
    midi_to_note_name,
    select_notes,
    write_instrument,
)
from src.metadata_parser import MetadataError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(**overrides) -> InstrumentMeta:
    defaults = dict(
        min_note=21,
        max_note=108,
        note_percentage=100.0,
        crossfade_percent=0.0,
        normalize=False,
        normalize_mode="lufs",
        normalize_target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        velocity_dynamic_range_db=40.0,
        instrument_name="TestInstrument",
        ampeg_release=0.5,
        loop_crossfade_ms=20.0,
    )
    defaults.update(overrides)
    return InstrumentMeta(**defaults)


def _sine_sample(midi_note: int, velocity: int, dur_s: float = 0.5, sr: int = 44100) -> Sample:
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    freq = 440.0 * 2 ** ((midi_note - 69) / 12.0)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float64)
    return Sample(midi_note=midi_note, velocity=velocity, audio=audio, sr=sr)


def _make_data(notes, velocities, with_release=False, **meta_overrides) -> InstrumentData:
    """Build InstrumentData; velocities may be a flat list (same for all notes) or
    a dict {note: [vel, ...]} for non-uniform per-note velocities."""
    meta = _make_meta(**meta_overrides)
    if isinstance(velocities, dict):
        sustain = [_sine_sample(n, v) for n, vels in velocities.items() for v in vels]
        release = [_sine_sample(n, v, 0.3) for n, vels in velocities.items() for v in vels] if with_release else []
    else:
        sustain = [_sine_sample(n, v) for n in notes for v in velocities]
        release = [_sine_sample(n, v, 0.3) for n in notes for v in velocities] if with_release else []
    return InstrumentData(sustain=sustain, release=release, meta=meta)


# ---------------------------------------------------------------------------
# midi_to_note_name
# ---------------------------------------------------------------------------

class TestMidiToNoteName:
    def test_middle_c(self):
        assert midi_to_note_name(60) == "C4"

    def test_a4(self):
        assert midi_to_note_name(69) == "A4"

    def test_low_a(self):
        assert midi_to_note_name(21) == "A0"

    def test_sharp(self):
        assert midi_to_note_name(61) == "C#4"

    def test_high_c(self):
        assert midi_to_note_name(108) == "C8"


# ---------------------------------------------------------------------------
# select_notes
# ---------------------------------------------------------------------------

class TestSelectNotes:
    def test_100_percent_returns_all(self):
        notes = [21, 33, 45, 57, 69, 81, 93, 105]
        assert select_notes(notes, 100.0) == notes

    def test_50_percent_halves(self):
        notes = list(range(0, 88))
        result = select_notes(notes, 50.0)
        assert len(result) == 44

    def test_includes_first_and_last_when_multiple(self):
        notes = [21, 33, 45, 57, 69]
        result = select_notes(notes, 60.0)
        assert result[0] == 21
        assert result[-1] == 69

    def test_single_note_target_returns_middle(self):
        notes = [21, 45, 69, 93]
        result = select_notes(notes, 25.0)
        assert len(result) == 1

    def test_empty_returns_empty(self):
        assert select_notes([], 100.0) == []

    def test_single_element(self):
        assert select_notes([60], 100.0) == [60]
        assert select_notes([60], 10.0) == [60]

    def test_result_is_sorted_subset(self):
        notes = [21, 33, 45, 57, 69, 81]
        result = select_notes(notes, 50.0)
        assert result == sorted(result)
        assert all(n in notes for n in result)

    def test_percentage_rounds_to_minimum_1(self):
        notes = [21, 45, 69]
        result = select_notes(notes, 1.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# compute_note_ranges
# ---------------------------------------------------------------------------

class TestComputeNoteRanges:
    def test_single_note_spans_full_range(self):
        result = compute_note_ranges([60], 21, 108)
        assert len(result) == 1
        assert result[0]["lokey"] == 21
        assert result[0]["hikey"] == 108
        assert result[0]["midi_note"] == 60

    def test_two_notes_midpoint_boundary(self):
        result = compute_note_ranges([40, 60], 21, 108)
        assert result[0]["lokey"] == 21
        assert result[0]["hikey"] == 50
        assert result[1]["lokey"] == 51
        assert result[1]["hikey"] == 108

    def test_three_notes_boundaries(self):
        result = compute_note_ranges([36, 60, 84], 21, 108)
        assert result[0]["lokey"] == 21
        assert result[0]["hikey"] == 48
        assert result[1]["lokey"] == 49
        assert result[1]["hikey"] == 72
        assert result[2]["lokey"] == 73
        assert result[2]["hikey"] == 108

    def test_contiguous_coverage(self):
        notes = [36, 48, 60, 72, 84]
        result = compute_note_ranges(notes, 21, 108)
        for i in range(1, len(result)):
            assert result[i]["lokey"] == result[i - 1]["hikey"] + 1


# ---------------------------------------------------------------------------
# compute_velocity_ranges
# ---------------------------------------------------------------------------

class TestComputeVelocityRanges:
    def test_no_crossfade_natural_boundaries(self):
        result = compute_velocity_ranges([64, 127], 0.0)
        assert result[0]["lovel"] == 1
        assert result[0]["hivel"] == 95
        assert result[1]["lovel"] == 96
        assert result[1]["hivel"] == 127

    def test_no_crossfade_no_xfin_xfout(self):
        result = compute_velocity_ranges([64, 127], 0.0)
        assert result[0]["has_xfout"] is False
        assert result[1]["has_xfin"] is False

    def test_single_velocity_full_range(self):
        result = compute_velocity_ranges([100], 0.0)
        assert len(result) == 1
        assert result[0]["lovel"] == 1
        assert result[0]["hivel"] == 127

    def test_crossfade_extends_boundaries(self):
        result = compute_velocity_ranges([30, 60, 90, 120], 20.0)
        layer_60 = result[1]
        assert layer_60["lovel"] < 46
        assert layer_60["hivel"] > 75
        assert layer_60["has_xfin"] is True
        assert layer_60["has_xfout"] is True

    def test_crossfade_lowest_layer_no_xfin(self):
        result = compute_velocity_ranges([30, 60, 90, 120], 20.0)
        assert result[0]["has_xfin"] is False
        assert result[0]["lovel"] == 1

    def test_crossfade_highest_layer_no_xfout(self):
        result = compute_velocity_ranges([30, 60, 90, 120], 20.0)
        assert result[-1]["has_xfout"] is False
        assert result[-1]["hivel"] == 127

    def test_four_velocities_no_crossfade_contiguous(self):
        vels = [30, 60, 90, 120]
        result = compute_velocity_ranges(vels, 0.0)
        for i in range(1, len(result)):
            assert result[i]["lovel"] == result[i - 1]["hivel"] + 1

    def test_xfin_range_correct(self):
        result = compute_velocity_ranges([30, 90], 50.0)
        layer_90 = result[1]
        assert layer_90["xfin_lovel"] == layer_90["lovel"]
        assert layer_90["xfin_hivel"] == 61  # lo_nat for second layer

    def test_clamped_to_1_127(self):
        result = compute_velocity_ranges([10, 120], 100.0)
        assert result[0]["lovel"] == 1
        assert result[-1]["hivel"] == 127

    def test_non_uniform_velocities(self):
        """Zones built from arbitrary velocity values."""
        result = compute_velocity_ranges([1, 40, 80, 127], 0.0)
        assert len(result) == 4
        assert result[0]["lovel"] == 1
        assert result[-1]["hivel"] == 127


# ---------------------------------------------------------------------------
# estimate_rt_decay
# ---------------------------------------------------------------------------

class TestEstimateRtDecay:
    def test_decaying_signal_returns_positive(self):
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2)
        audio = np.exp(-t * 3.45) * np.sin(2 * np.pi * 440 * t)
        result = estimate_rt_decay(audio, sr)
        assert 1.0 <= result <= 24.0

    def test_very_short_signal_returns_default(self):
        audio = np.sin(np.linspace(0, 0.01, 441))
        result = estimate_rt_decay(audio, 44100)
        assert result == 6.0

    def test_stereo_audio_handled(self):
        sr = 44100
        t = np.linspace(0, 1.0, sr)
        audio = np.column_stack([
            np.exp(-t * 2) * np.sin(2 * np.pi * 440 * t),
            np.exp(-t * 2) * np.sin(2 * np.pi * 440 * t),
        ])
        result = estimate_rt_decay(audio, sr)
        assert 1.0 <= result <= 24.0

    def test_clamped_to_range(self):
        sr = 44100
        audio = np.ones(sr * 2) * 0.5
        result = estimate_rt_decay(audio, sr)
        assert result >= 1.0


# ---------------------------------------------------------------------------
# encode_flac
# ---------------------------------------------------------------------------

class TestEncodeFlac:
    def test_roundtrip_mono(self, tmp_path):
        sr = 44100
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, sr)).astype(np.float64) * 0.5
        path = tmp_path / "test.flac"
        encode_flac(audio, sr, path)
        loaded, loaded_sr = sf.read(str(path))
        assert loaded_sr == sr
        assert loaded.shape == audio.shape
        assert np.allclose(audio, loaded, atol=1e-3)

    def test_roundtrip_stereo(self, tmp_path):
        sr = 44100
        audio = np.column_stack([
            np.sin(np.linspace(0, 2 * np.pi * 440, sr)) * 0.5,
            np.sin(np.linspace(0, 2 * np.pi * 880, sr)) * 0.5,
        ]).astype(np.float64)
        path = tmp_path / "stereo.flac"
        encode_flac(audio, sr, path)
        loaded, loaded_sr = sf.read(str(path))
        assert loaded_sr == sr
        assert loaded.shape == audio.shape

    def test_clips_above_1(self, tmp_path):
        sr = 22050
        audio = np.array([2.0, -2.0, 1.5, -1.5])
        path = tmp_path / "clipped.flac"
        encode_flac(audio, sr, path)
        loaded, _ = sf.read(str(path))
        assert np.all(np.abs(loaded) <= 1.0 + 1e-4)

    def test_creates_file(self, tmp_path):
        path = tmp_path / "out.flac"
        encode_flac(np.zeros(1000), 44100, path)
        assert path.exists()
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# generate_sustain_sfz
# ---------------------------------------------------------------------------

class TestGenerateSustainSfz:
    def _make_region(self, **overrides):
        base = dict(
            filename="Piano_C4_v64.flac",
            lokey=57, hikey=62, pitch_keycenter=60,
            lovel=1, hivel=127,
            has_xfin=False, has_xfout=False,
            xfin_lovel=1, xfin_hivel=1,
            xfout_lovel=127, xfout_hivel=127,
            loop_start=None, loop_end=None,
        )
        base.update(overrides)
        return base

    def test_contains_group_header(self):
        meta = _make_meta()
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "<group>" in sfz
        assert "ampeg_release=0.500" in sfz

    def test_amp_velcurve_1_from_dynamic_range_lufs_mode(self):
        meta = _make_meta(normalize_mode="lufs", velocity_dynamic_range_db=40.0)
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "amp_velcurve_1=0.010000" in sfz

    def test_velocity_mode_amp_velcurve_1_is_1(self):
        """In velocity mode, dynamics are baked into audio — amp_velcurve_1 = 1.0."""
        meta = _make_meta(normalize_mode="velocity", velocity_dynamic_range_db=40.0)
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "amp_velcurve_1=1.000000" in sfz

    def test_region_fields_present(self):
        meta = _make_meta()
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "sample=Piano_C4_v64.flac" in sfz
        assert "lokey=57" in sfz
        assert "hikey=62" in sfz
        assert "pitch_keycenter=60" in sfz
        assert "lovel=1" in sfz
        assert "hivel=127" in sfz

    def test_no_xfin_xfout_when_no_crossfade(self):
        meta = _make_meta()
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "xfin_lovel" not in sfz
        assert "xfout_lovel" not in sfz

    def test_xfin_xfout_when_has_crossfade(self):
        meta = _make_meta()
        region = self._make_region(
            has_xfin=True, has_xfout=True,
            xfin_lovel=40, xfin_hivel=46,
            xfout_lovel=75, xfout_hivel=81,
        )
        sfz = generate_sustain_sfz([region], meta)
        assert "xfin_lovel=40" in sfz
        assert "xfin_hivel=46" in sfz
        assert "xfout_lovel=75" in sfz
        assert "xfout_hivel=81" in sfz

    def test_loop_opcodes_when_loop_set(self):
        meta = _make_meta(loop_crossfade_ms=20.0)
        region = self._make_region(loop_start=44100, loop_end=88200)
        sfz = generate_sustain_sfz([region], meta)
        assert "loop_mode=loop_continuous" in sfz
        assert "loop_start=44100" in sfz
        assert "loop_end=88200" in sfz
        assert "loop_crossfade=0.0200" in sfz

    def test_no_loop_opcodes_when_no_loop(self):
        meta = _make_meta()
        sfz = generate_sustain_sfz([self._make_region(loop_start=None, loop_end=None)], meta)
        assert "loop_mode" not in sfz
        assert "loop_start" not in sfz

    def test_default_path_samples(self):
        meta = _make_meta()
        sfz = generate_sustain_sfz([self._make_region()], meta)
        assert "default_path=samples/" in sfz

    def test_multiple_regions(self):
        meta = _make_meta()
        regions = [
            self._make_region(filename="Piano_C4_v30.flac", lovel=1, hivel=45),
            self._make_region(filename="Piano_C4_v90.flac", lovel=46, hivel=127),
        ]
        sfz = generate_sustain_sfz(regions, meta)
        assert "Piano_C4_v30.flac" in sfz
        assert "Piano_C4_v90.flac" in sfz


# ---------------------------------------------------------------------------
# generate_release_sfz
# ---------------------------------------------------------------------------

class TestGenerateReleaseSfz:
    def _make_region(self, **overrides):
        base = dict(
            filename="Piano_C4_v64_rel.flac",
            lokey=57, hikey=62, pitch_keycenter=60,
            lovel=1, hivel=127,
            rt_decay=6.0,
        )
        base.update(overrides)
        return base

    def test_trigger_release_in_group(self):
        meta = _make_meta()
        sfz = generate_release_sfz([self._make_region()], meta)
        assert "trigger=release" in sfz

    def test_ampeg_attack_equals_ampeg_release(self):
        meta = _make_meta(ampeg_release=0.8)
        sfz = generate_release_sfz([self._make_region()], meta)
        assert "ampeg_attack=0.800" in sfz

    def test_ampeg_sustain_zero(self):
        meta = _make_meta()
        sfz = generate_release_sfz([self._make_region()], meta)
        assert "ampeg_sustain=0" in sfz

    def test_rt_decay_present(self):
        meta = _make_meta()
        sfz = generate_release_sfz([self._make_region(rt_decay=12.5)], meta)
        assert "rt_decay=12.50" in sfz

    def test_default_path_samples_release(self):
        meta = _make_meta()
        sfz = generate_release_sfz([self._make_region()], meta)
        assert "default_path=samples/release/" in sfz

    def test_region_fields(self):
        meta = _make_meta()
        sfz = generate_release_sfz([self._make_region()], meta)
        assert "sample=Piano_C4_v64_rel.flac" in sfz
        assert "lokey=57" in sfz
        assert "hikey=62" in sfz


# ---------------------------------------------------------------------------
# write_instrument integration
# ---------------------------------------------------------------------------

class TestWriteInstrument:
    def test_creates_output_directory(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        assert Path(result["output_dir"]).exists()

    def test_sustain_sfz_created(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        assert (Path(result["output_dir"]) / "TestInstrument_sustain.sfz").exists()

    def test_no_release_sfz_without_release_samples(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        write_instrument(data, tmp_path)
        assert not (tmp_path / "TestInstrument" / "TestInstrument_release.sfz").exists()

    def test_release_sfz_created_with_release_samples(self, tmp_path):
        data = _make_data([60, 72], [64, 100], with_release=True)
        write_instrument(data, tmp_path)
        assert (tmp_path / "TestInstrument" / "TestInstrument_release.sfz").exists()

    def test_flac_files_created(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        flac_files = list((Path(result["output_dir"]) / "samples").glob("*.flac"))
        assert len(flac_files) == 4
        assert result["files_written"] == 4

    def test_release_flac_files_created(self, tmp_path):
        data = _make_data([60], [64, 100], with_release=True)
        result = write_instrument(data, tmp_path)
        flac_files = list((Path(result["output_dir"]) / "samples" / "release").glob("*.flac"))
        assert len(flac_files) == 2

    def test_flac_files_readable(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        for f in (tmp_path / "TestInstrument" / "samples").glob("*.flac"):
            audio, sr = sf.read(str(f))
            assert sr == 44100
            assert len(audio) > 0

    def test_note_percentage_filters_notes(self, tmp_path):
        data = _make_data([36, 48, 60, 72], [64], note_percentage=50.0)
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 2

    def test_min_max_note_range_respected(self, tmp_path):
        data = _make_data([36, 48, 60, 72], [64], min_note=48, max_note=60)
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 2

    def test_all_velocities_written_per_note(self, tmp_path):
        """All recorded velocity layers are included — no subsampling."""
        data = _make_data([60], [30, 60, 90, 120])
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 4

    def test_non_uniform_velocities_per_note(self, tmp_path):
        """Each note gets its own velocity zones from its own recorded layers."""
        data = _make_data(
            notes=[],
            velocities={60: [1, 40, 80, 127], 72: [1, 127]},
        )
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 6  # 4 + 2

    def test_non_uniform_sfz_velocity_zones(self, tmp_path):
        """Verify SFZ contains per-note velocity zones reflecting non-uniform layers."""
        data = _make_data(
            notes=[],
            velocities={60: [1, 64, 127], 72: [1, 127]},
        )
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        # Both notes should appear; zones should differ
        assert sfz_text.count("<region>") == 5  # 3 + 2 regions

    def test_sfz_references_correct_filenames(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "TestInstrument_C4_v64.flac" in sfz_text

    def test_sfz_key_zones_cover_full_range(self, tmp_path):
        data = _make_data([60], [64], min_note=21, max_note=108)
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "lokey=21" in sfz_text
        assert "hikey=108" in sfz_text

    def test_loop_points_in_sfz_when_set(self, tmp_path):
        data = _make_data([60], [64])
        data.sustain[0].loop_start = 10000
        data.sustain[0].loop_end = 20000
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "loop_mode=loop_continuous" in sfz_text
        assert "loop_start=10000" in sfz_text
        assert "loop_end=20000" in sfz_text

    def test_no_loop_opcodes_when_loop_not_set(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "loop_mode" not in sfz_text

    def test_crossfade_opcodes_in_sfz(self, tmp_path):
        data = _make_data([60], [30, 90], crossfade_percent=20.0)
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "xfin_lovel" in sfz_text

    def test_returns_correct_summary(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        assert result["instrument_name"] == "TestInstrument"
        assert result["sustain_regions"] == 4
        assert result["release_regions"] == 0
        assert result["files_written"] == 4

    def test_release_sfz_trigger_release(self, tmp_path):
        data = _make_data([60], [64], with_release=True)
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_release.sfz").read_text()
        assert "trigger=release" in sfz_text

    def test_release_sfz_ampeg_attack_matches_sustain_ampeg_release(self, tmp_path):
        data = _make_data([60], [64], with_release=True, ampeg_release=0.8)
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_release.sfz").read_text()
        assert "ampeg_attack=0.800" in sfz_text

    def test_samples_dir_created(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        assert (tmp_path / "TestInstrument" / "samples").is_dir()

    def test_multiple_instruments_do_not_collide(self, tmp_path):
        data_a = _make_data([60], [64], instrument_name="PianoA")
        data_b = _make_data([60], [64], instrument_name="PianoB")
        write_instrument(data_a, tmp_path)
        write_instrument(data_b, tmp_path)
        assert (tmp_path / "PianoA").is_dir()
        assert (tmp_path / "PianoB").is_dir()

    def test_velocity_mode_amp_velcurve_1_is_flat(self, tmp_path):
        """In velocity mode, SFZ amp_velcurve_1 must be 1.0."""
        data = _make_data([60], [64, 127], normalize_mode="velocity")
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "amp_velcurve_1=1.000000" in sfz_text
