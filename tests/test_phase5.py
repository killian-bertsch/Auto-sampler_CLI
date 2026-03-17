"""
Phase 5 tests — OutputModule: subset selection, zone computation, SFZ generation,
FLAC encoding, and full write_instrument integration.
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
    select_velocities,
    select_velocities_segmented,
    write_instrument,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_meta(**overrides) -> InstrumentMeta:
    defaults = dict(
        velocity_layers=4,
        semitone_interval=12,
        hold_time=2.0,
        release_time=1.0,
        start_note=21,
        end_note=108,
        min_note=21,
        max_note=108,
        note_percentage=100.0,
        velocity_layers_out=4,
        crossfade_percent=0.0,
        normalize=False,
        normalize_mode="lufs",
        normalize_target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        velocity_dynamic_range_db=40.0,
        instrument_name="TestInstrument",
        ampeg_release=0.5,
        loop_crossfade_ms=20.0,
        collapse_to_mono=False,
    )
    defaults.update(overrides)
    return InstrumentMeta(**defaults)


def _sine_sample(midi_note: int, velocity: int, dur_s: float = 0.5, sr: int = 44100) -> Sample:
    """Create a synthetic mono sine Sample."""
    t = np.linspace(0, dur_s, int(sr * dur_s), endpoint=False)
    freq = 440.0 * 2 ** ((midi_note - 69) / 12.0)
    audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float64)
    return Sample(midi_note=midi_note, velocity=velocity, audio=audio, sr=sr)


def _make_data(notes, velocities, with_release=False, **meta_overrides) -> InstrumentData:
    """Create an InstrumentData with sustain (and optionally release) samples."""
    meta = _make_meta(
        velocity_layers=meta_overrides.pop("velocity_layers", len(velocities)),
        velocity_layers_out=meta_overrides.pop("velocity_layers_out", len(velocities)),
        **meta_overrides,
    )
    sustain = [_sine_sample(n, v) for n in notes for v in velocities]
    release = [_sine_sample(n, v, dur_s=0.3) for n in notes for v in velocities] if with_release else []
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
        result = select_notes(notes, 25.0)  # 25% of 4 = 1
        assert len(result) == 1
        assert result[0] == 69  # n//2 = 4//2 = 2, notes[2] = 69

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
        result = select_notes(notes, 1.0)  # 1% of 3 rounds to 0 -> clamped to 1
        assert len(result) == 1


# ---------------------------------------------------------------------------
# select_velocities
# ---------------------------------------------------------------------------

class TestSelectVelocities:
    def test_all_layers_returned_when_equal(self):
        vels = [30, 60, 90, 120]
        assert select_velocities(vels, 4) == vels

    def test_reduce_to_2_evenly_spaced(self):
        vels = [30, 60, 90, 120]
        result = select_velocities(vels, 2)
        assert len(result) == 2
        assert result[0] == 30
        assert result[-1] == 120

    def test_reduce_to_1_returns_highest(self):
        vels = [30, 60, 90, 120]
        result = select_velocities(vels, 1)
        assert result == [120]

    def test_empty_returns_empty(self):
        assert select_velocities([], 4) == []

    def test_clamps_to_available(self):
        vels = [64, 127]
        assert select_velocities(vels, 10) == [64, 127]

    def test_result_is_subset(self):
        vels = [20, 40, 60, 80, 100, 120]
        result = select_velocities(vels, 3)
        assert all(v in vels for v in result)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# select_velocities_segmented
# ---------------------------------------------------------------------------

class TestSelectVelocitiesSegmented:
    # Recorded velocities simulating 8 evenly-spread layers across 1–127
    VELS = [16, 32, 48, 64, 80, 96, 112, 127]

    def test_user_example_bottom_1_top_5(self):
        # "0-63:1, 64-127:5" — 1 layer from bottom half, 5 from top half
        result = select_velocities_segmented(self.VELS, "0-63:1, 64-127:5")
        bottom = [v for v in result if v <= 63]
        top = [v for v in result if v >= 64]
        assert len(bottom) == 1
        assert len(top) == 5
        assert result == sorted(result)  # sorted ascending

    def test_full_range_equivalent_to_n(self):
        # "0-127:3" should behave like select_velocities(vels, 3)
        result = select_velocities_segmented(self.VELS, "0-127:3")
        assert result == select_velocities(self.VELS, 3)

    def test_single_layer_per_half(self):
        result = select_velocities_segmented(self.VELS, "0-63:1, 64-127:1")
        assert len(result) == 2
        assert all(v in self.VELS for v in result)

    def test_all_layers_single_segment(self):
        result = select_velocities_segmented(self.VELS, "0-127:8")
        assert result == self.VELS

    def test_n_exceeds_available_uses_all(self):
        # Only 2 velocities in range 0-63; asking for 5 should give all 2
        result = select_velocities_segmented(self.VELS, "0-63:5")
        assert result == [v for v in self.VELS if v <= 63]

    def test_empty_segment_range_skipped(self):
        # No recorded velocities in 100-110 range
        result = select_velocities_segmented(self.VELS, "100-110:3, 111-127:1")
        assert all(v in self.VELS for v in result)
        assert all(v >= 100 for v in result)

    def test_overlapping_segments_deduped(self):
        # Both segments include velocity 64; result should not have duplicates
        result = select_velocities_segmented(self.VELS, "0-64:8, 64-127:8")
        assert len(result) == len(set(result))

    def test_result_sorted(self):
        result = select_velocities_segmented(self.VELS, "64-127:2, 0-63:2")
        assert result == sorted(result)

    def test_single_velocity_list(self):
        result = select_velocities_segmented([64], "0-127:1")
        assert result == [64]

    def test_empty_velocity_list(self):
        result = select_velocities_segmented([], "0-127:3")
        assert result == []


# ---------------------------------------------------------------------------
# parse_velocity_map errors
# ---------------------------------------------------------------------------

from src.metadata_parser import parse_velocity_map, MetadataError

class TestParseVelocityMap:
    def test_valid_two_segments(self):
        result = parse_velocity_map("0-63:1, 64-127:5")
        assert result == [(0, 63, 1), (64, 127, 5)]

    def test_single_segment(self):
        assert parse_velocity_map("0-127:4") == [(0, 127, 4)]

    def test_sorted_by_lo(self):
        result = parse_velocity_map("64-127:2, 0-63:1")
        assert result[0][0] == 0
        assert result[1][0] == 64

    def test_bad_format_raises(self):
        with pytest.raises(MetadataError):
            parse_velocity_map("garbage")

    def test_lo_greater_than_hi_raises(self):
        with pytest.raises(MetadataError):
            parse_velocity_map("90-10:2")

    def test_n_zero_raises(self):
        with pytest.raises(MetadataError):
            parse_velocity_map("0-127:0")

    def test_out_of_range_raises(self):
        with pytest.raises(MetadataError):
            parse_velocity_map("0-200:2")

    def test_empty_string_raises(self):
        with pytest.raises(MetadataError):
            parse_velocity_map("")


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
        assert result[0]["hikey"] == 50  # midpoint of 40 and 60 = 50
        assert result[1]["lokey"] == 51
        assert result[1]["hikey"] == 108

    def test_three_notes_boundaries(self):
        result = compute_note_ranges([36, 60, 84], 21, 108)
        assert result[0]["lokey"] == 21
        assert result[0]["hikey"] == 48     # (36+60)//2 = 48
        assert result[1]["lokey"] == 49
        assert result[1]["hikey"] == 72     # (60+84)//2 = 72
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
        # 2 velocities: [64, 127], midpoint = 95
        result = compute_velocity_ranges([64, 127], 0.0)
        assert result[0]["lovel"] == 1
        assert result[0]["hivel"] == 95  # (64+127)//2 = 95
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
        # Velocity 60 in [46, 75] with 20% crossfade: zone_width=30, xfade=6
        result = compute_velocity_ranges([30, 60, 90, 120], 20.0)
        layer_60 = result[1]
        assert layer_60["lovel"] < 46     # extended below natural boundary
        assert layer_60["hivel"] > 75     # extended above natural boundary
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
        mid = (30 + 90) // 2  # 60
        # Layer 90: lo_nat = 61, zone_width = 67, xfade = 33
        layer_90 = result[1]
        assert layer_90["xfin_lovel"] == layer_90["lovel"]
        assert layer_90["xfin_hivel"] == 61  # lo_nat for second layer

    def test_clamped_to_1_127(self):
        result = compute_velocity_ranges([10, 120], 100.0)
        assert result[0]["lovel"] == 1
        assert result[-1]["hivel"] == 127


# ---------------------------------------------------------------------------
# estimate_rt_decay
# ---------------------------------------------------------------------------

class TestEstimateRtDecay:
    def test_decaying_signal_returns_positive(self):
        sr = 44100
        t = np.linspace(0, 2.0, sr * 2)
        # Exponential decay: amplitude decays ~60 dB over 2 seconds = 30 dB/s
        audio = np.exp(-t * 3.45) * np.sin(2 * np.pi * 440 * t)
        result = estimate_rt_decay(audio, sr)
        assert 1.0 <= result <= 24.0

    def test_very_short_signal_returns_default(self):
        # Less than 2 windows
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
        # Very slow decay (nearly flat signal) → slope near 0 → clamped to 1.0
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
        assert np.allclose(audio, loaded, atol=1e-3)  # 24-bit quantization

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

    def test_amp_velcurve_1_from_dynamic_range(self):
        meta = _make_meta(velocity_dynamic_range_db=40.0)
        sfz = generate_sustain_sfz([self._make_region()], meta)
        # 10^(-40/20) = 0.01
        assert "amp_velcurve_1=0.010000" in sfz

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

    def test_velocity_mode_uses_computed_range(self):
        meta = _make_meta(
            normalize_mode="velocity",
            velocity_dynamic_range_db=40.0,
        )
        meta.computed_dynamic_range_db = 20.0  # should override
        sfz = generate_sustain_sfz([self._make_region()], meta)
        # 10^(-20/20) = 0.1
        assert "amp_velcurve_1=0.100000" in sfz


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
        out_dir = Path(result["output_dir"])
        assert out_dir.exists()

    def test_sustain_sfz_created(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        out_dir = Path(result["output_dir"])
        assert (out_dir / "TestInstrument_sustain.sfz").exists()

    def test_no_release_sfz_without_release_samples(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        write_instrument(data, tmp_path)
        out_dir = tmp_path / "TestInstrument"
        assert not (out_dir / "TestInstrument_release.sfz").exists()

    def test_release_sfz_created_with_release_samples(self, tmp_path):
        data = _make_data([60, 72], [64, 100], with_release=True)
        write_instrument(data, tmp_path)
        out_dir = tmp_path / "TestInstrument"
        assert (out_dir / "TestInstrument_release.sfz").exists()

    def test_flac_files_created(self, tmp_path):
        data = _make_data([60, 72], [64, 100])
        result = write_instrument(data, tmp_path)
        samples_dir = Path(result["output_dir"]) / "samples"
        flac_files = list(samples_dir.glob("*.flac"))
        # 2 notes × 2 velocities = 4 files
        assert len(flac_files) == 4
        assert result["files_written"] == 4

    def test_release_flac_files_created(self, tmp_path):
        data = _make_data([60], [64, 100], with_release=True)
        result = write_instrument(data, tmp_path)
        release_dir = Path(result["output_dir"]) / "samples" / "release"
        flac_files = list(release_dir.glob("*.flac"))
        assert len(flac_files) == 2

    def test_flac_files_readable(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        samples_dir = tmp_path / "TestInstrument" / "samples"
        for f in samples_dir.glob("*.flac"):
            audio, sr = sf.read(str(f))
            assert sr == 44100
            assert len(audio) > 0

    def test_note_percentage_filters_notes(self, tmp_path):
        # 4 notes at every octave, 50% should give 2
        data = _make_data(
            [36, 48, 60, 72], [64],
            note_percentage=50.0,
        )
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 2

    def test_velocity_layers_out_filters_velocities(self, tmp_path):
        data = _make_data(
            [60], [30, 60, 90, 120],
            velocity_layers_out=2,
        )
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 2

    def test_min_max_note_range_respected(self, tmp_path):
        # Samples at 36, 48, 60, 72 — only 48–60 should be included
        data = _make_data(
            [36, 48, 60, 72], [64],
            min_note=48, max_note=60,
        )
        result = write_instrument(data, tmp_path)
        assert result["sustain_regions"] == 2

    def test_sfz_references_correct_filenames(self, tmp_path):
        data = _make_data([60], [64])
        write_instrument(data, tmp_path)
        sfz_path = tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz"
        sfz_text = sfz_path.read_text()
        assert "TestInstrument_C4_v64.flac" in sfz_text

    def test_sfz_key_zones_cover_full_range(self, tmp_path):
        data = _make_data([60], [64], min_note=21, max_note=108)
        write_instrument(data, tmp_path)
        sfz_path = tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz"
        sfz_text = sfz_path.read_text()
        assert "lokey=21" in sfz_text
        assert "hikey=108" in sfz_text

    def test_loop_points_in_sfz_when_set(self, tmp_path):
        data = _make_data([60], [64])
        # Manually set loop points
        data.sustain[0].loop_start = 10000
        data.sustain[0].loop_end = 20000
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "loop_mode=loop_continuous" in sfz_text
        assert "loop_start=10000" in sfz_text
        assert "loop_end=20000" in sfz_text

    def test_no_loop_opcodes_when_loop_not_set(self, tmp_path):
        data = _make_data([60], [64])
        # loop_start/loop_end remain None
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        assert "loop_mode" not in sfz_text

    def test_crossfade_opcodes_in_sfz(self, tmp_path):
        data = _make_data([60], [30, 90], crossfade_percent=20.0)
        write_instrument(data, tmp_path)
        sfz_text = (tmp_path / "TestInstrument" / "TestInstrument_sustain.sfz").read_text()
        # Layer 90 should have xfin opcodes
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
