"""
test_phase4.py — Unit tests for Phase 4: LoopFinderProcessor.

All tests use synthetic signals (no disk I/O).

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase4.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import InstrumentData, InstrumentMeta, Sample
from src.processors.loop_finder import LoopFinderProcessor, find_loop_point


SR = 44100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(**kwargs) -> InstrumentMeta:
    defaults = dict(
        velocity_layers=1,
        semitone_interval=12,
        hold_time=1.0,
        release_time=0.5,
        start_note=60,
        end_note=60,
        min_note=60,
        max_note=60,
        note_percentage=100.0,
        velocity_layers_out=1,
        crossfade_percent=0.0,
        normalize=False,
        normalize_mode="rms",
        normalize_target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        velocity_dynamic_range_db=40.0,
        instrument_name="TestInst",
        ampeg_release=0.5,
        loop_crossfade_ms=20.0,
        onset_threshold_db=-40.0,
    )
    defaults.update(kwargs)
    return InstrumentMeta(**defaults)


def make_sine(freq: float, duration_s: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    """Generate a pure sine wave."""
    t = np.arange(int(sr * duration_s)) / sr
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float64)


def make_sample(audio: np.ndarray, midi_note: int = 60, velocity: int = 64) -> Sample:
    return Sample(midi_note=midi_note, velocity=velocity, audio=audio, sr=SR)


def make_instrument(sustain_samples: list[Sample], release_samples: list[Sample] | None = None, **meta_kwargs) -> InstrumentData:
    return InstrumentData(
        sustain=sustain_samples,
        release=release_samples or [],
        meta=make_meta(**meta_kwargs),
    )


# ---------------------------------------------------------------------------
# Tests for find_loop_point (core function)
# ---------------------------------------------------------------------------

class TestFindLoopPoint:

    def test_sine_returns_pair(self):
        """A sustained sine wave should yield a valid loop pair."""
        audio = make_sine(440, 2.0)
        result = find_loop_point(audio, SR)
        assert result is not None
        loop_start, loop_end = result
        assert isinstance(loop_start, int)
        assert isinstance(loop_end, int)

    def test_loop_start_before_loop_end(self):
        """loop_start must always be less than loop_end."""
        audio = make_sine(440, 2.0)
        result = find_loop_point(audio, SR)
        assert result is not None
        assert result[0] < result[1]

    def test_loop_points_in_search_region(self):
        """Both loop points must fall within the 85%–97% search window."""
        audio = make_sine(440, 2.0)
        n = len(audio)
        result = find_loop_point(audio, SR, region_start_frac=0.85, region_end_frac=0.97)
        assert result is not None
        loop_start, loop_end = result
        assert loop_start >= int(n * 0.85), f"loop_start {loop_start} before search region start {int(n * 0.85)}"
        assert loop_end <= int(n * 0.97), f"loop_end {loop_end} after search region end {int(n * 0.97)}"

    def test_minimum_gap_respected(self):
        """The gap between loop points must satisfy the min_loop_ms constraint."""
        audio = make_sine(440, 2.0)
        min_loop_ms = 100.0
        result = find_loop_point(audio, SR, min_loop_ms=min_loop_ms)
        assert result is not None
        loop_start, loop_end = result
        min_gap_samples = int(min_loop_ms / 1000.0 * SR)
        assert loop_end - loop_start >= min_gap_samples

    def test_both_indices_are_near_zero_crossings(self):
        """
        Both returned indices should be upward zero crossings (or very close to one),
        meaning the sample value transitions from negative to non-negative.
        """
        audio = make_sine(440, 2.0)
        result = find_loop_point(audio, SR)
        assert result is not None
        loop_start, loop_end = result
        # An upward zero crossing at index i means audio[i-1] < 0 and audio[i] >= 0
        # Allow ±2 samples tolerance
        def is_near_upward_zc(idx):
            for offset in range(-2, 3):
                i = idx + offset
                if 1 <= i < len(audio) and audio[i - 1] < 0.0 and audio[i] >= 0.0:
                    return True
            return False
        assert is_near_upward_zc(loop_start), f"loop_start {loop_start} not near an upward zero crossing"
        assert is_near_upward_zc(loop_end), f"loop_end {loop_end} not near an upward zero crossing"

    def test_short_sample_returns_none(self):
        """A sample too short to have a valid search region should return None."""
        audio = make_sine(440, 0.01)  # 10ms — far too short
        result = find_loop_point(audio, SR)
        assert result is None

    def test_silent_sample_returns_none(self):
        """A silent sample has no zero crossings and should return None."""
        audio = np.zeros(SR * 2, dtype=np.float64)
        result = find_loop_point(audio, SR)
        assert result is None

    def test_dc_only_returns_none(self):
        """A constant DC signal has no zero crossings; should return None."""
        audio = np.full(SR * 2, 0.5, dtype=np.float64)
        result = find_loop_point(audio, SR)
        assert result is None

    def test_stereo_input_works(self):
        """find_loop_point should handle 2D (n_samples, 2) stereo arrays."""
        mono = make_sine(440, 2.0)
        stereo = np.stack([mono, mono * 0.8], axis=1)
        result = find_loop_point(stereo, SR)
        assert result is not None
        loop_start, loop_end = result
        assert loop_start < loop_end

    def test_stereo_loop_region_bounds(self):
        """Stereo loop points should also fall within the search region."""
        mono = make_sine(220, 2.0)
        stereo = np.stack([mono, mono], axis=1)
        n = len(mono)
        result = find_loop_point(stereo, SR, region_start_frac=0.85, region_end_frac=0.97)
        assert result is not None
        loop_start, loop_end = result
        assert loop_start >= int(n * 0.85)
        assert loop_end <= int(n * 0.97)

    def test_custom_region_bounds(self):
        """Custom region fractions should be respected."""
        audio = make_sine(440, 2.0)
        n = len(audio)
        result = find_loop_point(audio, SR, region_start_frac=0.60, region_end_frac=0.80)
        assert result is not None
        loop_start, loop_end = result
        assert loop_start >= int(n * 0.60)
        assert loop_end <= int(n * 0.80)

    def test_multiple_frequencies(self):
        """Works for different frequencies (different zero crossing densities)."""
        for freq in [110, 220, 440, 880]:
            audio = make_sine(freq, 2.0)
            result = find_loop_point(audio, SR)
            assert result is not None, f"Expected loop point for freq={freq}"

    def test_amplitude_similarity(self):
        """
        For a pure sine the amplitude should be stable; the returned pair should
        have very similar local RMS (high amplitude similarity).
        """
        audio = make_sine(440, 2.0, amplitude=0.5)
        result = find_loop_point(audio, SR)
        assert result is not None
        loop_start, loop_end = result

        rms_win = max(1, int(0.010 * SR))
        def local_rms(idx):
            s = max(0, idx - rms_win // 2)
            e = min(len(audio), idx + rms_win // 2)
            chunk = audio[s:e]
            return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

        rms_s = local_rms(loop_start)
        rms_e = local_rms(loop_end)
        ratio = abs(rms_s - rms_e) / (rms_s + rms_e + 1e-10)
        assert ratio < 0.2, f"Amplitude mismatch ratio {ratio:.3f} too high"


# ---------------------------------------------------------------------------
# Tests for LoopFinderProcessor (ProcessingObject subclass)
# ---------------------------------------------------------------------------

class TestLoopFinderProcessor:

    def test_sets_loop_points_on_sustain(self):
        """Processor should set loop_start and loop_end on sustain samples."""
        audio = make_sine(440, 2.0)
        sample = make_sample(audio)
        data = make_instrument([sample])

        proc = LoopFinderProcessor()
        result = proc.process(data)

        assert result.sustain[0].loop_start is not None
        assert result.sustain[0].loop_end is not None

    def test_returns_instrument_data(self):
        """process() must return an InstrumentData instance."""
        audio = make_sine(440, 2.0)
        data = make_instrument([make_sample(audio)])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        assert isinstance(result, InstrumentData)

    def test_does_not_modify_release_samples(self):
        """Release samples should remain with loop_start=None, loop_end=None."""
        sustain_audio = make_sine(440, 2.0)
        release_audio = make_sine(440, 1.0)
        sustain = make_sample(sustain_audio)
        release = make_sample(release_audio)
        data = InstrumentData(
            sustain=[sustain],
            release=[release],
            meta=make_meta(),
        )
        proc = LoopFinderProcessor()
        result = proc.process(data)

        assert result.release[0].loop_start is None
        assert result.release[0].loop_end is None

    def test_loop_start_before_loop_end_processor(self):
        """Processor output must have loop_start < loop_end."""
        audio = make_sine(440, 2.0)
        data = make_instrument([make_sample(audio)])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        s = result.sustain[0]
        assert s.loop_start < s.loop_end

    def test_loop_points_in_valid_region(self):
        """Processor: loop points should be in the 85%+ region of the sample."""
        audio = make_sine(440, 2.0)
        n = len(audio)
        data = make_instrument([make_sample(audio)])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        s = result.sustain[0]
        assert s.loop_start >= int(n * 0.85), f"loop_start {s.loop_start} too early (n={n})"

    def test_multiple_sustain_samples_all_get_loops(self):
        """All sustain samples should receive loop points."""
        samples = [make_sample(make_sine(440, 2.0), midi_note=60, velocity=v) for v in [32, 64, 96]]
        data = make_instrument(samples)
        proc = LoopFinderProcessor()
        result = proc.process(data)
        for s in result.sustain:
            assert s.loop_start is not None, f"loop_start missing for note={s.midi_note} vel={s.velocity}"
            assert s.loop_end is not None, f"loop_end missing for note={s.midi_note} vel={s.velocity}"

    def test_short_sample_leaves_loop_points_none(self):
        """Too-short samples should remain with loop_start=None (graceful skip)."""
        audio = make_sine(440, 0.01)  # 10ms — too short for a valid loop region
        sample = make_sample(audio)
        data = make_instrument([sample])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        # Should not raise; loop points should be None
        s = result.sustain[0]
        assert s.loop_start is None
        assert s.loop_end is None

    def test_empty_sustain_list_no_error(self):
        """Empty sustain list should not raise."""
        data = make_instrument([])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        assert result.sustain == []

    def test_stereo_sustain_gets_loop_points(self):
        """Stereo sustain samples should also get valid loop points."""
        mono = make_sine(440, 2.0)
        stereo = np.stack([mono, mono * 0.9], axis=1)
        sample = make_sample(stereo)
        data = make_instrument([sample])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        s = result.sustain[0]
        assert s.loop_start is not None
        assert s.loop_end is not None
        assert s.loop_start < s.loop_end

    def test_loop_end_before_tail(self):
        """
        loop_end should be before the crossfade tail starts (region_end_frac < 1.0).
        With loop_crossfade_ms=20ms and a 2s sample, region_end should be ~0.97 or less.
        """
        audio = make_sine(440, 2.0)
        n = len(audio)
        data = make_instrument([make_sample(audio)], loop_crossfade_ms=20.0)
        proc = LoopFinderProcessor()
        result = proc.process(data)
        s = result.sustain[0]
        # 3% tail minimum: loop_end must be <= 97% of n
        assert s.loop_end <= int(n * 0.97) + 5, f"loop_end {s.loop_end} past expected tail margin"

    def test_loop_crossfade_ms_affects_region_end(self):
        """
        A larger loop_crossfade_ms should result in an equal or earlier region_end,
        meaning loop_end won't be pushed further into the tail.
        """
        audio = make_sine(440, 2.0)
        n = len(audio)

        # Small crossfade
        data_small = make_instrument([make_sample(audio.copy())], loop_crossfade_ms=5.0)
        proc = LoopFinderProcessor()
        r_small = proc.process(data_small)

        # Large crossfade
        data_large = make_instrument([make_sample(audio.copy())], loop_crossfade_ms=200.0)
        r_large = proc.process(data_large)

        # Large crossfade means more tail reserved, so loop_end should be <= small crossfade's loop_end
        if r_large.sustain[0].loop_end is not None and r_small.sustain[0].loop_end is not None:
            assert r_large.sustain[0].loop_end <= r_small.sustain[0].loop_end + 100  # allow small tolerance

    def test_indices_are_integers(self):
        """loop_start and loop_end must be Python ints."""
        audio = make_sine(440, 2.0)
        data = make_instrument([make_sample(audio)])
        proc = LoopFinderProcessor()
        result = proc.process(data)
        s = result.sustain[0]
        assert isinstance(s.loop_start, int)
        assert isinstance(s.loop_end, int)

    def test_different_frequencies_all_get_loops(self):
        """Processor should handle different fundamental frequencies."""
        for freq in [110, 220, 440, 880]:
            audio = make_sine(freq, 2.0)
            data = make_instrument([make_sample(audio)])
            proc = LoopFinderProcessor()
            result = proc.process(data)
            s = result.sustain[0]
            assert s.loop_start is not None, f"No loop for freq={freq}"

    def test_audio_not_modified(self):
        """The processor must not modify the audio data itself."""
        audio = make_sine(440, 2.0)
        original = audio.copy()
        data = make_instrument([make_sample(audio)])
        proc = LoopFinderProcessor()
        proc.process(data)
        np.testing.assert_array_equal(data.sustain[0].audio, original)
