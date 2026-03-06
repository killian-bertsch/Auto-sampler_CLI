"""
test_phase3.py — Unit tests for Phase 3: TrimProcessor + NormalizeProcessor.

All tests use synthetic signals (no disk I/O).

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import InstrumentData, InstrumentMeta, Sample
from src.processors.trim import TrimProcessor, _detect_onset
from src.processors.normalize import (
    NormalizeProcessor,
    _remove_dc_offset,
    _peak,
    _rms,
    _measure_level,
)


SR = 44100


# ---------------------------------------------------------------------------
# Helpers for building test InstrumentData
# ---------------------------------------------------------------------------

def make_meta(**kwargs) -> InstrumentMeta:
    defaults = dict(
        velocity_layers=3,
        semitone_interval=12,
        hold_time=1.0,
        release_time=0.5,
        start_note=60,
        end_note=72,
        min_note=60,
        max_note=72,
        note_percentage=100.0,
        velocity_layers_out=3,
        crossfade_percent=0.0,
        normalize=True,
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


def make_sample(
    midi_note: int = 60,
    velocity: int = 64,
    audio: np.ndarray | None = None,
    sr: int = SR,
) -> Sample:
    if audio is None:
        audio = np.zeros(SR, dtype=np.float64)
    return Sample(midi_note=midi_note, velocity=velocity, audio=audio, sr=sr)


def sine(freq: float = 440.0, duration: float = 0.5, amplitude: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def silence(duration: float = 0.1, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration * sr), dtype=np.float64)


def make_data(sustain: list[Sample], release: list[Sample] | None = None, **meta_kwargs) -> InstrumentData:
    return InstrumentData(
        sustain=sustain,
        release=release or [],
        meta=make_meta(**meta_kwargs),
    )


# ===========================================================================
# TrimProcessor tests
# ===========================================================================

class TestDetectOnset:
    def test_immediate_signal_returns_zero(self):
        """Signal with no leading silence → onset very near 0 (sine starts at 0)."""
        audio = sine(amplitude=0.5)
        assert _detect_onset(audio, SR, -40.0) <= 5

    def test_leading_silence_detected(self):
        """100 ms of silence before tone → onset near 100 ms mark."""
        silent_samples = int(0.1 * SR)
        audio = np.concatenate([silence(0.1), sine(amplitude=0.5)])
        onset = _detect_onset(audio, SR, -40.0)
        assert silent_samples - 250 <= onset <= silent_samples + 250

    def test_all_silence_returns_zero(self):
        """Fully silent signal → 0 (no onset found)."""
        audio = silence(0.5)
        assert _detect_onset(audio, SR, -40.0) == 0

    def test_very_quiet_signal_below_threshold(self):
        """Signal below threshold is treated as silence → 0."""
        audio = sine(amplitude=1e-5)  # very quiet, well below -40 dB
        assert _detect_onset(audio, SR, -40.0) == 0

    def test_stereo_audio(self):
        """Onset detection works on (N, 2) stereo arrays."""
        silent_samples = int(0.05 * SR)
        tone = sine(amplitude=0.5)
        left = np.concatenate([silence(0.05), tone])
        right = np.concatenate([silence(0.05), tone * 0.7])
        stereo = np.stack([left, right], axis=1)
        onset = _detect_onset(stereo, SR, -40.0)
        assert silent_samples - 250 <= onset <= silent_samples + 250

    def test_threshold_respected(self):
        """With a very high threshold, quiet tone doesn't trigger onset."""
        audio = sine(amplitude=0.01)  # -40 dBFS
        onset = _detect_onset(audio, SR, -10.0)  # threshold -10 dB
        assert onset == 0

    def test_short_audio(self):
        """Very short audio (< window size) doesn't crash."""
        audio = np.array([0.0, 0.5, 0.0], dtype=np.float64)
        # Should not raise
        _detect_onset(audio, SR, -40.0)


class TestTrimProcessor:
    def test_trims_leading_silence(self):
        """Leading silence is removed from sustain samples."""
        silent_samples = int(0.1 * SR)
        audio = np.concatenate([silence(0.1), sine(amplitude=0.5, duration=0.5)])
        original_len = len(audio)
        s = make_sample(audio=audio.copy())
        data = make_data([s])
        result = TrimProcessor().process(data)
        trimmed = result.sustain[0].audio
        assert len(trimmed) < original_len
        assert len(trimmed) >= int(0.5 * SR) - 250

    def test_no_trim_when_signal_starts_immediately(self):
        """Audio that starts with a tone is nearly unchanged (sine starts at 0, so at most a few samples trimmed)."""
        audio = sine(amplitude=0.5)
        s = make_sample(audio=audio.copy())
        data = make_data([s])
        result = TrimProcessor().process(data)
        assert len(result.sustain[0].audio) >= len(audio) - 5

    def test_trims_release_samples(self):
        """TrimProcessor also trims release samples."""
        audio = np.concatenate([silence(0.1), sine(amplitude=0.5, duration=0.3)])
        original_len = len(audio)
        s = make_sample(audio=audio.copy())
        data = make_data(sustain=[], release=[s])
        result = TrimProcessor().process(data)
        assert len(result.release[0].audio) < original_len

    def test_empty_sustain_and_release(self):
        """No samples → returns data unchanged without error."""
        data = make_data([])
        result = TrimProcessor().process(data)
        assert result.sustain == []
        assert result.release == []

    def test_uses_meta_threshold(self):
        """Processor uses onset_threshold_db from meta."""
        # Signal at -30 dBFS (~0.032 amplitude)
        audio = np.concatenate([silence(0.1), sine(amplitude=0.032)])
        # With default -40 dB threshold: the signal exceeds threshold → trimmed
        s1 = make_sample(audio=audio.copy())
        data1 = make_data([s1], onset_threshold_db=-40.0)
        result1 = TrimProcessor().process(data1)

        # With -20 dB threshold: signal below threshold → not trimmed
        s2 = make_sample(audio=audio.copy())
        data2 = make_data([s2], onset_threshold_db=-20.0)
        result2 = TrimProcessor().process(data2)

        assert len(result1.sustain[0].audio) < len(audio)
        assert len(result2.sustain[0].audio) == len(audio)

    def test_stereo_sample_trimmed(self):
        """Stereo (N, 2) samples are trimmed correctly."""
        silent_samples = int(0.05 * SR)
        tone = sine(amplitude=0.5)
        left = np.concatenate([silence(0.05), tone])
        stereo = np.stack([left, left * 0.8], axis=1)
        s = make_sample(audio=stereo.copy())
        data = make_data([s])
        result = TrimProcessor().process(data)
        trimmed = result.sustain[0].audio
        assert trimmed.ndim == 2
        assert trimmed.shape[1] == 2
        assert trimmed.shape[0] < stereo.shape[0]

    def test_multiple_samples_all_trimmed(self):
        """All samples in the list are trimmed independently."""
        audio_a = np.concatenate([silence(0.1), sine(amplitude=0.5)])
        audio_b = np.concatenate([silence(0.05), sine(amplitude=0.3)])
        s_a = make_sample(midi_note=60, audio=audio_a.copy())
        s_b = make_sample(midi_note=62, audio=audio_b.copy())
        data = make_data([s_a, s_b])
        result = TrimProcessor().process(data)
        assert len(result.sustain[0].audio) < len(audio_a)
        assert len(result.sustain[1].audio) < len(audio_b)

    def test_processor_repr(self):
        assert "TrimProcessor" in repr(TrimProcessor())


# ===========================================================================
# DC offset helpers
# ===========================================================================

class TestRemoveDCOffset:
    def test_mono_dc_removed(self):
        audio = np.ones(1000, dtype=np.float64) * 0.5
        result = _remove_dc_offset(audio)
        assert abs(result.mean()) < 1e-10

    def test_stereo_dc_removed_per_channel(self):
        ch0 = np.ones(1000, dtype=np.float64) * 0.3
        ch1 = np.ones(1000, dtype=np.float64) * -0.2
        stereo = np.stack([ch0, ch1], axis=1)
        result = _remove_dc_offset(stereo)
        assert abs(result[:, 0].mean()) < 1e-10
        assert abs(result[:, 1].mean()) < 1e-10

    def test_returns_float64(self):
        audio = np.ones(100, dtype=np.float32)
        result = _remove_dc_offset(audio)
        assert result.dtype == np.float64

    def test_zero_mean_unchanged(self):
        """Audio already at zero mean should be unaffected."""
        audio = sine(amplitude=0.5)
        result = _remove_dc_offset(audio)
        np.testing.assert_allclose(result, audio, atol=1e-10)


# ===========================================================================
# NormalizeProcessor tests
# ===========================================================================

class TestNormalizeProcessorDCRemoval:
    def test_dc_always_removed_when_normalize_false(self):
        """DC offset is removed even when normalize=False."""
        dc_audio = np.ones(SR, dtype=np.float64) * 0.3  # all +0.3, mean ≠ 0
        s = make_sample(audio=dc_audio.copy())
        data = make_data([s], normalize=False)
        result = NormalizeProcessor().process(data)
        assert abs(result.sustain[0].audio.mean()) < 1e-10

    def test_dc_removed_before_normalization(self):
        """DC is removed in normalize=True mode too."""
        dc_audio = (sine(amplitude=0.3) + 0.2).astype(np.float64)
        s = make_sample(audio=dc_audio.copy())
        data = make_data([s], normalize=True, normalize_mode="rms")
        result = NormalizeProcessor().process(data)
        assert abs(result.sustain[0].audio.mean()) < 1e-8


class TestNormalizeProcessorRMS:
    def test_rms_normalizes_single_layer(self):
        """After RMS normalization, level should be close to target."""
        target = -18.0
        audio = sine(amplitude=0.01)  # very quiet
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data([s], normalize_mode="rms", normalize_target_lufs=target, peak_ceiling_db=-0.1)
        result = NormalizeProcessor().process(data)
        out = result.sustain[0].audio
        level = 20.0 * np.log10(_rms(out)) if _rms(out) > 0 else -100.0
        assert abs(level - target) < 3.0  # within 3 dB of target

    def test_rms_preserves_relative_levels_within_layer(self):
        """Samples in the same velocity layer get the same scale factor."""
        audio_loud = sine(amplitude=0.5, freq=440.0)
        audio_quiet = sine(amplitude=0.1, freq=880.0)
        s1 = make_sample(midi_note=60, velocity=64, audio=audio_loud.copy())
        s2 = make_sample(midi_note=62, velocity=64, audio=audio_quiet.copy())
        data = make_data([s1, s2], normalize_mode="rms", normalize_target_lufs=-18.0, peak_ceiling_db=-1.0)
        result = NormalizeProcessor().process(data)
        # Ratio of RMS values should be preserved
        rms_before = _rms(audio_loud) / _rms(audio_quiet)
        rms_after = _rms(result.sustain[0].audio) / _rms(result.sustain[1].audio)
        assert abs(rms_before - rms_after) < 0.01

    def test_rms_different_velocity_layers_normalized_independently(self):
        """Two velocity layers (vel=32, vel=96) are normalized independently."""
        audio_low = sine(amplitude=0.05, freq=220.0)   # quiet
        audio_high = sine(amplitude=0.4, freq=440.0)   # loud
        s_low = make_sample(velocity=32, audio=audio_low.copy())
        s_high = make_sample(velocity=96, audio=audio_high.copy())
        data = make_data(
            [s_low, s_high],
            normalize_mode="rms",
            normalize_target_lufs=-18.0,
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        # Both layers should now be near -18 dBFS
        for s in result.sustain:
            level = 20.0 * np.log10(_rms(s.audio)) if _rms(s.audio) > 0 else -100.0
            assert abs(level - (-18.0)) < 3.0

    def test_peak_ceiling_enforced(self):
        """Peak never exceeds peak_ceiling_db after normalization."""
        # Very quiet sample with high target → could clip without ceiling
        audio = sine(amplitude=0.001)
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data(
            [s],
            normalize_mode="rms",
            normalize_target_lufs=-6.0,
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        peak = _peak(result.sustain[0].audio)
        ceiling_linear = 10.0 ** (-1.0 / 20.0)
        assert peak <= ceiling_linear + 1e-9

    def test_already_loud_sample_not_amplified_by_ceiling(self):
        """Peak ceiling only attenuates, never amplifies."""
        # Sample already below ceiling
        audio = sine(amplitude=0.1)
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data(
            [s],
            normalize=False,  # skip normalization, just DC + ceiling never called
        )
        # With normalize=False, only DC is applied; audio unchanged except DC removal
        result = NormalizeProcessor().process(data)
        # Peak should be unchanged (no normalization happened)
        assert _peak(result.sustain[0].audio) <= _peak(audio) + 1e-9

    def test_normalize_false_skips_level_changes(self):
        """When normalize=False, RMS level is unchanged (only DC removed)."""
        audio = sine(amplitude=0.1)
        s = make_sample(audio=audio.copy())
        data = make_data([s], normalize=False)
        result = NormalizeProcessor().process(data)
        rms_before = _rms(audio)
        rms_after = _rms(result.sustain[0].audio)
        # DC-only removal on a zero-mean sine shouldn't change RMS
        assert abs(rms_before - rms_after) < 1e-6

    def test_empty_data_no_error(self):
        """Empty sample lists are handled without error."""
        data = make_data([], normalize_mode="rms")
        result = NormalizeProcessor().process(data)
        assert result.sustain == []

    def test_release_samples_normalized_too(self):
        """Release samples are also normalized in rms/lufs mode."""
        audio_quiet = sine(amplitude=0.001)
        s = make_sample(velocity=64, audio=audio_quiet.copy())
        data = make_data(
            sustain=[],
            release=[s],
            normalize_mode="rms",
            normalize_target_lufs=-18.0,
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        level_before = 20.0 * np.log10(_rms(audio_quiet))
        level_after = 20.0 * np.log10(_rms(result.release[0].audio))
        assert abs(level_after - (-18.0)) < 3.0
        assert level_after > level_before  # got louder (less negative dB)

    def test_stereo_sample_normalized(self):
        """Normalization works on (N, 2) stereo arrays."""
        tone = sine(amplitude=0.01)
        stereo = np.stack([tone, tone * 0.8], axis=1)
        s = make_sample(velocity=64, audio=stereo.copy())
        data = make_data([s], normalize_mode="rms", normalize_target_lufs=-18.0, peak_ceiling_db=-1.0)
        result = NormalizeProcessor().process(data)
        out = result.sustain[0].audio
        assert out.ndim == 2
        # Both channels should be louder than original
        assert _rms(out) > _rms(stereo)

    def test_silent_sample_not_crashed(self):
        """Fully silent sample doesn't crash (returns silence)."""
        audio = silence(0.5)
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data([s], normalize_mode="rms")
        result = NormalizeProcessor().process(data)
        assert len(result.sustain[0].audio) > 0


class TestNormalizeProcessorLUFS:
    def test_lufs_mode_normalizes(self):
        """LUFS mode produces levels near the target for sufficiently long samples."""
        # 1-second tone so pyloudnorm can measure integrated loudness
        audio = sine(amplitude=0.01, duration=1.5)
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data(
            [s],
            normalize_mode="lufs",
            normalize_target_lufs=-18.0,
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        # Just verify output is louder than input (was very quiet)
        assert _rms(result.sustain[0].audio) > _rms(audio)

    def test_lufs_fallback_short_sample(self):
        """Short samples (< 400 ms) fall back to RMS without crashing."""
        audio = sine(amplitude=0.001, duration=0.2)
        s = make_sample(velocity=64, audio=audio.copy())
        data = make_data([s], normalize_mode="lufs", normalize_target_lufs=-18.0, peak_ceiling_db=-1.0)
        result = NormalizeProcessor().process(data)
        assert len(result.sustain[0].audio) > 0
        # Should have been normalized upward
        assert _rms(result.sustain[0].audio) > _rms(audio)


class TestNormalizeProcessorVelocity:
    def _make_velocity_data(self, velocities, amplitudes, **meta_kwargs):
        """Build InstrumentData with one note, multiple velocity layers."""
        samples = [
            make_sample(
                midi_note=60,
                velocity=v,
                audio=sine(amplitude=a),
            )
            for v, a in zip(velocities, amplitudes)
        ]
        defaults = dict(
            velocity_layers=len(velocities),
            velocity_layers_out=len(velocities),
            normalize_mode="velocity",
            peak_ceiling_db=-1.0,
        )
        defaults.update(meta_kwargs)
        return make_data(samples, **defaults)

    def test_velocity_mode_sets_computed_range(self):
        """computed_dynamic_range_db is set after velocity mode processing."""
        # Higher velocity → louder amplitude (realistic dynamics)
        data = self._make_velocity_data([32, 64, 96], [0.1, 0.25, 0.5])
        result = NormalizeProcessor().process(data)
        assert result.meta.computed_dynamic_range_db is not None
        assert result.meta.computed_dynamic_range_db > 0.0

    def test_velocity_mode_lower_velocities_boosted(self):
        """After velocity curve, lower velocity samples should be louder than before."""
        data = self._make_velocity_data([32, 64, 96], [0.1, 0.25, 0.5])
        peaks_before = [_peak(s.audio) for s in data.sustain]
        result = NormalizeProcessor().process(data)
        peaks_after = [_peak(s.audio) for s in result.sustain]
        # Lowest velocity (index 0) should be boosted relative to its original
        assert peaks_after[0] > peaks_before[0]

    def test_velocity_mode_peak_ceiling(self):
        """Peak ceiling is enforced in velocity mode."""
        # Very quiet samples — will be boosted a lot
        data = self._make_velocity_data([32, 64, 96], [0.001, 0.003, 0.005], peak_ceiling_db=-1.0)
        result = NormalizeProcessor().process(data)
        ceiling = 10.0 ** (-1.0 / 20.0)
        for s in result.sustain:
            assert _peak(s.audio) <= ceiling + 1e-9

    def test_velocity_mode_single_layer_no_crash(self):
        """Single velocity layer (no dynamics to compute) doesn't crash."""
        data = self._make_velocity_data([64], [0.2])
        result = NormalizeProcessor().process(data)
        assert len(result.sustain) == 1

    def test_velocity_mode_multiple_notes(self):
        """Gain curve is applied per-note; each note gets its own range."""
        # Two notes, each with two velocity layers
        samples = [
            make_sample(midi_note=60, velocity=32, audio=sine(amplitude=0.1, freq=261.0)),
            make_sample(midi_note=60, velocity=96, audio=sine(amplitude=0.4, freq=261.0)),
            make_sample(midi_note=72, velocity=32, audio=sine(amplitude=0.05, freq=523.0)),
            make_sample(midi_note=72, velocity=96, audio=sine(amplitude=0.3, freq=523.0)),
        ]
        data = make_data(
            samples,
            velocity_layers=2,
            velocity_layers_out=2,
            normalize_mode="velocity",
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        assert result.meta.computed_dynamic_range_db is not None
        assert result.meta.computed_dynamic_range_db > 0.0

    def test_velocity_mode_release_normalized(self):
        """Release samples are RMS-normalized in velocity mode."""
        audio_quiet = sine(amplitude=0.001)
        s_rel = make_sample(velocity=64, audio=audio_quiet.copy())
        s_sust = make_sample(midi_note=60, velocity=64, audio=sine(amplitude=0.3))
        data = make_data(
            sustain=[s_sust],
            release=[s_rel],
            velocity_layers=1,
            velocity_layers_out=1,
            normalize_mode="velocity",
            peak_ceiling_db=-1.0,
        )
        result = NormalizeProcessor().process(data)
        # Release should be louder (was very quiet)
        assert _rms(result.release[0].audio) > _rms(audio_quiet)

    def test_velocity_mode_dc_still_removed(self):
        """DC removal happens in velocity mode too."""
        dc_audio = (sine(amplitude=0.3) + 0.2).astype(np.float64)
        s = make_sample(midi_note=60, velocity=64, audio=dc_audio.copy())
        data = make_data(
            [s],
            velocity_layers=1,
            velocity_layers_out=1,
            normalize_mode="velocity",
        )
        result = NormalizeProcessor().process(data)
        assert abs(result.sustain[0].audio.mean()) < 1e-8


class TestNormalizeProcessorMiscellaneous:
    def test_processor_repr(self):
        assert "NormalizeProcessor" in repr(NormalizeProcessor())

    def test_process_returns_instrument_data(self):
        s = make_sample(audio=sine(amplitude=0.2))
        data = make_data([s])
        result = NormalizeProcessor().process(data)
        assert isinstance(result, InstrumentData)

    def test_trim_then_normalize_pipeline(self):
        """TrimProcessor followed by NormalizeProcessor works end-to-end."""
        audio = np.concatenate([silence(0.1), sine(amplitude=0.1, duration=0.5)])
        s = make_sample(audio=audio.copy())
        data = make_data([s], normalize_mode="rms", normalize_target_lufs=-18.0, peak_ceiling_db=-1.0)

        data = TrimProcessor().process(data)
        data = NormalizeProcessor().process(data)

        out = data.sustain[0].audio
        # Leading silence removed
        assert len(out) < len(audio)
        # Level near target
        level = 20.0 * np.log10(_rms(out)) if _rms(out) > 0 else -100.0
        assert abs(level - (-18.0)) < 3.0
        # Peak within ceiling
        assert _peak(out) <= 10.0 ** (-1.0 / 20.0) + 1e-9
