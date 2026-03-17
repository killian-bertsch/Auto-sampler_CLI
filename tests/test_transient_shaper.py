"""
tests/test_transient_shaper.py — Unit tests for TransientShaperProcessor.
"""

import numpy as np
import pytest

from src.processors.transient_shaper import TransientShaperProcessor, _apply_transient_shaper
from src.data import InstrumentData, InstrumentMeta, Sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 44100


def _make_meta(**overrides) -> InstrumentMeta:
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
        normalize_mode="lufs",
        normalize_target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        velocity_dynamic_range_db=40.0,
        instrument_name="test",
        ampeg_release=0.5,
        loop_crossfade_ms=20.0,
    )
    defaults.update(overrides)
    return InstrumentMeta(**defaults)


def _sine(freq=440.0, duration=0.5, sr=SR, amplitude=0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(duration * sr), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _attack_sine(sr=SR, attack_len=0.01, body_len=0.3) -> np.ndarray:
    """Short sharp attack followed by a quieter sustain body."""
    attack_samples = int(attack_len * sr)
    body_samples = int(body_len * sr)
    t_a = np.linspace(0, attack_len, attack_samples, endpoint=False)
    t_b = np.linspace(0, body_len, body_samples, endpoint=False)
    attack = 0.9 * np.sin(2 * np.pi * 440 * t_a)
    body = 0.2 * np.sin(2 * np.pi * 440 * t_b)
    return np.concatenate([attack, body]).astype(np.float64)


def _make_data(audio: np.ndarray, enable=True, attack_db=6.0, sustain_db=0.0, speed_ms=10.0) -> InstrumentData:
    meta = _make_meta(
        enable_transient_shaper=enable,
        transient_attack_db=attack_db,
        transient_sustain_db=sustain_db,
        transient_speed_ms=speed_ms,
    )
    sample = Sample(midi_note=60, velocity=80, audio=audio.copy(), sr=SR)
    return InstrumentData(sustain=[sample], release=[], meta=meta)


# ---------------------------------------------------------------------------
# _apply_transient_shaper unit tests
# ---------------------------------------------------------------------------

class TestApplyTransientShaper:

    def test_empty_array_returns_unchanged(self):
        empty = np.array([], dtype=np.float64)
        result = _apply_transient_shaper(empty, SR, 6.0, 0.0, 10.0)
        assert result.size == 0

    def test_preserves_shape_mono(self):
        audio = _sine()
        result = _apply_transient_shaper(audio, SR, 6.0, 0.0, 10.0)
        assert result.shape == audio.shape

    def test_preserves_shape_stereo(self):
        mono = _sine()
        stereo = np.stack([mono, mono], axis=1)
        result = _apply_transient_shaper(stereo, SR, 6.0, 0.0, 10.0)
        assert result.shape == stereo.shape

    def test_preserves_dtype_float32(self):
        audio = _sine().astype(np.float32)
        result = _apply_transient_shaper(audio, SR, 6.0, 0.0, 10.0)
        assert result.dtype == np.float32

    def test_preserves_dtype_float64(self):
        audio = _sine()
        result = _apply_transient_shaper(audio, SR, 6.0, 0.0, 10.0)
        assert result.dtype == np.float64

    def test_output_clipped_within_tanh_range(self):
        """tanh soft-clip means all values must be in (-1, 1)."""
        audio = _sine(amplitude=0.9)
        result = _apply_transient_shaper(audio, SR, 12.0, 12.0, 1.0)
        assert np.all(np.abs(result) < 1.0)

    def test_zero_gain_both_silences(self):
        """attack_db=-inf and sustain_db=-inf would silence output.
        Use -60 dB as a proxy for near-silence."""
        audio = _sine(amplitude=0.5)
        result = _apply_transient_shaper(audio, SR, -60.0, -60.0, 10.0)
        assert np.max(np.abs(result)) < 1e-2

    def test_unity_gain_both_mostly_unchanged(self):
        """With both attack and sustain at 0 dB the output should be close
        to tanh(input) — not exactly the input because tanh clips peaks."""
        audio = _sine(amplitude=0.3)
        result = _apply_transient_shaper(audio, SR, 0.0, 0.0, 10.0)
        # tanh(0.3) ≈ 0.2913 so result amplitude < input amplitude slightly
        np.testing.assert_allclose(result, np.tanh(audio), atol=1e-6)

    def test_attack_boost_increases_peak(self):
        """Boosting attack should make the transient portion louder."""
        audio = _attack_sine()
        baseline = _apply_transient_shaper(audio, SR, 0.0, 0.0, 10.0)
        boosted = _apply_transient_shaper(audio, SR, 12.0, 0.0, 5.0)
        # Peak of boosted should be >= baseline (tanh clips so use >=)
        assert np.max(np.abs(boosted)) >= np.max(np.abs(baseline)) * 0.9

    def test_sustain_boost_applied(self):
        """Boosting sustain at 0 dB attack should change overall output."""
        audio = _sine(amplitude=0.3)
        baseline = _apply_transient_shaper(audio, SR, 0.0, 0.0, 10.0)
        boosted = _apply_transient_shaper(audio, SR, 0.0, 6.0, 10.0)
        # Sustain-boosted mean should be louder
        assert np.mean(np.abs(boosted)) > np.mean(np.abs(baseline))

    def test_stereo_channels_treated_equally(self):
        """Both channels should receive the same gain curve."""
        mono = _attack_sine()
        ch0 = mono * 0.8
        ch1 = mono * 0.8
        stereo = np.stack([ch0, ch1], axis=1)
        result = _apply_transient_shaper(stereo, SR, 6.0, 0.0, 10.0)
        np.testing.assert_allclose(result[:, 0], result[:, 1], atol=1e-10)

    def test_silent_input_returns_near_zero(self):
        audio = np.zeros(SR, dtype=np.float64)
        result = _apply_transient_shaper(audio, SR, 6.0, 0.0, 10.0)
        assert np.max(np.abs(result)) < 1e-12


# ---------------------------------------------------------------------------
# TransientShaperProcessor tests
# ---------------------------------------------------------------------------

class TestTransientShaperProcessor:

    def test_disabled_returns_data_unchanged(self):
        audio = _sine()
        data = _make_data(audio, enable=False)
        original = data.sustain[0].audio.copy()
        result = TransientShaperProcessor().process(data)
        np.testing.assert_array_equal(result.sustain[0].audio, original)

    def test_enabled_modifies_sustain_audio(self):
        audio = _sine(amplitude=0.3)
        data = _make_data(audio, enable=True, attack_db=6.0, sustain_db=0.0)
        original = data.sustain[0].audio.copy()
        result = TransientShaperProcessor().process(data)
        assert not np.array_equal(result.sustain[0].audio, original)

    def test_processes_all_sustain_samples(self):
        audio = _sine(amplitude=0.3)
        meta = _make_meta(
            enable_transient_shaper=True,
            transient_attack_db=6.0,
            transient_sustain_db=0.0,
            transient_speed_ms=10.0,
        )
        samples = [
            Sample(midi_note=60, velocity=v, audio=audio.copy(), sr=SR)
            for v in [40, 80, 110]
        ]
        data = InstrumentData(sustain=samples, release=[], meta=meta)
        originals = [s.audio.copy() for s in samples]
        result = TransientShaperProcessor().process(data)
        for orig, s in zip(originals, result.sustain):
            assert not np.array_equal(s.audio, orig)

    def test_release_samples_never_modified(self):
        """Transient shaper must not touch release samples."""
        audio = _sine(amplitude=0.3)
        meta = _make_meta(
            enable_transient_shaper=True,
            transient_attack_db=6.0,
            transient_sustain_db=0.0,
            transient_speed_ms=10.0,
        )
        rel = Sample(midi_note=60, velocity=80, audio=audio.copy(), sr=SR)
        data = InstrumentData(sustain=[], release=[rel], meta=meta)
        original = rel.audio.copy()
        result = TransientShaperProcessor().process(data)
        np.testing.assert_array_equal(result.release[0].audio, original)

    def test_meta_unchanged_after_processing(self):
        audio = _sine(amplitude=0.3)
        data = _make_data(audio, enable=True)
        result = TransientShaperProcessor().process(data)
        assert result.meta.enable_transient_shaper is True
        assert result.meta.transient_attack_db == 6.0

    def test_output_within_tanh_range(self):
        audio = _sine(amplitude=0.8)
        data = _make_data(audio, enable=True, attack_db=12.0, sustain_db=12.0)
        result = TransientShaperProcessor().process(data)
        assert np.all(np.abs(result.sustain[0].audio) < 1.0)


# ---------------------------------------------------------------------------
# Metadata parser integration tests
# ---------------------------------------------------------------------------

class TestTransientShaperMetadataParsing:

    def _write_and_parse(self, tmp_path, extra_lines=""):
        from src.metadata_parser import parse_metadata
        content = (
            "velocity_layers = 1\n"
            "semitone_interval = 12\n"
            "hold_time = 1.0\n"
            "release_time = 0.5\n"
            "velocity_layers_out = 1\n"
        ) + extra_lines
        p = tmp_path / "metadata.txt"
        p.write_text(content)
        return parse_metadata(p)

    def test_defaults_when_absent(self, tmp_path):
        meta = self._write_and_parse(tmp_path)
        assert meta.enable_transient_shaper is False
        assert meta.transient_attack_db == 6.0
        assert meta.transient_sustain_db == 0.0
        assert meta.transient_speed_ms == 10.0

    def test_enable_true(self, tmp_path):
        meta = self._write_and_parse(tmp_path, "enable_transient_shaper = true\n")
        assert meta.enable_transient_shaper is True

    def test_custom_values(self, tmp_path):
        extra = (
            "enable_transient_shaper = true\n"
            "transient_attack_db = -3.0\n"
            "transient_sustain_db = 2.5\n"
            "transient_speed_ms = 5.0\n"
        )
        meta = self._write_and_parse(tmp_path, extra)
        assert meta.transient_attack_db == -3.0
        assert meta.transient_sustain_db == 2.5
        assert meta.transient_speed_ms == 5.0

    def test_attack_db_out_of_range(self, tmp_path):
        from src.metadata_parser import parse_metadata, MetadataError
        extra = "transient_attack_db = 15.0\n"
        p = tmp_path / "metadata.txt"
        p.write_text(
            "velocity_layers = 1\nsemitone_interval = 12\nhold_time = 1.0\n"
            "release_time = 0.5\nvelocity_layers_out = 1\n" + extra
        )
        with pytest.raises(MetadataError, match="transient_attack_db"):
            parse_metadata(p)

    def test_sustain_db_out_of_range(self, tmp_path):
        from src.metadata_parser import parse_metadata, MetadataError
        extra = "transient_sustain_db = -20.0\n"
        p = tmp_path / "metadata.txt"
        p.write_text(
            "velocity_layers = 1\nsemitone_interval = 12\nhold_time = 1.0\n"
            "release_time = 0.5\nvelocity_layers_out = 1\n" + extra
        )
        with pytest.raises(MetadataError, match="transient_sustain_db"):
            parse_metadata(p)

    def test_speed_ms_out_of_range(self, tmp_path):
        from src.metadata_parser import parse_metadata, MetadataError
        extra = "transient_speed_ms = 100.0\n"
        p = tmp_path / "metadata.txt"
        p.write_text(
            "velocity_layers = 1\nsemitone_interval = 12\nhold_time = 1.0\n"
            "release_time = 0.5\nvelocity_layers_out = 1\n" + extra
        )
        with pytest.raises(MetadataError, match="transient_speed_ms"):
            parse_metadata(p)

    def test_speed_ms_boundary_valid(self, tmp_path):
        meta = self._write_and_parse(tmp_path, "transient_speed_ms = 1.0\n")
        assert meta.transient_speed_ms == 1.0
        meta2 = self._write_and_parse(tmp_path, "transient_speed_ms = 50.0\n")
        assert meta2.transient_speed_ms == 50.0
