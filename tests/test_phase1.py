"""
test_phase1.py — Unit tests for Phase 1: scaffolding, data structures,
                 metadata parsers, and pipeline.

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase1.py -v
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.data import InstrumentData, InstrumentMeta, InputSampleDef, Sample
from src.base import ProcessingObject, PassthroughProcessor
from src.pipeline import Pipeline
from src.metadata_parser import MetadataError, parse_input_meta, parse_output_meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(**overrides) -> InstrumentMeta:
    """Return a minimal valid InstrumentMeta with optional field overrides."""
    defaults = dict(
        min_note=21,
        max_note=108,
        note_percentage=100.0,
        crossfade_percent=20.0,
        normalize=True,
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


def make_sample(midi_note=60, velocity=64, duration_samples=1000, sr=44100) -> Sample:
    audio = np.zeros(duration_samples, dtype=np.float64)
    return Sample(midi_note=midi_note, velocity=velocity, audio=audio, sr=sr)


def make_instrument_data(n_notes=3, n_velocities=2) -> InstrumentData:
    notes = [60, 63, 66][:n_notes]
    velocities = [32, 127][:n_velocities]
    sustain = [make_sample(n, v) for n in notes for v in velocities]
    return InstrumentData(sustain=sustain, release=[], meta=make_meta())


def write_input_toml(tmp_path: Path, defs, collapse_to_mono=False) -> Path:
    """Write an input.toml to tmp_path using TOML inline-table array format."""
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


def write_output_toml(tmp_path: Path, content: str = "") -> Path:
    """Write an output.toml to tmp_path."""
    p = tmp_path / "output.toml"
    p.write_text(content)
    return p


MINIMAL_INPUT_DEFS = [
    {"note": 60, "velocity": 64, "hold_s": 1.0, "tail_s": 0.5, "is_release": False},
]


# ---------------------------------------------------------------------------
# InputSampleDef dataclass
# ---------------------------------------------------------------------------

class TestInputSampleDef:
    def test_fields(self):
        d = InputSampleDef(note=60, velocity=64, hold_s=5.0, tail_s=1.0, is_release=False)
        assert d.note == 60
        assert d.velocity == 64
        assert d.hold_s == 5.0
        assert d.tail_s == 1.0
        assert d.is_release is False

    def test_release_flag(self):
        d = InputSampleDef(note=60, velocity=127, hold_s=0.5, tail_s=2.0, is_release=True)
        assert d.is_release is True


# ---------------------------------------------------------------------------
# Sample dataclass
# ---------------------------------------------------------------------------

class TestSample:
    def test_basic_fields(self):
        audio = np.ones(100)
        s = Sample(midi_note=60, velocity=100, audio=audio, sr=44100)
        assert s.midi_note == 60
        assert s.velocity == 100
        assert s.sr == 44100
        assert s.audio.shape == (100,)

    def test_loop_points_default_none(self):
        s = make_sample()
        assert s.loop_start is None
        assert s.loop_end is None

    def test_loop_points_can_be_set(self):
        s = make_sample()
        s.loop_start = 500
        s.loop_end = 900
        assert s.loop_start == 500
        assert s.loop_end == 900


# ---------------------------------------------------------------------------
# InstrumentMeta
# ---------------------------------------------------------------------------

class TestInstrumentMeta:
    def test_required_fields_present(self):
        meta = make_meta()
        assert meta.instrument_name == "TestInstrument"
        assert meta.min_note == 21
        assert meta.max_note == 108

    def test_computed_dynamic_range_default_none(self):
        meta = make_meta()
        assert meta.computed_dynamic_range_db is None

    def test_no_recording_fields(self):
        """Recording fields were removed — verify they don't exist on InstrumentMeta."""
        meta = make_meta()
        assert not hasattr(meta, "velocity_layers")
        assert not hasattr(meta, "semitone_interval")
        assert not hasattr(meta, "hold_time")
        assert not hasattr(meta, "release_time")
        assert not hasattr(meta, "start_note")
        assert not hasattr(meta, "end_note")
        assert not hasattr(meta, "velocity_layers_out")

    def test_no_velocity_map(self):
        meta = make_meta()
        assert not hasattr(meta, "velocity_map")


# ---------------------------------------------------------------------------
# InstrumentData
# ---------------------------------------------------------------------------

class TestInstrumentData:
    def test_sustain_notes(self):
        data = make_instrument_data(n_notes=3, n_velocities=2)
        assert data.sustain_notes() == [60, 63, 66]

    def test_sustain_velocities(self):
        data = make_instrument_data(n_notes=3, n_velocities=2)
        assert data.sustain_velocities() == [32, 127]

    def test_get_sustain_found(self):
        data = make_instrument_data(n_notes=3, n_velocities=2)
        s = data.get_sustain(60, 32)
        assert s is not None
        assert s.midi_note == 60
        assert s.velocity == 32

    def test_get_sustain_not_found(self):
        data = make_instrument_data(n_notes=3, n_velocities=2)
        assert data.get_sustain(99, 99) is None

    def test_all_samples_combines(self):
        data = make_instrument_data(n_notes=2, n_velocities=2)
        data.release = [make_sample(60, 64)]
        assert len(data.all_samples()) == 5  # 2*2 sustain + 1 release

    def test_empty_release(self):
        data = make_instrument_data()
        assert data.release == []
        assert len(data.all_samples()) == len(data.sustain)


# ---------------------------------------------------------------------------
# ProcessingObject + PassthroughProcessor
# ---------------------------------------------------------------------------

class TestProcessingObject:
    def test_passthrough_returns_same_data(self):
        proc = PassthroughProcessor()
        data = make_instrument_data()
        result = proc.process(data)
        assert result is data

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ProcessingObject()  # type: ignore

    def test_custom_processor_subclass(self):
        class TagProcessor(ProcessingObject):
            def process(self, data: InstrumentData) -> InstrumentData:
                data.meta.instrument_name = "tagged"
                return data

        proc = TagProcessor()
        data = make_instrument_data()
        result = proc.process(data)
        assert result.meta.instrument_name == "tagged"

    def test_repr(self):
        assert "PassthroughProcessor" in repr(PassthroughProcessor())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_empty_pipeline(self):
        pipeline = Pipeline([])
        data = make_instrument_data()
        result = pipeline.run(data)
        assert result is data

    def test_single_processor(self):
        pipeline = Pipeline([PassthroughProcessor()])
        data = make_instrument_data()
        result = pipeline.run(data)
        assert result is data

    def test_chain_order(self):
        log = []

        class LogProcessor(ProcessingObject):
            def __init__(self, tag):
                self.tag = tag

            def process(self, data: InstrumentData) -> InstrumentData:
                log.append(self.tag)
                return data

        pipeline = Pipeline([LogProcessor("A"), LogProcessor("B"), LogProcessor("C")])
        pipeline.run(make_instrument_data())
        assert log == ["A", "B", "C"]

    def test_processor_can_modify_data(self):
        class DoubleAudioProcessor(ProcessingObject):
            def process(self, data: InstrumentData) -> InstrumentData:
                for s in data.sustain:
                    s.audio = s.audio * 2
                return data

        data = make_instrument_data(n_notes=1, n_velocities=1)
        data.sustain[0].audio = np.ones(100)
        Pipeline([DoubleAudioProcessor()]).run(data)
        assert np.all(data.sustain[0].audio == 2.0)

    def test_repr(self):
        p = Pipeline([PassthroughProcessor(), PassthroughProcessor()])
        assert "PassthroughProcessor" in repr(p)


# ---------------------------------------------------------------------------
# parse_input_meta
# ---------------------------------------------------------------------------

class TestParseInputMeta:
    def test_parse_minimal(self, tmp_path):
        write_input_toml(tmp_path, MINIMAL_INPUT_DEFS)
        defs, collapse = parse_input_meta(tmp_path)
        assert len(defs) == 1
        assert defs[0].note == 60
        assert defs[0].velocity == 64
        assert defs[0].hold_s == 1.0
        assert defs[0].tail_s == 0.5
        assert defs[0].is_release is False
        assert collapse is False

    def test_collapse_to_mono_true(self, tmp_path):
        write_input_toml(tmp_path, MINIMAL_INPUT_DEFS, collapse_to_mono=True)
        _, collapse = parse_input_meta(tmp_path)
        assert collapse is True

    def test_multiple_samples_in_order(self, tmp_path):
        defs = [
            {"note": 60, "velocity": 1,   "hold_s": 5.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 5.0, "tail_s": 0.5, "is_release": False},
            {"note": 60, "velocity": 1,   "hold_s": 0.3, "tail_s": 2.0, "is_release": True},
        ]
        write_input_toml(tmp_path, defs)
        result, _ = parse_input_meta(tmp_path)
        assert len(result) == 3
        assert result[0].velocity == 1
        assert result[1].velocity == 127
        assert result[2].is_release is True

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_input_meta(tmp_path / "nonexistent.toml")

    def test_dir_resolves_to_input_toml(self, tmp_path):
        write_input_toml(tmp_path, MINIMAL_INPUT_DEFS)
        defs, _ = parse_input_meta(tmp_path)
        assert len(defs) == 1

    def test_missing_note_field(self, tmp_path):
        p = tmp_path / "input.toml"
        p.write_text("[[samples]]\nvelocity=64\nhold_s=1.0\ntail_s=0.5\nis_release=false\n")
        with pytest.raises(MetadataError, match="note"):
            parse_input_meta(p)

    def test_note_out_of_range(self, tmp_path):
        p = tmp_path / "input.toml"
        p.write_text("[[samples]]\nnote=200\nvelocity=64\nhold_s=1.0\ntail_s=0.5\nis_release=false\n")
        with pytest.raises(MetadataError, match="note"):
            parse_input_meta(p)

    def test_velocity_out_of_range(self, tmp_path):
        p = tmp_path / "input.toml"
        p.write_text("[[samples]]\nnote=60\nvelocity=0\nhold_s=1.0\ntail_s=0.5\nis_release=false\n")
        with pytest.raises(MetadataError, match="velocity"):
            parse_input_meta(p)

    def test_hold_s_zero_rejected(self, tmp_path):
        p = tmp_path / "input.toml"
        p.write_text("[[samples]]\nnote=60\nvelocity=64\nhold_s=0.0\ntail_s=0.5\nis_release=false\n")
        with pytest.raises(MetadataError, match="hold_s"):
            parse_input_meta(p)

    def test_empty_samples_rejected(self, tmp_path):
        p = tmp_path / "input.toml"
        p.write_text("collapse_to_mono = false\n")
        with pytest.raises(MetadataError, match="samples"):
            parse_input_meta(p)


# ---------------------------------------------------------------------------
# parse_output_meta
# ---------------------------------------------------------------------------

MINIMAL_OUTPUT = ""  # all fields optional


class TestParseOutputMeta:
    def test_empty_file_uses_all_defaults(self, tmp_path):
        write_output_toml(tmp_path, "")
        meta = parse_output_meta(tmp_path)
        assert meta.min_note == 21
        assert meta.max_note == 108
        assert meta.note_percentage == 100.0
        assert meta.crossfade_percent == 0.0
        assert meta.normalize is True
        assert meta.normalize_mode == "lufs"
        assert meta.ampeg_release == 0.5

    def test_override_fields(self, tmp_path):
        content = "min_note=36\nmax_note=96\nnote_percentage=50.0\n"
        write_output_toml(tmp_path, content)
        meta = parse_output_meta(tmp_path)
        assert meta.min_note == 36
        assert meta.max_note == 96
        assert meta.note_percentage == 50.0

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_output_meta(tmp_path / "nonexistent.toml")

    def test_dir_resolves_to_output_toml(self, tmp_path):
        write_output_toml(tmp_path, "")
        meta = parse_output_meta(tmp_path)
        assert meta is not None

    def test_min_note_greater_than_max_note(self, tmp_path):
        write_output_toml(tmp_path, "min_note=80\nmax_note=40\n")
        with pytest.raises(MetadataError, match="min_note"):
            parse_output_meta(tmp_path)

    def test_note_percentage_out_of_range(self, tmp_path):
        write_output_toml(tmp_path, "note_percentage=0\n")
        with pytest.raises(MetadataError, match="note_percentage"):
            parse_output_meta(tmp_path)

    def test_normalize_mode_invalid(self, tmp_path):
        write_output_toml(tmp_path, 'normalize_mode="peak"\n')
        with pytest.raises(MetadataError, match="normalize_mode"):
            parse_output_meta(tmp_path)

    def test_normalize_mode_velocity(self, tmp_path):
        write_output_toml(tmp_path, 'normalize_mode="velocity"\n')
        meta = parse_output_meta(tmp_path)
        assert meta.normalize_mode == "velocity"

    def test_crossfade_out_of_range(self, tmp_path):
        write_output_toml(tmp_path, "crossfade_percent=150\n")
        with pytest.raises(MetadataError, match="crossfade_percent"):
            parse_output_meta(tmp_path)

    def test_wrong_type_raises(self, tmp_path):
        # normalize expects bool, not a string
        write_output_toml(tmp_path, 'normalize="yes"\n')
        with pytest.raises(MetadataError):
            parse_output_meta(tmp_path)

    def test_int_coerced_to_float(self, tmp_path):
        write_output_toml(tmp_path, "ampeg_release=1\n")
        meta = parse_output_meta(tmp_path)
        assert meta.ampeg_release == 1.0
        assert isinstance(meta.ampeg_release, float)

    def test_no_recording_fields_parsed(self, tmp_path):
        """output.toml has no concept of recording params — they're silently ignored."""
        write_output_toml(tmp_path, "velocity_layers=4\nhold_time=5.0\n")
        meta = parse_output_meta(tmp_path)
        assert not hasattr(meta, "velocity_layers")
        assert not hasattr(meta, "hold_time")
