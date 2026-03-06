"""
test_phase1.py — Unit tests for Phase 1: scaffolding, data structures, metadata parser, pipeline.

Run with:
    cd Auto-sampler_CLI
    python -m pytest tests/test_phase1.py -v
"""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from src.data import InstrumentData, InstrumentMeta, Sample
from src.base import ProcessingObject, PassthroughProcessor
from src.pipeline import Pipeline
from src.metadata_parser import MetadataError, parse_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_meta(**overrides) -> InstrumentMeta:
    """Return a minimal valid InstrumentMeta, with optional field overrides."""
    defaults = dict(
        velocity_layers=4,
        semitone_interval=3,
        hold_time=4.0,
        release_time=2.0,
        start_note=21,
        end_note=108,
        min_note=21,
        max_note=108,
        note_percentage=100.0,
        velocity_layers_out=4,
        crossfade_percent=20.0,
        normalize=True,
        normalize_mode="lufs",
        normalize_target_lufs=-18.0,
        peak_ceiling_db=-1.0,
        velocity_dynamic_range_db=40.0,
        instrument_name="TestInstrument",
        ampeg_release=0.5,
        loop_crossfade_ms=20.0,
        onset_threshold_db=-40.0,
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


def write_metadata(tmp_path: Path, content: str) -> Path:
    """Write a metadata.txt to a temp dir and return its path."""
    p = tmp_path / "metadata.txt"
    p.write_text(textwrap.dedent(content))
    return p


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
        assert meta.velocity_layers == 4

    def test_release_overrides_default_to_none(self):
        meta = make_meta()
        assert meta.release_hold_time is None
        assert meta.release_release_time is None

    def test_effective_release_hold_time_falls_back(self):
        meta = make_meta(hold_time=4.0, release_hold_time=None)
        assert meta.effective_release_hold_time() == 4.0

    def test_effective_release_hold_time_uses_override(self):
        meta = make_meta(hold_time=4.0, release_hold_time=6.0)
        assert meta.effective_release_hold_time() == 6.0

    def test_effective_release_release_time_falls_back(self):
        meta = make_meta(release_time=2.0, release_release_time=None)
        assert meta.effective_release_release_time() == 2.0

    def test_effective_release_release_time_uses_override(self):
        meta = make_meta(release_time=2.0, release_release_time=3.5)
        assert meta.effective_release_release_time() == 3.5

    def test_computed_dynamic_range_default_none(self):
        meta = make_meta()
        assert meta.computed_dynamic_range_db is None


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
        release = [make_sample(60, 64)]
        data.release = release
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
        """Processors must run in the order they are listed."""
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
# Metadata Parser
# ---------------------------------------------------------------------------

VALID_METADATA = """\
    velocity_layers = 4
    semitone_interval = 3
    hold_time = 4.0
    release_time = 2.0
    velocity_layers_out = 4
    instrument_name = TestPiano
"""


class TestMetadataParser:
    def test_parse_valid_minimal(self, tmp_path):
        p = write_metadata(tmp_path, VALID_METADATA)
        meta = parse_metadata(p)
        assert meta.velocity_layers == 4
        assert meta.semitone_interval == 3
        assert meta.hold_time == 4.0
        assert meta.release_time == 2.0
        assert meta.instrument_name == "TestPiano"

    def test_defaults_applied(self, tmp_path):
        p = write_metadata(tmp_path, VALID_METADATA)
        meta = parse_metadata(p)
        # Optional fields should have their defaults
        assert meta.start_note == 21
        assert meta.end_note == 108
        assert meta.min_note == 21
        assert meta.max_note == 108
        assert meta.note_percentage == 100.0
        assert meta.crossfade_percent == 0.0
        assert meta.normalize is True
        assert meta.normalize_mode == "lufs"
        assert meta.normalize_target_lufs == -18.0
        assert meta.peak_ceiling_db == -1.0
        assert meta.ampeg_release == 0.5
        assert meta.onset_threshold_db == -40.0

    def test_release_overrides_absent_gives_none(self, tmp_path):
        p = write_metadata(tmp_path, VALID_METADATA)
        meta = parse_metadata(p)
        assert meta.release_hold_time is None
        assert meta.release_release_time is None

    def test_release_overrides_parsed_when_present(self, tmp_path):
        content = VALID_METADATA + "\nrelease_hold_time = 6.0\nrelease_release_time = 3.5\n"
        p = write_metadata(tmp_path, content)
        meta = parse_metadata(p)
        assert meta.release_hold_time == 6.0
        assert meta.release_release_time == 3.5

    def test_comments_ignored(self, tmp_path):
        content = "# This is a comment\n" + VALID_METADATA + "\n# Another comment\n"
        p = write_metadata(tmp_path, content)
        meta = parse_metadata(p)
        assert meta.velocity_layers == 4

    def test_blank_lines_ignored(self, tmp_path):
        content = "\n\n" + VALID_METADATA + "\n\n"
        p = write_metadata(tmp_path, content)
        meta = parse_metadata(p)
        assert meta.velocity_layers == 4

    def test_case_insensitive_keys(self, tmp_path):
        content = VALID_METADATA.replace("velocity_layers = 4", "VELOCITY_LAYERS = 4")
        p = write_metadata(tmp_path, content)
        meta = parse_metadata(p)
        assert meta.velocity_layers == 4

    def test_unknown_keys_ignored(self, tmp_path):
        content = VALID_METADATA + "\nsome_future_field = 99\n"
        p = write_metadata(tmp_path, content)
        meta = parse_metadata(p)
        assert meta.velocity_layers == 4  # parse succeeded despite unknown key

    def test_bool_true(self, tmp_path):
        content = VALID_METADATA + "\nnormalize = true\n"
        p = write_metadata(tmp_path, content)
        assert parse_metadata(p).normalize is True

    def test_bool_false(self, tmp_path):
        content = VALID_METADATA + "\nnormalize = false\n"
        p = write_metadata(tmp_path, content)
        assert parse_metadata(p).normalize is False

    def test_bool_invalid(self, tmp_path):
        content = VALID_METADATA + "\nnormalize = yes\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="true/false"):
            parse_metadata(p)

    def test_missing_required_field(self, tmp_path):
        # Remove instrument_name (required)
        content = "\n".join(
            line for line in VALID_METADATA.splitlines()
            if "instrument_name" not in line
        )
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="instrument_name"):
            parse_metadata(p)

    def test_invalid_int(self, tmp_path):
        content = VALID_METADATA.replace("velocity_layers = 4", "velocity_layers = abc")
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError):
            parse_metadata(p)

    def test_invalid_float(self, tmp_path):
        content = VALID_METADATA.replace("hold_time = 4.0", "hold_time = notanumber")
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError):
            parse_metadata(p)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_metadata(tmp_path / "nonexistent.txt")

    def test_malformed_line(self, tmp_path):
        content = VALID_METADATA + "\nthis line has no equals sign\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="key = value"):
            parse_metadata(p)

    # --- Semantic validation ---

    def test_velocity_layers_out_exceeds_recorded(self, tmp_path):
        content = VALID_METADATA.replace("velocity_layers_out = 4", "velocity_layers_out = 8")
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="velocity_layers_out"):
            parse_metadata(p)

    def test_min_note_greater_than_max_note(self, tmp_path):
        content = VALID_METADATA + "\nmin_note = 80\nmax_note = 40\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="min_note"):
            parse_metadata(p)

    def test_note_percentage_out_of_range(self, tmp_path):
        content = VALID_METADATA + "\nnote_percentage = 0\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="note_percentage"):
            parse_metadata(p)

    def test_normalize_mode_invalid(self, tmp_path):
        content = VALID_METADATA + "\nnormalize_mode = peak\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="normalize_mode"):
            parse_metadata(p)

    def test_normalize_mode_velocity(self, tmp_path):
        content = VALID_METADATA + "\nnormalize_mode = velocity\n"
        p = write_metadata(tmp_path, content)
        assert parse_metadata(p).normalize_mode == "velocity"

    def test_release_hold_time_invalid(self, tmp_path):
        content = VALID_METADATA + "\nrelease_hold_time = -1.0\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="release_hold_time"):
            parse_metadata(p)

    def test_crossfade_percent_out_of_range(self, tmp_path):
        content = VALID_METADATA + "\ncrossfade_percent = 150\n"
        p = write_metadata(tmp_path, content)
        with pytest.raises(MetadataError, match="crossfade_percent"):
            parse_metadata(p)
