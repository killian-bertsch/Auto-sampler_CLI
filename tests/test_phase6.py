"""
test_phase6.py — End-to-end integration tests for the CLI + bulk orchestrator.

Tests cover:
  - find_instruments() discovery logic
  - process_instrument() full pipeline on a synthetic instrument
  - main() argument parsing and return codes
  - Bulk processing of multiple instruments
  - Graceful skip on missing / invalid files
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import find_instruments, process_instrument, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050

# Two notes (C3=48, C4=60), two velocities each
_DEFS = [
    {"note": 48, "velocity": 43,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
    {"note": 48, "velocity": 85,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
    {"note": 60, "velocity": 43,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
    {"note": 60, "velocity": 85,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
]

_DEFS_WITH_RELEASE = _DEFS + [
    {"note": 48, "velocity": 43,  "hold_s": 0.1, "tail_s": 0.2, "is_release": True},
    {"note": 48, "velocity": 85,  "hold_s": 0.1, "tail_s": 0.2, "is_release": True},
    {"note": 60, "velocity": 43,  "hold_s": 0.1, "tail_s": 0.2, "is_release": True},
    {"note": 60, "velocity": 85,  "hold_s": 0.1, "tail_s": 0.2, "is_release": True},
]


def _make_samples_flac(folder: Path, defs: List[dict]) -> None:
    total_s = sum(d["hold_s"] + d["tail_s"] for d in defs)
    total_frames = int(round(total_s * SR))
    audio = np.zeros(total_frames, dtype=np.float32)

    offset_s = 0.0
    for d in defs:
        start = int(round(offset_s * SR))
        hold_f = int(round(d["hold_s"] * SR))
        tail_f = int(round(d["tail_s"] * SR))
        freq = 440.0 * 2 ** ((d["note"] - 69) / 12.0)
        amp = (d["velocity"] / 127.0) * 0.5
        t = np.arange(hold_f + tail_f) / SR
        tone = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        end = min(start + len(tone), total_frames)
        audio[start:end] = tone[:end - start]
        offset_s += d["hold_s"] + d["tail_s"]

    sf.write(str(folder / "samples.flac"), audio, SR)


def _make_input_toml(folder: Path, defs: List[dict]) -> None:
    entries = []
    for d in defs:
        is_rel = "true" if d.get("is_release", False) else "false"
        entries.append(
            f'  {{note={d["note"]}, velocity={d["velocity"]}, '
            f'hold_s={d["hold_s"]}, tail_s={d["tail_s"]}, is_release={is_rel}}}'
        )
    content = (
        "collapse_to_mono = false\n"
        "samples = [\n"
        + ",\n".join(entries) + "\n"
        "]\n"
    )
    (folder / "input.toml").write_text(content)


def _make_output_toml(folder: Path, instrument_name: str = "") -> None:
    content = (
        f'instrument_name = "{instrument_name}"\n'
        "min_note = 48\n"
        "max_note = 60\n"
        "note_percentage = 100\n"
        "crossfade_percent = 0\n"
        "normalize = false\n"
        'normalize_mode = "rms"\n'
        "normalize_target_lufs = -18.0\n"
        "peak_ceiling_db = -1.0\n"
        "velocity_dynamic_range_db = 30.0\n"
        "ampeg_release = 0.5\n"
        "loop_crossfade_ms = 0\n"
    )
    (folder / "output.toml").write_text(content)


def _make_instrument_folder(
    base: Path, name: str = "TestInstrument", with_release: bool = False
) -> Path:
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    defs = _DEFS_WITH_RELEASE if with_release else _DEFS
    _make_samples_flac(folder, defs)
    _make_input_toml(folder, defs)
    _make_output_toml(folder, instrument_name=name)
    return folder


# ---------------------------------------------------------------------------
# find_instruments()
# ---------------------------------------------------------------------------

class TestFindInstruments:
    def test_finds_valid_subfolders(self, tmp_path):
        _make_instrument_folder(tmp_path, "Piano")
        _make_instrument_folder(tmp_path, "Strings")
        names = [p.name for p in find_instruments(tmp_path)]
        assert "Piano" in names
        assert "Strings" in names

    def test_skips_folders_missing_input_toml(self, tmp_path):
        folder = tmp_path / "NoInput"
        folder.mkdir()
        _make_samples_flac(folder, _DEFS)
        _make_output_toml(folder)
        assert find_instruments(tmp_path) == []

    def test_skips_folders_missing_output_toml(self, tmp_path):
        folder = tmp_path / "NoOutput"
        folder.mkdir()
        _make_samples_flac(folder, _DEFS)
        _make_input_toml(folder, _DEFS)
        assert find_instruments(tmp_path) == []

    def test_skips_folders_missing_samples_flac(self, tmp_path):
        folder = tmp_path / "NoAudio"
        folder.mkdir()
        _make_input_toml(folder, _DEFS)
        _make_output_toml(folder)
        assert find_instruments(tmp_path) == []

    def test_ignores_files_at_top_level(self, tmp_path):
        (tmp_path / "input.toml").write_text("x=1")
        assert find_instruments(tmp_path) == []

    def test_returns_sorted_order(self, tmp_path):
        for name in ["Cello", "Banjo", "Accordion"]:
            _make_instrument_folder(tmp_path, name)
        names = [p.name for p in find_instruments(tmp_path)]
        assert names == sorted(names)

    def test_empty_target_dir(self, tmp_path):
        assert find_instruments(tmp_path) == []


# ---------------------------------------------------------------------------
# process_instrument()
# ---------------------------------------------------------------------------

class TestProcessInstrument:
    def test_succeeds_on_valid_instrument(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        assert process_instrument(instr, tmp_path / "output") is True

    def test_creates_output_directory(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        process_instrument(instr, tmp_path / "output")
        assert (tmp_path / "output" / "Piano").is_dir()

    def test_writes_sustain_sfz(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        process_instrument(instr, tmp_path / "output")
        sfz = tmp_path / "output" / "Piano" / "Piano_sustain.sfz"
        assert sfz.exists()
        assert "<region>" in sfz.read_text()

    def test_writes_flac_files(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        process_instrument(instr, tmp_path / "output")
        flacs = list((tmp_path / "output" / "Piano" / "Piano-sustain").glob("*.flac"))
        assert len(flacs) > 0

    def test_with_release_samples(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Organ", with_release=True)
        out = tmp_path / "output"
        assert process_instrument(instr, out) is True
        assert (out / "Organ" / "Organ_release.sfz").exists()
        rel_flacs = list((out / "Organ" / "Organ-release").glob("*.flac"))
        assert len(rel_flacs) > 0

    def test_skips_missing_input_toml(self, tmp_path):
        folder = tmp_path / "Broken"
        folder.mkdir()
        _make_samples_flac(folder, _DEFS)
        _make_output_toml(folder)
        assert process_instrument(folder, tmp_path / "output") is False

    def test_skips_missing_samples_flac(self, tmp_path):
        folder = tmp_path / "NoAudio"
        folder.mkdir()
        _make_input_toml(folder, _DEFS)
        _make_output_toml(folder)
        assert process_instrument(folder, tmp_path / "output") is False

    def test_skips_invalid_input_toml(self, tmp_path):
        folder = tmp_path / "BadInput"
        folder.mkdir()
        _make_samples_flac(folder, _DEFS)
        _make_output_toml(folder)
        (folder / "input.toml").write_text("this is not valid toml :\n")
        assert process_instrument(folder, tmp_path / "output") is False

    def test_no_output_written_on_skip(self, tmp_path):
        folder = tmp_path / "Broken"
        folder.mkdir()
        _make_samples_flac(folder, _DEFS)
        _make_output_toml(folder)
        out = tmp_path / "output"
        process_instrument(folder, out)
        assert not (out / "Broken").exists()

    def test_non_uniform_velocities_all_written(self, tmp_path):
        """Each note's velocity layers are all included in output."""
        defs = [
            {"note": 48, "velocity": 1,   "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
            {"note": 48, "velocity": 40,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
            {"note": 48, "velocity": 80,  "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
            {"note": 48, "velocity": 127, "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
            {"note": 60, "velocity": 1,   "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
            {"note": 60, "velocity": 127, "hold_s": 0.3, "tail_s": 0.1, "is_release": False},
        ]
        folder = tmp_path / "NonUniform"
        folder.mkdir()
        _make_samples_flac(folder, defs)
        _make_input_toml(folder, defs)
        _make_output_toml(folder, "NonUniform")
        out = tmp_path / "output"
        assert process_instrument(folder, out) is True
        flacs = list((out / "NonUniform" / "NonUniform-sustain").glob("*.flac"))
        assert len(flacs) == 6  # 4 + 2


# ---------------------------------------------------------------------------
# main() — argument parsing and return codes
# ---------------------------------------------------------------------------

class TestMain:
    def _run(self, argv: list[str]) -> int:
        old = sys.argv
        sys.argv = ["main.py"] + argv
        try:
            return main()
        finally:
            sys.argv = old

    def test_missing_target_dir_returns_1(self, tmp_path):
        assert self._run([str(tmp_path / "does_not_exist")]) == 1

    def test_empty_target_dir_returns_1(self, tmp_path):
        assert self._run([str(tmp_path)]) == 1

    def test_valid_instrument_returns_0(self, tmp_path):
        _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        assert self._run([str(tmp_path / "instruments"), "--output-dir", str(out)]) == 0

    def test_instrument_filter_single(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        _make_instrument_folder(instr_dir, "Strings")
        out = tmp_path / "output"
        code = self._run([str(instr_dir), "--output-dir", str(out), "--instrument", "Piano"])
        assert code == 0
        assert (out / "Piano").is_dir()
        assert not (out / "Strings").exists()

    def test_instrument_filter_nonexistent_returns_1(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        instr_dir.mkdir()
        assert self._run([str(instr_dir), "--output-dir", str(tmp_path / "out"), "--instrument", "Ghost"]) == 1

    def test_default_output_dir_is_output(self, tmp_path, monkeypatch):
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        monkeypatch.chdir(tmp_path)
        assert self._run([str(instr_dir)]) == 0
        assert (tmp_path / "output" / "Piano").is_dir()

    def test_all_invalid_returns_1(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        broken = instr_dir / "Broken"
        broken.mkdir(parents=True)
        _make_samples_flac(broken, _DEFS)
        _make_output_toml(broken)
        assert self._run([str(instr_dir), "--output-dir", str(tmp_path / "out")]) == 1


# ---------------------------------------------------------------------------
# Bulk processing
# ---------------------------------------------------------------------------

class TestBulkProcessing:
    def test_processes_multiple_instruments(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        names = ["Piano", "Strings", "Organ"]
        for name in names:
            _make_instrument_folder(instr_dir, name)
        out = tmp_path / "output"

        old = sys.argv
        sys.argv = ["main.py", str(instr_dir), "--output-dir", str(out)]
        try:
            code = main()
        finally:
            sys.argv = old

        assert code == 0
        for name in names:
            assert (out / name / f"{name}_sustain.sfz").exists()

    def test_partial_success_returns_0(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        broken = instr_dir / "Broken"
        broken.mkdir()
        _make_samples_flac(broken, _DEFS)
        _make_output_toml(broken)
        out = tmp_path / "output"

        old = sys.argv
        sys.argv = ["main.py", str(instr_dir), "--output-dir", str(out)]
        try:
            code = main()
        finally:
            sys.argv = old

        assert code == 0
        assert (out / "Piano").is_dir()
        assert not (out / "Broken").exists()

    def test_flac_files_are_readable(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        for flac in (out / "Piano" / "Piano-sustain").glob("*.flac"):
            audio, sr = sf.read(str(flac))
            assert sr == SR
            assert len(audio) > 0
            assert np.all(np.isfinite(audio))

    def test_sustain_sfz_is_valid(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        sfz_text = (out / "Piano" / "Piano_sustain.sfz").read_text()
        assert "ampeg_release" in sfz_text
        assert "<region>" in sfz_text
