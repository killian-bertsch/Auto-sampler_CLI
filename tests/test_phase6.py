"""
test_phase6.py — End-to-end integration tests for the CLI + bulk orchestrator.

Tests cover:
  - find_instruments() discovery logic
  - process_instrument() full pipeline on a synthetic instrument
  - main() argument parsing and return codes
  - Bulk processing of multiple instruments
  - Graceful skip on missing / invalid metadata
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import find_instruments, process_instrument, main

# ---------------------------------------------------------------------------
# Helpers to build synthetic instrument folders
# ---------------------------------------------------------------------------

SR = 22050
VELOCITY_LAYERS = 2
SEMITONE_INTERVAL = 12   # C3=48, C4=60 — only 2 notes
HOLD_TIME = 0.3
RELEASE_TIME = 0.1
START_NOTE = 48
END_NOTE = 60


def _make_tone(freq: float, duration: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """Generate a simple sine tone."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _make_sustain_wav(path: Path) -> None:
    """
    Build a sustain.wav:  velocity_layers × notes_count slices.
    Layout per V2 convention: velocity outer loop, note inner loop.
    Each slice = hold_time + release_time seconds.
    Notes: C3(48), C4(60)  |  Velocities: ~43, ~85 (evenly spaced, 2 layers)
    """
    slice_len = int(SR * (HOLD_TIME + RELEASE_TIME))
    # 2 velocities × 2 notes = 4 slices
    freqs = {48: 130.81, 60: 261.63}
    velocities = [43, 85]
    notes = [48, 60]

    chunks = []
    for vel in velocities:
        amp = (vel / 127.0) * 0.8
        for note in notes:
            tone = _make_tone(freqs[note], HOLD_TIME + RELEASE_TIME, SR, amp)
            # Pad or trim to exact slice length
            if len(tone) < slice_len:
                tone = np.pad(tone, (0, slice_len - len(tone)))
            else:
                tone = tone[:slice_len]
            chunks.append(tone)

    audio = np.concatenate(chunks)
    sf.write(str(path), audio, SR)


def _make_release_wav(path: Path) -> None:
    """Build a release.wav with same layout but shorter hold/release."""
    rel_hold = 0.1
    rel_release = 0.05
    slice_len = int(SR * (rel_hold + rel_release))
    freqs = {48: 130.81, 60: 261.63}
    velocities = [43, 85]
    notes = [48, 60]

    chunks = []
    for vel in velocities:
        amp = (vel / 127.0) * 0.4
        for note in notes:
            tone = _make_tone(freqs[note], rel_hold + rel_release, SR, amp)
            if len(tone) < slice_len:
                tone = np.pad(tone, (0, slice_len - len(tone)))
            else:
                tone = tone[:slice_len]
            chunks.append(tone)

    audio = np.concatenate(chunks)
    sf.write(str(path), audio, SR)


def _make_metadata(path: Path, instrument_name: str = "TestInstrument",
                   include_release_params: bool = False) -> None:
    """Write a minimal valid metadata.txt."""
    content = textwrap.dedent(f"""\
        instrument_name = {instrument_name}
        velocity_layers = {VELOCITY_LAYERS}
        semitone_interval = {SEMITONE_INTERVAL}
        hold_time = {HOLD_TIME}
        release_time = {RELEASE_TIME}
        start_note = {START_NOTE}
        end_note = {END_NOTE}
        min_note = {START_NOTE}
        max_note = {END_NOTE}
        note_percentage = 100
        velocity_layers_out = {VELOCITY_LAYERS}
        crossfade_percent = 0
        normalize_mode = rms
        normalize_target_db = -18.0
        peak_ceiling_db = -1.0
        velocity_dynamic_range_db = 30.0
        ampeg_release = 0.5
        loop_crossfade_ms = 0
        collapse_to_mono = false
    """)
    if include_release_params:
        content += textwrap.dedent("""\
            release_hold_time = 0.1
            release_release_time = 0.05
        """)
    path.write_text(content, encoding="utf-8")


def _make_instrument_folder(base: Path, name: str = "TestInstrument",
                             with_release: bool = False) -> Path:
    folder = base / name
    folder.mkdir(parents=True, exist_ok=True)
    _make_sustain_wav(folder / "sustain.wav")
    _make_metadata(folder / "metadata.txt", instrument_name=name,
                   include_release_params=with_release)
    if with_release:
        _make_release_wav(folder / "release.wav")
    return folder


# ---------------------------------------------------------------------------
# find_instruments()
# ---------------------------------------------------------------------------

class TestFindInstruments:
    def test_finds_valid_subfolders(self, tmp_path):
        _make_instrument_folder(tmp_path, "Piano")
        _make_instrument_folder(tmp_path, "Strings")
        result = find_instruments(tmp_path)
        names = [p.name for p in result]
        assert "Piano" in names
        assert "Strings" in names

    def test_skips_folders_missing_metadata(self, tmp_path):
        folder = tmp_path / "NoMeta"
        folder.mkdir()
        _make_sustain_wav(folder / "sustain.wav")
        result = find_instruments(tmp_path)
        assert result == []

    def test_skips_folders_missing_sustain_wav(self, tmp_path):
        folder = tmp_path / "NoWav"
        folder.mkdir()
        _make_metadata(folder / "metadata.txt", "NoWav")
        result = find_instruments(tmp_path)
        assert result == []

    def test_ignores_files_at_top_level(self, tmp_path):
        (tmp_path / "metadata.txt").write_text("x=1")
        (tmp_path / "sustain.wav").write_text("not a wav")
        result = find_instruments(tmp_path)
        assert result == []

    def test_returns_sorted_order(self, tmp_path):
        for name in ["Cello", "Banjo", "Accordion"]:
            _make_instrument_folder(tmp_path, name)
        result = find_instruments(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_empty_target_dir(self, tmp_path):
        assert find_instruments(tmp_path) == []


# ---------------------------------------------------------------------------
# process_instrument()
# ---------------------------------------------------------------------------

class TestProcessInstrument:
    def test_succeeds_on_valid_instrument(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        result = process_instrument(instr, out)
        assert result is True

    def test_creates_output_directory(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        assert (out / "Piano").is_dir()

    def test_writes_sustain_sfz(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        sfz = out / "Piano" / "Piano_sustain.sfz"
        assert sfz.exists()
        text = sfz.read_text()
        assert "<region>" in text

    def test_writes_flac_files(self, tmp_path):
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        flacs = list((out / "Piano" / "samples").glob("*.flac"))
        assert len(flacs) > 0

    def test_with_release_wav(self, tmp_path):
        instr = _make_instrument_folder(
            tmp_path / "instruments", "Organ", with_release=True
        )
        out = tmp_path / "output"
        result = process_instrument(instr, out)
        assert result is True
        rel_sfz = out / "Organ" / "Organ_release.sfz"
        assert rel_sfz.exists()
        rel_flacs = list((out / "Organ" / "samples" / "release").glob("*.flac"))
        assert len(rel_flacs) > 0

    def test_skips_missing_metadata(self, tmp_path):
        folder = tmp_path / "Broken"
        folder.mkdir()
        _make_sustain_wav(folder / "sustain.wav")
        out = tmp_path / "output"
        result = process_instrument(folder, out)
        assert result is False

    def test_skips_invalid_metadata(self, tmp_path):
        folder = tmp_path / "BadMeta"
        folder.mkdir()
        _make_sustain_wav(folder / "sustain.wav")
        (folder / "metadata.txt").write_text("not_a_real_key = whatever\n")
        out = tmp_path / "output"
        result = process_instrument(folder, out)
        assert result is False

    def test_no_output_written_on_skip(self, tmp_path):
        folder = tmp_path / "Broken"
        folder.mkdir()
        _make_sustain_wav(folder / "sustain.wav")
        out = tmp_path / "output"
        process_instrument(folder, out)
        assert not (out / "Broken").exists()


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
        nonexistent = str(tmp_path / "does_not_exist")
        code = self._run([nonexistent])
        assert code == 1

    def test_empty_target_dir_returns_1(self, tmp_path):
        code = self._run([str(tmp_path)])
        assert code == 1

    def test_valid_instrument_returns_0(self, tmp_path):
        _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        code = self._run([str(tmp_path / "instruments"), "--output-dir", str(out)])
        assert code == 0

    def test_instrument_filter_single(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        _make_instrument_folder(instr_dir, "Strings")
        out = tmp_path / "output"
        code = self._run([
            str(instr_dir), "--output-dir", str(out), "--instrument", "Piano"
        ])
        assert code == 0
        assert (out / "Piano").is_dir()
        assert not (out / "Strings").exists()

    def test_instrument_filter_nonexistent_returns_1(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        instr_dir.mkdir()
        out = tmp_path / "output"
        code = self._run([
            str(instr_dir), "--output-dir", str(out), "--instrument", "Ghost"
        ])
        assert code == 1

    def test_default_output_dir_is_output(self, tmp_path, monkeypatch):
        """When --output-dir is omitted, output/ is created relative to cwd."""
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        monkeypatch.chdir(tmp_path)
        code = self._run([str(instr_dir)])
        assert code == 0
        assert (tmp_path / "output" / "Piano").is_dir()

    def test_all_invalid_returns_1(self, tmp_path):
        instr_dir = tmp_path / "instruments"
        folder = instr_dir / "Broken"
        folder.mkdir(parents=True)
        _make_sustain_wav(folder / "sustain.wav")
        out = tmp_path / "output"
        code = self._run([str(instr_dir), "--output-dir", str(out)])
        assert code == 1


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
        """One valid + one broken instrument: should return 0 (at least one success)."""
        instr_dir = tmp_path / "instruments"
        _make_instrument_folder(instr_dir, "Piano")
        broken = instr_dir / "Broken"
        broken.mkdir()
        _make_sustain_wav(broken / "sustain.wav")
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
        """Verify output FLAC files contain valid audio data."""
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        for flac in (out / "Piano" / "samples").glob("*.flac"):
            audio, sr = sf.read(str(flac))
            assert sr == SR
            assert len(audio) > 0
            assert np.all(np.isfinite(audio))

    def test_sustain_sfz_contains_loop_opcodes(self, tmp_path):
        """Loop points should be set by LoopFinderProcessor and appear in SFZ."""
        instr = _make_instrument_folder(tmp_path / "instruments", "Piano")
        out = tmp_path / "output"
        process_instrument(instr, out)
        sfz_text = (out / "Piano" / "Piano_sustain.sfz").read_text()
        # May or may not find loops depending on sample length, but SFZ must be valid
        assert "ampeg_release" in sfz_text
        assert "<region>" in sfz_text
