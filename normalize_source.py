"""
Normalize all sustain.flac / release.flac files in input/source to -6 dBFS peak.

Strategy (streaming, two-pass — no full file held in RAM):
  Pass 1: read file in blocks, track global absolute peak
  Pass 2: apply a single linear gain, write to a temp file as 24-bit FLAC
  Then atomically replace the original.

Usage:
    python normalize_source.py [--source-dir input/source] [--target-db -6.0] [--dry-run]
"""

import argparse
import math
import os
import sys
import tempfile
import numpy as np
import soundfile as sf

TARGET_DB_DEFAULT = -6.0
BLOCK_SIZE = 65536  # frames per block (~1.5 s @ 44100)


def find_peak_db(path: str) -> float:
    """Return the true peak of a FLAC file in dBFS (streaming, no full load)."""
    peak_linear = 0.0
    with sf.SoundFile(path) as f:
        for block in f.blocks(blocksize=BLOCK_SIZE, dtype="float64"):
            block_peak = np.max(np.abs(block))
            if block_peak > peak_linear:
                peak_linear = block_peak
    if peak_linear == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak_linear)


def apply_gain(src_path: str, dst_path: str, gain_linear: float) -> None:
    """Read src, multiply every sample by gain_linear, write dst as 24-bit FLAC."""
    with sf.SoundFile(src_path) as reader:
        sr = reader.samplerate
        ch = reader.channels
        with sf.SoundFile(
            dst_path,
            mode="w",
            samplerate=sr,
            channels=ch,
            format="FLAC",
            subtype="PCM_24",
        ) as writer:
            for block in reader.blocks(blocksize=BLOCK_SIZE, dtype="float64"):
                writer.write(block * gain_linear)


def normalize_file(path: str, target_db: float, dry_run: bool) -> None:
    peak_db = find_peak_db(path)
    if peak_db == -math.inf:
        print(f"  SKIP  {path}  (silent file)")
        return

    gain_db = target_db - peak_db
    gain_linear = 10.0 ** (gain_db / 20.0)

    rel = os.path.relpath(path)
    print(f"  {rel}")
    print(f"    peak: {peak_db:+.2f} dBFS  →  gain: {gain_db:+.2f} dB", end="")

    if dry_run:
        print("  [DRY RUN — no write]")
        return

    # Write to a temp file in the same directory so rename is atomic on same fs
    dir_ = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".flac.tmp")
    os.close(fd)
    try:
        apply_gain(path, tmp_path, gain_linear)
        os.replace(tmp_path, path)
        print("  ✓")
    except Exception:
        os.unlink(tmp_path)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize source FLAC files to target peak dBFS.")
    parser.add_argument("--source-dir", default="input/source")
    parser.add_argument("--target-db", type=float, default=TARGET_DB_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="Print what would happen, no writes")
    args = parser.parse_args()

    source_dir = args.source_dir
    if not os.path.isdir(source_dir):
        sys.exit(f"Source directory not found: {source_dir}")

    files_to_process: list[str] = []
    for entry in sorted(os.scandir(source_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        for fname in ("sustain.flac", "release.flac"):
            fpath = os.path.join(entry.path, fname)
            if os.path.isfile(fpath):
                files_to_process.append(fpath)

    if not files_to_process:
        sys.exit("No sustain.flac / release.flac files found.")

    print(f"Target: {args.target_db:+.1f} dBFS peak")
    print(f"Files:  {len(files_to_process)}")
    if args.dry_run:
        print("Mode:   DRY RUN\n")
    else:
        print()

    errors = []
    for i, path in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}]")
        try:
            normalize_file(path, args.target_db, args.dry_run)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append(path)

    print(f"\nDone. {len(files_to_process) - len(errors)} succeeded, {len(errors)} failed.")
    if errors:
        for p in errors:
            print(f"  FAILED: {p}")
        sys.exit(1)


if __name__ == "__main__":
    main()
