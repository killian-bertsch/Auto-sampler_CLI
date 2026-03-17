#!/usr/bin/env python3
"""
main.py — Auto-Sampler CLI entry point.

Usage:
    python main.py <target_dir> [--output-dir DIR] [--instrument NAME]

target_dir should contain one or more instrument subfolders, each with:
    metadata.txt / metadata.toml
    sustain.wav or sustain.flac
    release.wav or release.flac  (optional)
"""

import argparse
import sys
import traceback
from pathlib import Path


def find_instruments(target_dir: Path) -> list[Path]:
    """Return subdirectories of target_dir that contain metadata.toml (or legacy metadata.txt) + sustain.wav/.flac."""
    instruments = []
    for subdir in sorted(target_dir.iterdir()):
        if not subdir.is_dir():
            continue
        has_meta = (subdir / "metadata.toml").exists() or (subdir / "metadata.txt").exists()
        has_sustain = (subdir / "sustain.wav").exists() or (subdir / "sustain.flac").exists()
        if has_meta and has_sustain:
            instruments.append(subdir)
    return instruments


def process_instrument(instrument_dir: Path, output_dir: Path) -> bool:
    """
    Run the full processing pipeline for a single instrument directory.

    Returns True on success, False if the instrument should be skipped.
    """
    from src.metadata_parser import MetadataError
    from src.input_module import load_instrument
    from src.pipeline import Pipeline
    from src.processors import TrimProcessor, NormalizeProcessor, LoopFinderProcessor, TransientShaperProcessor
    from src.output_module import write_instrument

    name = instrument_dir.name
    print(f"\n[{name}] Loading...")

    # ── Load ──────────────────────────────────────────────────────────────────
    try:
        data = load_instrument(instrument_dir)
    except (FileNotFoundError, MetadataError) as exc:
        print(f"  WARNING: Skipping '{name}' — {exc}")
        return False
    except Exception as exc:
        print(f"  WARNING: Skipping '{name}' — unexpected load error: {exc}")
        traceback.print_exc()
        return False

    n_sus = len(data.sustain)
    n_rel = len(data.release)
    print(f"  {n_sus} sustain sample(s), {n_rel} release sample(s)")

    # ── Process ───────────────────────────────────────────────────────────────
    try:
        pipeline = Pipeline([
            TrimProcessor(),
            NormalizeProcessor(),
            TransientShaperProcessor(),
            LoopFinderProcessor(),
        ])
        data = pipeline.run(data)
    except Exception as exc:
        print(f"  WARNING: Skipping '{name}' — processing error: {exc}")
        traceback.print_exc()
        return False

    # ── Write output ──────────────────────────────────────────────────────────
    try:
        result = write_instrument(data, output_dir)
    except Exception as exc:
        print(f"  WARNING: Skipping '{name}' — output error: {exc}")
        traceback.print_exc()
        return False

    print(f"  {result['files_written']} FLAC file(s) written")
    print(f"  {result['sustain_regions']} sustain region(s), "
          f"{result['release_regions']} release region(s)")
    print(f"  Output: {result['output_dir']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-Sampler CLI — bulk keyboard multisampler processor"
    )
    parser.add_argument(
        "target_dir",
        type=Path,
        help="Directory containing instrument subfolders (each with metadata.txt + sustain.wav)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for processed output (default: ./output/)"
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Process only the named instrument subfolder (optional)"
    )

    args = parser.parse_args()

    target_dir: Path = args.target_dir
    output_dir: Path = args.output_dir

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"ERROR: target_dir does not exist or is not a directory: {target_dir}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.instrument:
        instrument_dirs = [target_dir / args.instrument]
        if not instrument_dirs[0].exists():
            print(f"ERROR: Instrument folder not found: {instrument_dirs[0]}")
            return 1
    else:
        instrument_dirs = find_instruments(target_dir)

    if not instrument_dirs:
        print(f"No valid instrument folders found in: {target_dir}")
        print("Each instrument folder must contain metadata.toml (or metadata.txt) and sustain.wav or sustain.flac")
        return 1

    print(f"Found {len(instrument_dirs)} instrument(s) to process.")

    success_count = 0
    for instr_dir in instrument_dirs:
        if process_instrument(instr_dir, output_dir):
            success_count += 1

    print(f"\nDone. {success_count}/{len(instrument_dirs)} instrument(s) processed successfully.")
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
