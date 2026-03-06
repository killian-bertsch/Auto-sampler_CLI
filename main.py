#!/usr/bin/env python3
"""
main.py — Auto-Sampler CLI entry point.

Usage:
    python main.py <target_dir> [--output-dir DIR] [--instrument NAME]

target_dir should contain one or more instrument subfolders, each with:
    metadata.txt
    sustain.wav
    release.wav  (optional)
"""

import argparse
import sys
from pathlib import Path


def find_instruments(target_dir: Path) -> list[Path]:
    """Return subdirectories of target_dir that contain metadata.txt + sustain.wav."""
    instruments = []
    for subdir in sorted(target_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / "metadata.txt").exists() and (subdir / "sustain.wav").exists():
            instruments.append(subdir)
    return instruments


def process_instrument(instrument_dir: Path, output_dir: Path) -> bool:
    """
    Run the full processing pipeline for a single instrument directory.

    Returns True on success, False if the instrument should be skipped.
    """
    from src.metadata_parser import parse_metadata, MetadataError
    from src.pipeline import Pipeline
    from src.base import PassthroughProcessor  # placeholder until phases 2–5

    name = instrument_dir.name
    print(f"\n[{name}] Loading metadata...")

    try:
        meta = parse_metadata(instrument_dir / "metadata.txt")
    except (FileNotFoundError, MetadataError) as exc:
        print(f"  WARNING: Skipping '{name}' — {exc}")
        return False

    # Phase 2 will replace this stub with the real InputModule
    print(f"  instrument_name : {meta.instrument_name}")
    print(f"  velocity_layers : {meta.velocity_layers}")
    print(f"  semitone_interval: {meta.semitone_interval}")
    print(f"  note range       : {meta.start_note}–{meta.end_note}")
    print(f"  [stub] InputModule not yet implemented — skipping audio load")

    # Processor chain (phases 2–5 will populate this)
    # pipeline = Pipeline([TrimProcessor(), NormalizeProcessor(), LoopFinderProcessor()])
    # result = pipeline.run(instrument_data)
    # OutputModule(output_dir).write(result)

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
        print("Each instrument folder must contain metadata.txt and sustain.wav")
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
