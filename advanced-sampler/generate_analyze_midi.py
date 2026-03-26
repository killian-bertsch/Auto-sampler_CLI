#!/usr/bin/env python3
"""
generate_analyze_midi.py — Generate the one-time analysis MIDI file.

Creates analyze_instrument.mid:
  • All 88 piano notes  A0 (21) – C8 (108)
  • All 127 velocities  1 – 127  per note
  • Configurable hold and gap times (defaults: 100 ms hold, 500 ms gap)

The gap must be long enough for your instrument's notes to mostly decay before
the next event starts — otherwise bleed contaminates the spectral analysis.

Feed this MIDI to your target instrument, export the audio to a single .flac, then run:
    python advanced_sampler.py <rendered.flac> --hold-ms <H> --gap-ms <G>
    (pass the same --hold-ms / --gap-ms values you used here)
"""

import argparse
import sys

try:
    from midiutil import MIDIFile
except ImportError:
    print("midiutil not found.  Run: pip install midiutil")
    sys.exit(1)

BPM        = 120.0          # fixed; timing is driven by hold-ms / gap-ms
PIANO_LOW  = 21             # A0
PIANO_HIGH = 108            # C8
NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def midi_note_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"


def ms_to_beats(ms: float, bpm: float = BPM) -> float:
    return (ms / 1000.0) * (bpm / 60.0)


def generate(hold_ms: float, gap_ms: float,
             start_note: int, end_note: int,
             output_path: str):
    notes      = list(range(start_note, end_note + 1))
    velocities = list(range(1, 128))
    total      = len(notes) * len(velocities)

    hold_beats    = ms_to_beats(hold_ms)
    gap_beats     = ms_to_beats(gap_ms)
    spacing_beats = hold_beats + gap_beats

    event_s   = (hold_ms + gap_ms) / 1000.0
    total_s   = total * event_s
    total_min = total_s / 60.0

    print("Generating analyze MIDI:")
    print(f"  Notes       : {start_note} ({midi_note_to_name(start_note)}) "
          f"– {end_note} ({midi_note_to_name(end_note)})  ({len(notes)} notes)")
    print(f"  Velocities  : 1 – 127  ({len(velocities)} per note)")
    print(f"  Total events: {total:,}")
    print(f"  Hold        : {hold_ms:.0f} ms")
    print(f"  Gap         : {gap_ms:.0f} ms  (silence after note-off)")
    print(f"  Per event   : {event_s*1000:.0f} ms")
    print(f"  Total length: {total_s:.0f} s  ({total_min:.1f} min)")

    if gap_ms < 200:
        print(f"\n  WARNING: gap={gap_ms:.0f} ms may be too short for instruments with "
              f"noticeable decay.\n  If you see inconsistent switch detection, re-generate "
              f"with a larger --gap-ms.")

    midi = MIDIFile(1)
    midi.addTempo(0, 0, BPM)
    midi.addTrackName(0, 0, "AnalyzeInstrument")

    beat = 0.0
    for note in notes:
        for vel in velocities:
            midi.addNote(track=0, channel=0, pitch=note,
                         time=beat, duration=hold_beats, volume=vel)
            beat += spacing_beats

    with open(output_path, "wb") as f:
        midi.writeFile(f)

    print(f"\nWritten: {output_path}")
    print(f"\nWhen running advanced_sampler.py, pass matching timing args:")
    print(f"  python advanced_sampler.py <rendered.flac> --hold-ms {hold_ms:.0f} --gap-ms {gap_ms:.0f}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-o", "--output", default="analyze_instrument.mid",
                   help="Output MIDI path (default: analyze_instrument.mid)")
    p.add_argument("--hold-ms", type=float, default=100.0,
                   help="Note-on duration in ms (default: 100). "
                        "Long enough to capture the attack character.")
    p.add_argument("--gap-ms",  type=float, default=200.0,
                   help="Silence after note-off in ms (default: 500). "
                        "Must be >= your instrument's decay time to avoid bleed.")
    p.add_argument("--start-note", type=int, default=PIANO_LOW,
                   help=f"Lowest MIDI note (default: {PIANO_LOW} = A0)")
    p.add_argument("--end-note",   type=int, default=PIANO_HIGH,
                   help=f"Highest MIDI note (default: {PIANO_HIGH} = C8)")
    args = p.parse_args()

    if args.hold_ms <= 0:
        p.error("--hold-ms must be positive")
    if args.gap_ms < 0:
        p.error("--gap-ms must be non-negative")

    generate(args.hold_ms, args.gap_ms,
             args.start_note, args.end_note,
             args.output)


if __name__ == "__main__":
    main()
