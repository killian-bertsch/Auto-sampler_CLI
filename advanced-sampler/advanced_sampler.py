#!/usr/bin/env python3
"""
advanced_sampler.py — Multi-stage instrument sampling analysis tool.

Steps performed:
  Step 2: Spectrum-analyze every note at every velocity in the analysis recording.
          Detects velocity points where the sample switches (timbre change, NOT just
          a level/volume change).  Comparison is done on level-normalized FFT shapes,
          so a note getting louder without changing timbre is ignored.

  Step 3: Generate sampler.mid — plays each note at the detected switch velocities
          plus velocity 1 and 127.  For each note/velocity pair the MIDI contains:
            • Sustain sample  (sustain_hold on, then sustain_release silence)
            • Release sample  (release_hold on, then release_release silence)
              — only added when --sample-release flag is set

  Step 4: Generate <output>_input.toml — exactly matches sampler.mid so the
          Auto-Sampler CLI pipeline can slice the resulting recording correctly.

Usage:
    python advanced_sampler.py <analyze_flac> [options]

    <analyze_flac>           Path to the FLAC exported by feeding your instrument
                             analyze_instrument.mid (88 notes × 127 velocities).

    --hold-ms / --gap-ms     MUST match the values used in generate_analyze_midi.py.
                             Defaults: 100 ms hold, 500 ms gap.

Options:
    -o, --output STEM        Output stem (default: "sampler")
                             Writes <stem>.mid and <stem>_input.toml

    --hold-ms FLOAT          Note-on duration used in the analysis MIDI (default: 100)
    --gap-ms  FLOAT          Silence after note-off used in the analysis MIDI (default: 500)

    --analysis-offset-ms FLOAT
                             Skip this many ms from the START of each hold window before
                             analyzing.  Use to avoid residual bleed from the previous
                             note that hasn't fully decayed.  (default: 20)

    --max-layers INT         Maximum velocity layers to detect per note (default: 8).
                             Hard cap — the algorithm will never find more than this many groups.

    --tolerance SPEC         Gap sensitivity for cluster detection (default: 0.5).
                             Two forms accepted:
                               Single float — applied globally to all velocities:
                                 --tolerance 0.3
                               Per-velocity-range — space-separated "lo-hi:tol" tokens:
                                 --tolerance "1-50:.5 51-75:.2 76-127:.1"
                             Each range is clustered independently with its own tolerance,
                             so you can be more sensitive in the upper velocity registers
                             without over-detecting in the lower ones.
                             A dendrogram gap must be >= tol × max_gap to count as a boundary.
                               1.0 → only the single biggest gap counts  → 2 clusters max
                               0.5 → any gap ≥ 50% of the biggest counts → more clusters
                               0.1 → very sensitive, finds many smaller boundaries
                             Lower = more switches found (up to --max-layers cap per segment).

    --sustain-hold FLOAT     Sustain note-on duration in seconds (default: 10.0)
    --sustain-release FLOAT  Silence after sustain note-off in seconds (default: 0.5)
    --sample-release         If set, also record release tail after each sustain sample.
    --release-hold FLOAT     Release note-on duration in seconds (default: 0.5)
    --release-release FLOAT  Silence after release note-off in seconds (default: 1.0)

    --start-note INT         Lowest MIDI note in analyze_instrument.mid (default: 21 = A0)
    --end-note   INT         Highest MIDI note in analyze_instrument.mid (default: 108 = C8)
    --collapse-to-mono       Write collapse_to_mono=true in the output input.toml
    --debug-note INT         Print per-velocity cluster assignments for this MIDI note.
"""

import argparse
import sys
import numpy as np
import soundfile as sf
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

try:
    from midiutil import MIDIFile
except ImportError:
    print("midiutil not found.  Run: pip install midiutil")
    sys.exit(1)

NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]


def midi_note_to_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{(n // 12) - 1}"


# ── Step 2: slicing + spectral analysis ───────────────────────────────────────

def extract_note_samples(audio: np.ndarray, sr: int,
                         start_note: int, end_note: int,
                         hold_ms: float, gap_ms: float,
                         analysis_offset_ms: float = 0.0) -> dict:
    """
    Slice the analysis recording into {note: {velocity: audio_array}}.
    Recording order: outer loop = note (start_note→end_note),
                     inner loop = velocity (1→127).

    hold_ms / gap_ms must match the values used in generate_analyze_midi.py.
    analysis_offset_ms: skip this many ms from the start of each hold window
                        to avoid bleed from the previous note's decay.
    """
    hold_s   = hold_ms   / 1000.0
    event_s  = (hold_ms + gap_ms) / 1000.0
    offset_s = analysis_offset_ms / 1000.0

    event_samp  = int(round(event_s  * sr))
    hold_samp   = int(round(hold_s   * sr))
    offset_samp = int(round(offset_s * sr))

    # Clamp offset so we always have at least a few samples to analyze
    min_analysis = max(4, hold_samp // 4)
    if offset_samp >= hold_samp - min_analysis:
        offset_samp = max(0, hold_samp - min_analysis)

    notes = list(range(start_note, end_note + 1))

    # Mix to mono for analysis
    mono = audio.mean(axis=1) if audio.ndim > 1 else audio

    result = {}
    idx = 0
    for note in notes:
        result[note] = {}
        for vel in range(1, 128):
            event_start = idx * event_samp
            seg_start   = event_start + offset_samp   # skip bleed
            seg_end     = event_start + hold_samp
            if seg_end <= len(mono):
                result[note][vel] = mono[seg_start:seg_end].copy()
            elif seg_start < len(mono):
                chunk = mono[seg_start:].copy()
                needed = seg_end - seg_start
                result[note][vel] = np.pad(chunk, (0, max(0, needed - len(chunk))))
            else:
                result[note][vel] = np.zeros(max(hold_samp - offset_samp, 4),
                                             dtype=np.float32)
            idx += 1

    return result


def _spectral_shape(audio: np.ndarray, sr: int, n_bands: int = 128) -> np.ndarray:
    """
    Compute a level-normalized, log-frequency-binned power spectral shape.

    1. Hann-windowed FFT → power spectrum
    2. Bin into n_bands log-spaced bands (20 Hz – Nyquist)
    3. Normalize to unit sum → level-independent shape

    Returns a 1-D float array of length n_bands.
    """
    if len(audio) < 4:
        return np.full(n_bands, 1.0 / n_bands)

    win  = np.hanning(len(audio))
    spec = np.abs(np.fft.rfft(audio * win)) ** 2        # power
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)

    fmin, fmax = 20.0, sr / 2.0
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)

    bands = np.zeros(n_bands, dtype=np.float64)
    for i in range(n_bands):
        lo = int(np.searchsorted(freqs, edges[i]))
        hi = int(np.searchsorted(freqs, edges[i + 1]))
        if hi > lo:
            bands[i] = spec[lo:hi].mean()
        elif lo > 0:
            bands[i] = spec[lo - 1]
        # else bands[i] stays 0

    total = bands.sum()
    if total > 1e-30:
        bands /= total
    else:
        bands[:] = 1.0 / n_bands

    return bands


def _rms_db(audio: np.ndarray) -> float:
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    return 20.0 * np.log10(rms + 1e-12)


def _spectral_distance(s1: np.ndarray, s2: np.ndarray) -> float:
    """
    Log-spectral distance between two normalized spectral shapes.
    Level-insensitive because shapes are already normalized to unit sum.
    """
    eps = 1e-10
    return float(np.mean(np.abs(np.log(s1 + eps) - np.log(s2 + eps))))


def _parse_tolerance(raw: str):
    """
    Parse the --tolerance argument.  Two accepted forms:

      Single float:   "0.5"
      Per-range:      "0-50:.5 51-75:.2 76-127:.1"
                      Each token is  lo-hi:tol  (velocity range, inclusive).

    Returns either:
      float                        — single global tolerance (existing behaviour)
      list of (lo, hi, tol) tuples — per-range tolerances
    """
    raw = raw.strip()
    # Try plain float first
    try:
        return float(raw)
    except ValueError:
        pass

    segments = []
    for token in raw.split():
        try:
            range_part, tol_part = token.split(":")
            lo_str, hi_str = range_part.split("-")
            lo, hi, tol = int(lo_str), int(hi_str), float(tol_part)
            if not (1 <= lo <= hi <= 127):
                raise ValueError(f"velocity range {lo}-{hi} out of bounds (1–127)")
            if not (0.0 <= tol <= 1.0):
                raise ValueError(f"tolerance {tol} out of range [0, 1]")
            segments.append((lo, hi, tol))
        except (ValueError, TypeError) as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid tolerance token '{token}'. "
                f"Use a float (e.g. 0.5) or range specs like '0-50:.5 51-75:.2 76-127:.1'. "
                f"Detail: {exc}"
            )
    if not segments:
        raise argparse.ArgumentTypeError("--tolerance: no valid segments found")
    return segments


def _cluster_segment(seg_vels: list, shapes: dict,
                     tol: float, max_layers: int,
                     debug: bool = False):
    """
    Run hierarchical clustering on a subset of velocities (seg_vels) using
    the given tolerance.  Returns (switch_velocities, debug_info_dict).

    switch_velocities: sorted list of velocities where a cluster boundary occurs.
    debug_info_dict:   keys useful for --debug-note printing (may be empty if
                       the segment has fewer than 2 audible velocities).
    """
    n = len(seg_vels)
    if n < 2:
        return [], {}

    X = np.array([shapes[v] for v in seg_vels])

    pw = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _spectral_distance(X[i], X[j])
            pw[i, j] = d
            pw[j, i] = d

    condensed = squareform(pw)
    Z = linkage(condensed, method='average')
    merge_dists = Z[:, 2]

    look_back = min(max_layers - 1, len(merge_dists))
    target_clusters = 1
    last_dists = np.array([])
    gaps        = np.array([])
    threshold   = 0.0
    n_significant = 0

    if look_back >= 1:
        last_dists = merge_dists[-look_back:]
        gaps = np.diff(last_dists)
        if len(gaps) > 0 and gaps.max() > 0:
            threshold = tol * gaps.max()
            n_significant = int((gaps >= threshold).sum())
            target_clusters = min(n_significant + 1, max_layers)

    if target_clusters >= 2:
        labels = fcluster(Z, t=target_clusters, criterion='maxclust')
    else:
        labels = np.ones(n, dtype=int)

    sw = []
    prev = labels[0]
    for i in range(1, n):
        if labels[i] != prev:
            sw.append(seg_vels[i])
            prev = labels[i]

    dbg = {
        "target_clusters": target_clusters,
        "n_clusters": int(np.unique(labels).size),
        "look_back": look_back,
        "last_dists": last_dists,
        "gaps": gaps,
        "threshold": threshold,
        "n_significant": n_significant,
        "labels": labels,
        "vels": seg_vels,
    }
    return sw, dbg


def _apply_min_distance(sw_list: list, min_distance: int) -> list:
    """
    Filter switch velocities so no two consecutive switches are closer than
    min_distance apart.  Greedy left-to-right: keep the first switch, drop any
    following switch that is within min_distance of the last kept one.
    """
    if min_distance <= 1 or not sw_list:
        return sw_list
    kept = [sw_list[0]]
    for v in sw_list[1:]:
        if v - kept[-1] >= min_distance:
            kept.append(v)
    return kept


def detect_sample_switches(note_data: dict, sr: int,
                            tolerance=0.5,
                            max_layers: int = 8,
                            min_distance: int = 1,
                            min_level_db: float = -50.0,
                            n_bands: int = 128,
                            debug_note: int = None) -> dict:
    """
    For each note, cluster all 127 velocity spectra into natural groups using
    hierarchical clustering.  Cluster boundaries = sample switch points.

    tolerance accepts two forms:
      float                        — single global tolerance (original behaviour).
      list of (lo, hi, tol) tuples — per-velocity-range tolerances; each range is
                                     clustered independently, results are merged.

    When using per-range tolerances each segment is clustered separately with its
    own sensitivity, making it easy to apply tighter tolerances in the upper
    velocity registers where sample switches are often closer together.

    min_distance: after all clustering and the max_layers cap, drop any switch
                  that is fewer than min_distance velocity steps from the previous
                  kept switch (greedy left-to-right).  Default 1 = no filtering.

    Algorithm (per segment):
      1. Compute level-normalized spectral shape for every audible velocity.
      2. Build the full pairwise log-spectral distance matrix for the segment.
      3. Run average-linkage hierarchical clustering.
      4. Count dendrogram gaps >= tolerance * max_gap as cluster boundaries.
      5. Cut dendrogram → find velocity transitions.
    """
    # Normalise tolerance to a list of segments for uniform handling
    if isinstance(tolerance, (int, float)):
        tol_segments = [(1, 127, float(tolerance))]
    else:
        tol_segments = list(tolerance)   # already a list of (lo, hi, tol)

    switches = {}
    for note, vel_data in sorted(note_data.items()):
        # ── compute spectral shapes for all audible velocities ───────────────
        shapes = {}
        for vel in range(1, 128):
            audio = vel_data[vel]
            if _rms_db(audio) >= min_level_db:
                shapes[vel] = _spectral_shape(audio, sr, n_bands)

        audible_vels = sorted(shapes.keys())
        if len(audible_vels) < 2:
            switches[note] = []
            continue

        # ── cluster each tolerance segment independently ──────────────────────
        all_sw   = set()
        dbg_segs = []   # for --debug-note output

        for lo, hi, tol in tol_segments:
            seg_vels = [v for v in audible_vels if lo <= v <= hi]
            sw, dbg = _cluster_segment(seg_vels, shapes, tol, max_layers,
                                       debug=(note == debug_note))
            all_sw.update(sw)
            if note == debug_note and dbg:
                dbg_segs.append((lo, hi, tol, dbg))

        sw_list = sorted(all_sw)

        # ── Hard cap: merged switches across all segments must not exceed max_layers-1
        if len(sw_list) >= max_layers:
            indices = np.round(np.linspace(0, len(sw_list) - 1, max_layers - 1)).astype(int)
            sw_list = sorted({sw_list[i] for i in indices})

        # ── Minimum distance: drop switches too close to their predecessor ────
        sw_list = _apply_min_distance(sw_list, min_distance)

        # ── --debug-note output ──────────────────────────────────────────────
        if note == debug_note:
            print(f"\n  DEBUG note {note} ({midi_note_to_name(note)}):")
            if len(tol_segments) == 1:
                lo, hi, tol = tol_segments[0]
                if dbg_segs:
                    _, _, _, dbg = dbg_segs[0]
                    print(f"    tolerance={tol}  target={dbg['target_clusters']} "
                          f"clusters found={dbg['n_clusters']}  |  "
                          f"last {dbg['look_back']} merge dists: "
                          f"{dbg['last_dists'].round(4).tolist()}")
                    if dbg['look_back'] >= 2 and len(dbg['gaps']) > 0:
                        print(f"    Gaps: {dbg['gaps'].round(4).tolist()}  "
                              f"threshold={dbg['threshold']:.4f}  "
                              f"significant={dbg['n_significant']}")
                    print(f"    Velocity → cluster:")
                    for i, vel in enumerate(dbg['vels']):
                        flag = " ← SWITCH" if vel in sw_list else ""
                        print(f"      vel {vel:3d}: cluster {dbg['labels'][i]}{flag}")
            else:
                print(f"    Per-range mode: {len(tol_segments)} segment(s)")
                for lo, hi, tol, dbg in dbg_segs:
                    print(f"\n    Segment vel {lo}–{hi}  tolerance={tol}  "
                          f"target={dbg['target_clusters']} "
                          f"clusters found={dbg['n_clusters']}")
                    if dbg['look_back'] >= 2 and len(dbg['gaps']) > 0:
                        print(f"      Gaps: {dbg['gaps'].round(4).tolist()}  "
                              f"threshold={dbg['threshold']:.4f}  "
                              f"significant={dbg['n_significant']}")
                    for i, vel in enumerate(dbg['vels']):
                        flag = " ← SWITCH" if vel in sw_list else ""
                        print(f"      vel {vel:3d}: cluster {dbg['labels'][i]}{flag}")

        switches[note] = sw_list

    return switches


# ── Switch point grid visualizer ──────────────────────────────────────────────

def print_switch_grid(switches: dict, start_note: int, end_note: int):
    """Print an ASCII velocity×note grid highlighting sample switch points."""
    notes = list(range(start_note, end_note + 1))
    n = len(notes)

    switch_set = set()
    for note, sw_list in switches.items():
        for vel in sw_list:
            switch_set.add((note, vel))

    total = sum(len(v) for v in switches.values())
    print(f"\n── Velocity × Note Switch Map  ({total} switches) {'─' * max(0, 40 - len(str(total)))}")
    print("   ■ = switch point   · = no switch")
    print()

    for vel in range(127, 0, -1):
        row = "".join("■" if (note, vel) in switch_set else "·" for note in notes)
        print(f"{vel:3d}│{row}")

    print("   └" + "─" * n)

    # Note name labels at every C note
    name_chars = [' '] * n
    for i, note in enumerate(notes):
        if note % 12 == 0:
            name = midi_note_to_name(note)
            for j, ch in enumerate(name):
                if i + j < n:
                    name_chars[i + j] = ch
    print("    " + ''.join(name_chars))
    print()


# ── Step 3: MIDI generation ────────────────────────────────────────────────────

def _velocities_for_note(switches: dict, note: int) -> list:
    """Always include v=1 and v=127; add all detected switch velocities."""
    vels = set([1, 127])
    vels.update(switches.get(note, []))
    return sorted(vels)


def generate_sampler_midi(switches: dict,
                           start_note: int, end_note: int,
                           sustain_hold: float, sustain_release: float,
                           sample_release: bool,
                           release_hold: float, release_release: float,
                           output_path: str) -> list:
    """
    Write sampler MIDI.  For each (note, velocity) pair:
      • Sustain event : note-on for sustain_hold s, then sustain_release s silence
      • Release event : note-on for release_hold s, then release_release s silence
                        (only if sample_release=True)

    Returns the ordered list of event dicts for input.toml.
    """
    bpm = 120.0
    bps = bpm / 60.0   # beats per second

    s_hold_b  = sustain_hold    * bps
    s_rel_b   = sustain_release * bps
    r_hold_b  = release_hold    * bps if sample_release else 0.0
    r_rel_b   = release_release * bps if sample_release else 0.0

    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    midi.addTrackName(0, 0, "Autosample")

    events = []
    beat   = 0.0

    for note in range(start_note, end_note + 1):
        vels = _velocities_for_note(switches, note)
        for vel in vels:
            # ── Sustain ──────────────────────────────────────────────────────
            midi.addNote(track=0, channel=0, pitch=note,
                         time=beat, duration=s_hold_b, volume=vel)
            events.append({
                "note": note, "velocity": vel,
                "hold_s": sustain_hold, "tail_s": sustain_release,
                "is_release": False,
            })
            beat += s_hold_b + s_rel_b

            # ── Release tail (optional) ───────────────────────────────────
            if sample_release:
                midi.addNote(track=0, channel=0, pitch=note,
                             time=beat, duration=r_hold_b, volume=vel)
                events.append({
                    "note": note, "velocity": vel,
                    "hold_s": release_hold, "tail_s": release_release,
                    "is_release": True,
                })
                beat += r_hold_b + r_rel_b

    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return events


# ── Step 4: input.toml generation ─────────────────────────────────────────────

def generate_input_toml(events: list, collapse_to_mono: bool, output_path: str):
    """Write input.toml that exactly describes the sampler MIDI recording."""
    mono_val = "true" if collapse_to_mono else "false"
    lines = [
        "# Auto-Sampler CLI — Input Sample Manifest",
        "# Generated by advanced_sampler.py",
        "#",
        "# Entries appear in the SAME ORDER as events in the rendered .flac.",
        "# For is_release=false (sustain): slicer extracts [0, hold_s].",
        "# For is_release=true  (release): slicer extracts [hold_s, hold_s+tail_s].",
        "",
        f"collapse_to_mono = {mono_val}",
        "",
        "samples = [",
    ]
    for e in events:
        rel_str = "true " if e["is_release"] else "false"
        lines.append(
            f"  {{note={e['note']:3d}, velocity={e['velocity']:3d}, "
            f"hold_s={e['hold_s']:.4f}, tail_s={e['tail_s']:.4f}, "
            f"is_release={rel_str}}},"
        )
    lines += ["]", ""]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze instrument recording and generate targeted sampling MIDI + input.toml.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("analyze_flac",
                        help="FLAC exported from analyze_instrument.mid")
    parser.add_argument("-o", "--output", default="sampler",
                        help="Output stem name (default: 'sampler')")

    # Analysis MIDI timing — must match generate_analyze_midi.py
    parser.add_argument("--hold-ms", type=float, default=100.0,
                        help="Note-on duration used in the analysis MIDI (default: 100 ms)")
    parser.add_argument("--gap-ms",  type=float, default=200.0,
                        help="Gap duration used in the analysis MIDI (default: 500 ms)")
    parser.add_argument("--analysis-offset-ms", type=float, default=20.0,
                        help="Skip this many ms from the start of each analysis window "
                             "to avoid bleed from the previous note (default: 20 ms)")

    # Switch detection
    parser.add_argument("--max-layers", type=int, default=8,
                        help="Hard cap on velocity layers detected per note (default: 8).")
    parser.add_argument("--tolerance", type=_parse_tolerance, default=0.5,
                        help="Gap sensitivity for cluster detection. "
                             "Either a single float (default: 0.5) applied to all velocities, "
                             "or per-velocity-range specs separated by spaces: "
                             "'lo-hi:tol lo-hi:tol …'  e.g. '1-50:.5 51-75:.2 76-127:.1'. "
                             "Each range is clustered independently with its own sensitivity. "
                             "Lower tolerance = more sensitive = more switches found.")
    parser.add_argument("--minimum-distance", type=int, default=1,
                        help="Minimum velocity gap between consecutive switch points (default: 1 = no filtering). "
                             "Switches closer than this to the previous kept switch are dropped (greedy). "
                             "Useful to collapse clusters of nearby switches into one.")
    parser.add_argument("--debug-note", type=int, default=None,
                        help="Print per-velocity cluster assignments for this MIDI note number.")

    # Recording parameters for sampler MIDI
    parser.add_argument("--sustain-hold",    type=float, default=10.0,
                        help="Sustain note-on duration in seconds (default: 10.0)")
    parser.add_argument("--sustain-release", type=float, default=0.5,
                        help="Silence after sustain note-off in seconds (default: 0.5)")
    parser.add_argument("--sample-release",  action="store_true",
                        help="Also record release tail samples")
    parser.add_argument("--release-hold",    type=float, default=0.5,
                        help="Release note-on duration in seconds (default: 0.5)")
    parser.add_argument("--release-release", type=float, default=1.0,
                        help="Silence after release note-off in seconds (default: 1.0)")

    # Note range (must match what was used in generate_analyze_midi.py)
    parser.add_argument("--start-note", type=int, default=21,
                        help="Lowest MIDI note in analyze recording (default: 21 = A0)")
    parser.add_argument("--end-note",   type=int, default=108,
                        help="Highest MIDI note in analyze recording (default: 108 = C8)")

    parser.add_argument("--collapse-to-mono", action="store_true",
                        help="Set collapse_to_mono=true in the output input.toml")

    args = parser.parse_args()

    # ── Resolve output stem relative to the input file's directory ────────────
    import os as _os
    _input_dir = _os.path.dirname(_os.path.abspath(args.analyze_flac))
    # If the user gave a bare stem (no directory separator), place it next to the input file.
    # If they gave an explicit path (relative or absolute), respect it as-is.
    if _os.sep not in args.output and (not _os.path.dirname(args.output)):
        args.output = _os.path.join(_input_dir, args.output)

    # ── Load FLAC ──────────────────────────────────────────────────────────────
    print(f"Loading: {args.analyze_flac}")
    audio, sr = sf.read(args.analyze_flac, dtype="float32")
    dur_s  = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
    shape  = audio.shape
    print(f"  {dur_s:.1f} s  |  sr={sr}  |  shape={shape}")

    n_notes  = args.end_note - args.start_note + 1
    expected = n_notes * 127 * (args.hold_ms + args.gap_ms) / 1000.0
    if dur_s < expected * 0.9:
        print(f"  WARNING: expected ~{expected:.0f} s for {n_notes} notes × 127 velocities "
              f"at {args.hold_ms:.0f}+{args.gap_ms:.0f} ms/event, "
              f"got {dur_s:.1f} s.  Some samples may be truncated.")

    # ── Step 2: slice + detect switches ───────────────────────────────────────
    print(f"\nStep 2: Slicing into {n_notes} × 127 = {n_notes * 127:,} segments ...")
    print(f"        hold={args.hold_ms:.0f} ms  gap={args.gap_ms:.0f} ms  "
          f"analysis-offset={args.analysis_offset_ms:.0f} ms")
    note_data = extract_note_samples(
        audio, sr, args.start_note, args.end_note,
        hold_ms=args.hold_ms, gap_ms=args.gap_ms,
        analysis_offset_ms=args.analysis_offset_ms,
    )

    min_dist_str = f"  min-distance={args.minimum_distance}" if args.minimum_distance > 1 else ""
    print(f"        Detecting sample switches "
          f"(tolerance={args.tolerance}  max-layers={args.max_layers}{min_dist_str}) ...")
    switches = detect_sample_switches(
        note_data, sr,
        tolerance=args.tolerance,
        max_layers=args.max_layers,
        min_distance=args.minimum_distance,
        debug_note=args.debug_note,
    )

    notes_with_switches = sum(1 for sw in switches.values() if sw)
    total_switches      = sum(len(sw) for sw in switches.values())
    print(f"        Found {total_switches} sample switches across "
          f"{notes_with_switches}/{n_notes} notes.")

    if total_switches:
        print()
        for note in sorted(switches):
            sw = switches[note]
            if sw:
                name = midi_note_to_name(note)
                print(f"    {name:5s} ({note:3d}): switch at vel {sw}")

    print_switch_grid(switches, args.start_note, args.end_note)

    # ── Step 3: generate sampler MIDI ─────────────────────────────────────────
    midi_path = args.output + ".mid"
    print(f"\nStep 3: Generating {midi_path} ...")
    events = generate_sampler_midi(
        switches,
        args.start_note, args.end_note,
        args.sustain_hold, args.sustain_release,
        args.sample_release,
        args.release_hold, args.release_release,
        midi_path,
    )

    total_time = sum(e["hold_s"] + e["tail_s"] for e in events)
    n_sustain  = sum(1 for e in events if not e["is_release"])
    n_release  = sum(1 for e in events if     e["is_release"])
    print(f"  {n_sustain} sustain events"
          + (f"  +  {n_release} release events" if n_release else ""))
    print(f"  Total recording time: {total_time:.0f} s  ({total_time / 60:.1f} min)")

    # ── Step 4: generate input.toml ───────────────────────────────────────────
    toml_path = _os.path.join(_os.path.dirname(args.output), "input.toml")
    print(f"\nStep 4: Generating {toml_path} ...")
    generate_input_toml(events, args.collapse_to_mono, toml_path)

    print(f"\nDone.")
    print(f"  MIDI       : {midi_path}")
    print(f"  input.toml : {toml_path}")
    print()
    print("Next steps:")
    print(f"  1. Feed {midi_path} into your instrument and export the audio to a single .flac.")
    print(f"  2. Use {toml_path} as the input.toml for the Auto-Sampler CLI pipeline.")


if __name__ == "__main__":
    main()
