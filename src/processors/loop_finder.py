"""
loop_finder.py — LoopFinderProcessor: auto-detect loop points for sustain samples.

Searches the last 15% of each sustain sample (85%–97% of total length, leaving
a 3% tail for crossfade) for a pair of upward zero crossings whose amplitude
and waveform shape match well.

Algorithm:
  1. Convert to mono for analysis.
  2. Search window: [85%, 97%] of sample length.
  3. Collect all upward zero crossings in the window.
  4. Score candidate (loop_start, loop_end) pairs by:
       - Amplitude similarity: local 10ms RMS at both points should match.
       - Waveform correlation: 20ms window centred on each crossing.
  5. Best combined score wins. Falls back to first/last crossing if scoring
     finds no valid pair above the minimum gap threshold.

Results are stored in Sample.loop_start / Sample.loop_end (sample indices).
Only sustain samples are processed; release samples are left unchanged.
"""

from __future__ import annotations

import numpy as np

from ..base import ProcessingObject
from ..data import InstrumentData, Sample


# ---------------------------------------------------------------------------
# Core loop-point detection function
# ---------------------------------------------------------------------------

def find_loop_point(
    data: np.ndarray,
    sr: int,
    region_start_frac: float = 0.85,
    region_end_frac: float = 0.97,
    min_loop_ms: float = 50.0,
) -> tuple[int, int] | None:
    """
    Find a good (loop_start, loop_end) sample-index pair near the end of a sample.

    Both returned indices are upward zero crossings so the loop boundary has
    matching waveform phase, minimising clicks. The caller uses loop_crossfade
    in the SFZ to smooth any residual amplitude step.

    Args:
        data:               Audio array (float32/64, mono or stereo).
        sr:                 Sample rate.
        region_start_frac:  Start of search window as fraction of total length.
        region_end_frac:    End of search window (leaves tail for crossfade).
        min_loop_ms:        Minimum gap between loop_start and loop_end in ms.

    Returns:
        (loop_start, loop_end) as integer sample indices, or None if the
        sample is too short or no suitable crossings are found.
    """
    # Work in mono for analysis
    mono = data if data.ndim == 1 else data.mean(axis=1)
    n = len(mono)

    start_idx = int(n * region_start_frac)
    end_idx = int(n * region_end_frac)
    min_gap = int(min_loop_ms / 1000.0 * sr)

    if end_idx - start_idx < min_gap:
        return None  # Search region too short

    # --- Upward zero crossings in [start_idx, end_idx) -----------------------
    region = mono[start_idx:end_idx]
    crossings: list[int] = []
    for i in range(1, len(region)):
        if region[i - 1] < 0.0 and region[i] >= 0.0:
            crossings.append(start_idx + i)

    if len(crossings) < 2:
        return None  # Not enough crossings to form a pair

    # --- Local RMS helper (10 ms window) -------------------------------------
    rms_win = max(1, int(0.010 * sr))

    def local_rms(idx: int) -> float:
        s = max(0, idx - rms_win // 2)
        e = min(n, idx + rms_win // 2)
        chunk = mono[s:e]
        return float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0

    amplitudes = [local_rms(c) for c in crossings]

    # --- Waveform correlation helper (20 ms window) ---------------------------
    corr_win = max(1, int(0.020 * sr))

    def corr_score(i_idx: int, j_idx: int) -> float:
        """Normalised dot-product correlation in [-1, 1]; higher = better."""
        si = max(0, i_idx - corr_win // 2)
        ei = min(n, i_idx + corr_win // 2)
        sj = max(0, j_idx - corr_win // 2)
        ej = min(n, j_idx + corr_win // 2)
        wi = mono[si:ei]
        wj = mono[sj:ej]
        length = min(len(wi), len(wj))
        if length < 4:
            return 0.0
        wi = wi[:length]
        wj = wj[:length]
        norm = np.linalg.norm(wi) * np.linalg.norm(wj)
        if norm < 1e-12:
            return 0.0
        return float(np.dot(wi, wj) / norm)

    # --- Score candidate pairs ------------------------------------------------
    # Use first MAX_CANDS as potential loop_start candidates,
    # last MAX_CANDS as potential loop_end candidates.
    # This keeps scoring O(MAX_CANDS²) regardless of crossing count.
    MAX_CANDS = 10
    start_cands = crossings[:MAX_CANDS]
    end_cands = crossings[max(0, len(crossings) - MAX_CANDS):]
    start_amps = amplitudes[:MAX_CANDS]
    end_amps = amplitudes[max(0, len(crossings) - MAX_CANDS):]

    best_score = -1.0
    best_pair: tuple[int, int] = (crossings[0], crossings[-1])

    for ls, a_ls in zip(start_cands, start_amps):
        for le, a_le in zip(end_cands, end_amps):
            if le - ls < min_gap:
                continue
            # Amplitude similarity: 1.0 = identical amplitude, 0.0 = one is silent
            amp_sum = a_ls + a_le + 1e-10
            amp_sim = 1.0 - abs(a_ls - a_le) / amp_sum

            # Waveform shape correlation
            corr = corr_score(ls, le)

            # Combined score: amplitude similarity weighted slightly more
            score = 0.65 * amp_sim + 0.35 * max(corr, 0.0)
            if score > best_score:
                best_score = score
                best_pair = (ls, le)

    # Ensure the final pair satisfies the minimum gap (fallback pair might not
    # if the region is extremely sparse in crossings)
    if best_pair[1] - best_pair[0] < min_gap:
        return None

    return best_pair


# ---------------------------------------------------------------------------
# ProcessingObject subclass
# ---------------------------------------------------------------------------

class LoopFinderProcessor(ProcessingObject):
    """
    Detect and store loop points for every sustain sample.

    Loop points are stored in Sample.loop_start / Sample.loop_end as integer
    sample indices. Release samples are left unchanged (no loop needed).

    Configuration is read from data.meta:
        loop_crossfade_ms — used to compute the tail margin (region_end_frac).
                            The search window ends where the crossfade tail begins,
                            so there is always enough audio after loop_end for
                            the sampler to crossfade.
    """

    def process(self, data: InstrumentData) -> InstrumentData:
        for sample in data.sustain:
            n = len(sample.audio) if sample.audio.ndim == 1 else sample.audio.shape[0]
            if n == 0:
                continue

            # Leave enough tail for the SFZ crossfade (at minimum 3% or crossfade length)
            crossfade_samples = int(data.meta.loop_crossfade_ms / 1000.0 * sample.sr)
            tail_frac = max(crossfade_samples / n, 0.03)
            region_end_frac = min(1.0 - tail_frac, 0.97)

            result = find_loop_point(
                sample.audio,
                sample.sr,
                region_start_frac=0.85,
                region_end_frac=region_end_frac,
            )
            if result is not None:
                sample.loop_start, sample.loop_end = result
            # If None: loop points remain None; OutputModule will skip loop opcodes

        return data
