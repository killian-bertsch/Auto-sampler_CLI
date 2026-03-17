"""
transient_shaper.py — SPL-style transient designer processor.

Algorithm: dual-envelope follower (fast + slow).

  fast_env  ≈ instantaneous amplitude (tracks transients)
  slow_env  ≈ RMS body / sustain (slow_TC = fast_TC × 20)

  transient_signal = max(0, fast_env - slow_env)   — 1 during attack, 0 otherwise
  sustain_signal   = 1 - transient_signal

  gain(t) = attack_lin * transient_signal(t) + sustain_lin * sustain_signal(t)

Applied to all sustain and release samples when enable_transient_shaper = true.
"""

import numpy as np

from ..base import ProcessingObject
from ..data import InstrumentData


def _make_coeff(time_ms: float, samplerate: int) -> float:
    """First-order IIR smoothing coefficient for a given time constant."""
    if time_ms <= 0:
        return 0.0
    tc_samples = (time_ms / 1000.0) * samplerate
    return float(np.exp(-1.0 / max(tc_samples, 1.0)))


def _apply_transient_shaper(
    data: np.ndarray,
    samplerate: int,
    attack_db: float,
    sustain_db: float,
    speed_ms: float,
) -> np.ndarray:
    """
    Apply a transient shaper to `data`.

    Works on mono (N,) or stereo (N, C) float arrays.
    Envelope detection uses the mid channel; gain applied to all channels.
    Output is soft-clipped with tanh to prevent intersample overs.
    """
    if data.size == 0:
        return data

    orig_dtype = data.dtype
    x = data.astype(np.float64)

    mid = x if x.ndim == 1 else x.mean(axis=1)
    n = len(mid)
    amp = np.abs(mid)

    coeff_fast = _make_coeff(speed_ms, samplerate)
    coeff_slow = _make_coeff(speed_ms * 20.0, samplerate)

    fast_env = np.zeros(n, dtype=np.float64)
    slow_env = np.zeros(n, dtype=np.float64)
    fast_prev = slow_prev = 0.0

    for i in range(n):
        fast_prev = max(amp[i], coeff_fast * fast_prev)
        slow_prev = max(amp[i], coeff_slow * slow_prev)
        fast_env[i] = fast_prev
        slow_env[i] = slow_prev

    diff = np.maximum(0.0, fast_env - slow_env)
    max_diff = diff.max()
    if max_diff > 1e-12:
        transient_sig = diff / max_diff
    else:
        transient_sig = np.zeros(n, dtype=np.float64)

    sustain_sig = 1.0 - transient_sig

    attack_lin = 10.0 ** (attack_db / 20.0)
    sustain_lin = 10.0 ** (sustain_db / 20.0)
    gain = attack_lin * transient_sig + sustain_lin * sustain_sig  # (N,)

    out = x * gain if x.ndim == 1 else x * gain[:, np.newaxis]
    out = np.tanh(out)
    return out.astype(orig_dtype)


class TransientShaperProcessor(ProcessingObject):
    """
    Applies an SPL-style transient shaper to all sustain and release samples.

    Skipped entirely when meta.enable_transient_shaper is False.
    Parameters read from meta:
        transient_attack_db  — gain on transient portion  (default +6 dB)
        transient_sustain_db — gain on body/sustain       (default  0 dB)
        transient_speed_ms   — fast envelope time constant (default 10 ms)
    """

    def process(self, data: InstrumentData) -> InstrumentData:
        if not data.meta.enable_transient_shaper:
            return data

        attack_db = data.meta.transient_attack_db
        sustain_db = data.meta.transient_sustain_db
        speed_ms = data.meta.transient_speed_ms

        for sample in data.sustain:
            sample.audio = _apply_transient_shaper(
                sample.audio,
                sample.sr,
                attack_db=attack_db,
                sustain_db=sustain_db,
                speed_ms=speed_ms,
            )

        return data
