"""
loop_roll.py — Cloudy DJ 2.0

Generates a DJ-style vocal loop roll that sounds human, not mechanical.

How a real DJ loop roll works:
  1. A word/phrase repeats at accelerating rates (1×, 2×, 4×, 8×, 16×)
  2. The PITCH stays constant — achieved via time_stretch, not waveform chopping
  3. An HPF sweep strips bass away as tension builds
  4. Reverb/delay wash builds up with each repeat
  5. Drums keep playing underneath
  6. The last half-beat is SILENCE (tension gap before the drop)
"""

import numpy as np
import librosa
from scipy import signal


def _to_mono(audio):
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)


def _fit_word_to_slot(word, slot_len, sr):
    """
    Time-stretch the word to fit exactly into slot_len samples.
    Preserves pitch. Returns mono float32.
    """
    if len(word) == 0 or slot_len == 0:
        return np.zeros(slot_len, dtype=np.float32)

    rate = len(word) / slot_len
    # Clamp to sane range (librosa vocoder limits)
    rate = max(0.25, min(rate, 8.0))

    if abs(rate - 1.0) < 0.01:
        # No stretching needed, just trim/pad
        if len(word) >= slot_len:
            return word[:slot_len].copy()
        else:
            return np.pad(word, (0, slot_len - len(word)))

    try:
        stretched = librosa.effects.time_stretch(word, rate=rate)
    except Exception:
        # Fallback: tile or trim
        if len(word) >= slot_len:
            return word[:slot_len].copy()
        repeats = (slot_len // len(word)) + 2
        return np.tile(word, repeats)[:slot_len]

    # Trim or pad to exact length
    if len(stretched) >= slot_len:
        return stretched[:slot_len].astype(np.float32)
    else:
        return np.pad(stretched, (0, slot_len - len(stretched))).astype(np.float32)


def _apply_crossfade_at_boundaries(chunk, cf_samples):
    """Apply tiny fade-in and fade-out to eliminate clicks at chunk boundaries."""
    cf = min(cf_samples, len(chunk) // 4)
    if cf > 1:
        chunk[:cf]  *= np.linspace(0.0, 1.0, cf, dtype=np.float32)
        chunk[-cf:] *= np.linspace(1.0, 0.0, cf, dtype=np.float32)
    return chunk


def generate_loop_roll(vocal_stem, drum_stem, bpm, sr, drop_sample_idx,
                       num_beats=8, exact_word_start=None, exact_word_end=None):
    """
    Generate a DJ-style loop roll with pitch-preserved acceleration,
    HPF sweep, reverb buildup, and drums underneath.

    Parameters:
      vocal_stem:      mono float32 array — the vocal stem
      drum_stem:       mono float32 array — the drum stem (plays underneath)
      bpm:             track BPM
      sr:              sample rate
      drop_sample_idx: sample index of the drop
      num_beats:       how many beats the roll spans (default 8)
      exact_word_start: optional exact word start sample
      exact_word_end:   optional exact word end sample

    Returns:
      stereo float32 array (N, 2) — the complete loop roll section
    """
    from src.engine.dsp_utils import find_best_loop_source

    vocal_mono = _to_mono(vocal_stem)
    drum_mono = _to_mono(drum_stem)
    spb = int((60.0 / bpm) * sr)
    total_len = num_beats * spb
    cf_samples = int(0.003 * sr)  # 3ms crossfade

    # ---------------------------------------------------------------
    # 1. Get the source word
    # ---------------------------------------------------------------
    if exact_word_start is not None and exact_word_end is not None:
        source_word = vocal_mono[exact_word_start:exact_word_end].copy()
    else:
        ws, we = find_best_loop_source(
            {"vocals": vocal_stem}, drop_sample_idx, bpm, sr)
        source_word = vocal_mono[ws:we].copy()

    # Safety: if source is silent, grab 1 beat before the drop
    if len(source_word) == 0 or np.max(np.abs(source_word)) < 0.001:
        fallback_start = max(0, drop_sample_idx - spb)
        source_word = vocal_mono[fallback_start:drop_sample_idx].copy()

    if len(source_word) == 0:
        return np.zeros((total_len, 2), dtype=np.float32)

    # ---------------------------------------------------------------
    # 2. Build the vocal roll — pitch-preserved acceleration
    # ---------------------------------------------------------------
    # Schedule: [reps_per_beat] for each beat
    # Beat 1-2: 1 rep/beat (full speed)
    # Beat 3-4: 2 reps/beat (double time)
    # Beat 5-6: 4 reps/beat (quad time)
    # Beat 7:   8 reps/beat (machine gun)
    # Beat 8:   16 reps for first half, SILENCE for second half (tension gap)

    schedule = [1, 1, 2, 2, 4, 4, 8, 16]
    if num_beats != 8:
        # Scale schedule to match num_beats
        schedule = []
        for i in range(num_beats):
            progress = i / max(1, num_beats - 1)
            if progress < 0.25:
                schedule.append(1)
            elif progress < 0.5:
                schedule.append(2)
            elif progress < 0.75:
                schedule.append(4)
            elif progress < 0.9:
                schedule.append(8)
            else:
                schedule.append(16)

    vocal_roll = np.zeros(total_len, dtype=np.float32)
    pos = 0

    for beat_idx, reps in enumerate(schedule):
        beat_samples = spb
        is_last_beat = (beat_idx == len(schedule) - 1)

        if is_last_beat:
            # Last beat: fill first half with fastest reps, second half = silence
            active_samples = beat_samples // 2
            slot_len = max(int(0.02 * sr), active_samples // reps)  # min 20ms per rep

            for r in range(reps):
                rep_start = pos + r * slot_len
                if rep_start + slot_len > pos + active_samples:
                    break
                chunk = _fit_word_to_slot(source_word, slot_len, sr)
                chunk = _apply_crossfade_at_boundaries(chunk, cf_samples)
                # Decaying volume per rep within the burst
                chunk *= (1.0 - 0.3 * (r / max(1, reps)))
                end = min(rep_start + slot_len, total_len)
                wl = end - rep_start
                if wl > 0:
                    vocal_roll[rep_start:end] += chunk[:wl]
            # Second half = silence (tension gap) — leave zeros
        else:
            slot_len = max(int(0.02 * sr), beat_samples // reps)

            for r in range(reps):
                rep_start = pos + r * slot_len
                if rep_start >= pos + beat_samples:
                    break
                actual_slot = min(slot_len, (pos + beat_samples) - rep_start)
                chunk = _fit_word_to_slot(source_word, actual_slot, sr)
                chunk = _apply_crossfade_at_boundaries(chunk, cf_samples)
                end = min(rep_start + actual_slot, total_len)
                wl = end - rep_start
                if wl > 0:
                    vocal_roll[rep_start:end] += chunk[:wl]

        pos += beat_samples

    # ---------------------------------------------------------------
    # 3. HPF sweep on the vocal roll (builds tension)
    # ---------------------------------------------------------------
    # Sweep from 150 Hz to 3000 Hz over the duration
    num_hpf_chunks = 16
    chunk_size = max(1, total_len // num_hpf_chunks)
    nyq = 0.5 * sr
    freqs = np.logspace(np.log10(150), np.log10(3000), num_hpf_chunks)

    vocal_hpf = np.zeros_like(vocal_roll)
    for i in range(num_hpf_chunks):
        s = i * chunk_size
        e = s + chunk_size if i < num_hpf_chunks - 1 else total_len
        freq = freqs[i]
        norm = freq / nyq
        if 0.0 < norm < 1.0 and (e - s) > 20:
            try:
                b, a = signal.butter(4, norm, btype='high')
                vocal_hpf[s:e] = signal.filtfilt(b, a, vocal_roll[s:e]).astype(np.float32)
            except Exception:
                vocal_hpf[s:e] = vocal_roll[s:e]
        else:
            vocal_hpf[s:e] = vocal_roll[s:e]

    # ---------------------------------------------------------------
    # 4. Reverb buildup — increasing wet mix over duration
    # ---------------------------------------------------------------
    # Simple delay-based reverb approximation
    delay_samples = spb // 2  # half-beat delay
    reverb_buf = np.zeros(total_len + delay_samples * 4, dtype=np.float32)
    reverb_buf[:total_len] = vocal_hpf

    for tap in range(1, 5):
        offset = tap * delay_samples
        feedback = 0.3 / tap  # decreasing feedback
        if offset < len(reverb_buf):
            end_pos = min(total_len, len(reverb_buf) - offset)
            reverb_buf[offset:offset + end_pos] += vocal_hpf[:end_pos] * feedback

    # Mix dry + wet with increasing wet amount
    wet_curve = np.linspace(0.0, 0.5, total_len, dtype=np.float32)
    dry_curve = 1.0 - wet_curve * 0.3  # keep dry mostly full

    vocal_final = vocal_hpf * dry_curve + reverb_buf[:total_len] * wet_curve

    # ---------------------------------------------------------------
    # 5. Crescendo — subtle volume automation
    # ---------------------------------------------------------------
    crescendo = np.linspace(0.8, 1.1, total_len, dtype=np.float32)
    vocal_final *= crescendo

    # Apply silence in the last half-beat (tension gap)
    gap_len = spb // 2
    # Fade into silence over 50ms instead of hard cut
    fade_len = min(int(0.05 * sr), gap_len)
    if total_len > gap_len:
        fade_start = total_len - gap_len
        # Fade out
        if fade_len > 0 and fade_start > 0:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            vocal_final[fade_start:fade_start + fade_len] *= fade[:min(fade_len, total_len - fade_start)]
        # Silence
        vocal_final[fade_start + fade_len:] = 0.0

    # ---------------------------------------------------------------
    # 6. Layer drums underneath
    # ---------------------------------------------------------------
    # Use A's drums from the buildup region (before the drop)
    drum_start = max(0, drop_sample_idx - total_len)
    drum_end = min(drum_start + total_len, len(drum_mono))
    drum_section = np.zeros(total_len, dtype=np.float32)

    available = drum_end - drum_start
    if available > 0:
        drum_section[:available] = drum_mono[drum_start:drum_end]

    # Drums fade out in the last 2 beats (they also stop before the tension gap)
    drum_fade_len = min(2 * spb, total_len)
    drum_fade_start = max(0, total_len - drum_fade_len)
    drum_fade = np.ones(total_len, dtype=np.float32)
    drum_fade[drum_fade_start:] = np.linspace(1.0, 0.0, total_len - drum_fade_start,
                                               dtype=np.float32)
    drum_section *= drum_fade * 0.7  # slightly duck drums below the vocal

    # ---------------------------------------------------------------
    # 7. Combine and return stereo
    # ---------------------------------------------------------------
    combined = vocal_final + drum_section

    # Peak normalize to avoid clipping
    peak = np.max(np.abs(combined))
    if peak > 0.95:
        combined *= 0.95 / peak

    return np.stack([combined, combined], axis=1).astype(np.float32)
