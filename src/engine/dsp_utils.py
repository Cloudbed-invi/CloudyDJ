"""
dsp_utils.py — Core DSP helpers for Cloudy DJ 2.0

All audio arrays are expected to be stereo (N, 2) float32.
Functions that receive mono (N,) arrays will handle them and return the
same shape they received, so callers don't need to worry.
"""

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_mono(audio):
    """Return a 1-D float32 view regardless of input shape."""
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)


def _is_stereo(audio):
    return audio.ndim == 2 and audio.shape[1] == 2


def _apply_filter_to_audio(audio, b_coeff, a_coeff):
    """
    Apply a scipy IIR filter to mono OR stereo audio.
    Handles the minimum-length guard (filtfilt needs >= padlen samples).
    """
    # filtfilt needs at least 3*(max(len(b), len(a))-1)+1 samples
    min_len = 3 * (max(len(b_coeff), len(a_coeff)) - 1) + 1
    if len(audio) < min_len:
        return audio.copy()

    if _is_stereo(audio):
        left  = signal.filtfilt(b_coeff, a_coeff, audio[:, 0]).astype(np.float32)
        right = signal.filtfilt(b_coeff, a_coeff, audio[:, 1]).astype(np.float32)
        return np.stack([left, right], axis=1)
    else:
        return signal.filtfilt(b_coeff, a_coeff, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# HPF / LPF — single-shot Butterworth
# ---------------------------------------------------------------------------

def apply_hpf(audio, sr, cutoff_hz):
    """
    High-pass filter.  Removes everything below cutoff_hz.

    Edge cases handled:
      - cutoff_hz <= 0          → return audio unchanged (no filtering)
      - cutoff_hz >= Nyquist    → return silence (all energy above Nyquist)
      - audio too short to filter → return audio unchanged
    """
    nyq = 0.5 * sr
    if cutoff_hz <= 0:
        return audio.copy()
    if cutoff_hz >= nyq:
        return np.zeros_like(audio)
    b, a = signal.butter(4, cutoff_hz / nyq, btype='high')
    return _apply_filter_to_audio(audio, b, a)


def apply_lpf(audio, sr, cutoff_hz):
    """
    Low-pass filter.  Removes everything above cutoff_hz.

    Edge cases handled:
      - cutoff_hz <= 0          → return silence
      - cutoff_hz >= Nyquist    → return audio unchanged
      - audio too short to filter → return audio unchanged
    """
    nyq = 0.5 * sr
    if cutoff_hz <= 0:
        return np.zeros_like(audio)
    if cutoff_hz >= nyq:
        return audio.copy()
    b, a = signal.butter(4, cutoff_hz / nyq, btype='low')
    return _apply_filter_to_audio(audio, b, a)


def apply_bandpass(audio, sr, low=300, high=4000):
    """Band-pass filter between low and high Hz."""
    nyq = 0.5 * sr
    low  = max(1.0, min(low,  nyq * 0.99))
    high = max(low + 1.0, min(high, nyq * 0.99))
    b, a = signal.butter(2, [low / nyq, high / nyq], btype='band')
    return _apply_filter_to_audio(audio, b, a)


# ---------------------------------------------------------------------------
# LUFS-based loudness measurement (ITU-R BS.1770 approximation)
# ---------------------------------------------------------------------------

def measure_lufs(audio, sr):
    """
    Measure integrated loudness in LUFS using a K-weighted approximation.

    Returns a float in dBFS (negative number; 0 dBFS = full scale).
    Returns -70.0 (digital floor) for silence or very short clips.

    Edge cases:
      - silence / all-zeros        → -70.0 LUFS
      - audio shorter than 400ms   → use full clip (momentary window)
      - NaN / Inf in audio         → clamp to ±1.0 first
    """
    DIGITAL_FLOOR = -70.0

    # Sanitise
    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    mono = _to_mono(audio)

    if len(mono) == 0 or np.max(np.abs(mono)) < 1e-9:
        return DIGITAL_FLOOR

    # Stage 1 — pre-filter (high-shelf boost at ~1.5 kHz, approximated)
    b_pre = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285])
    a_pre = np.array([1.0,              -1.69065929318241, 0.73248077421585])
    if len(mono) >= 3 * max(len(b_pre), len(a_pre)):
        mono = signal.filtfilt(b_pre, a_pre, mono).astype(np.float32)

    # Stage 2 — RLB weight (high-pass at ~38 Hz)
    b_rlb = np.array([1.0, -2.0, 1.0])
    a_rlb = np.array([1.0, -1.99004745483398, 0.99007225036621])
    if len(mono) >= 3 * max(len(b_rlb), len(a_rlb)):
        mono = signal.filtfilt(b_rlb, a_rlb, mono).astype(np.float32)

    # Gated mean-square (simplified: single gate at -70 LUFS)
    ms = np.mean(mono ** 2)
    if ms < 1e-10:
        return DIGITAL_FLOOR

    lufs = 10.0 * np.log10(ms) - 0.691  # BS.1770 offset
    return float(np.clip(lufs, DIGITAL_FLOOR, 0.0))


def compute_gain_match(a_lufs, b_lufs):
    """
    Return a linear gain scalar to apply to Track B so its loudness
    matches Track A.

    Rules:
      - NEVER boost above 1.0 (0 dB) — only duck or leave as-is
      - Minimum scalar is 0.25 (−12 dB) to prevent extreme ducking
      - If A is quieter than B, duck A instead — caller handles that

    Returns: float in [0.25, 1.0]
    """
    diff_db = a_lufs - b_lufs          # positive = B is quieter → needs boost
    gain = 10.0 ** (diff_db / 20.0)   # convert dB difference to linear
    return float(np.clip(gain, 0.25, 1.0))


# ---------------------------------------------------------------------------
# Echo tail — stereo-aware
# ---------------------------------------------------------------------------

def generate_echo_tail(audio_chunk, sr, bpm, beats=4):
    """
    BPM-synced feedback delay.  Returns a tail-only array (same shape as input).

    Edge cases:
      - empty / silent chunk → return zeros of correct shape
      - bpm <= 0             → use 120 BPM as fallback
    """
    if bpm <= 0:
        bpm = 120.0

    if len(audio_chunk) == 0:
        return np.zeros((int((60.0 / bpm) * sr * beats),), dtype=np.float32)

    is_silent = np.max(np.abs(audio_chunk)) < 1e-6
    stereo = _is_stereo(audio_chunk)

    delay_samples = int((60.0 / bpm) * sr)
    tail_length   = delay_samples * beats

    if stereo:
        out = np.zeros((len(audio_chunk) + tail_length, 2), dtype=np.float32)
    else:
        out = np.zeros(len(audio_chunk) + tail_length, dtype=np.float32)

    if not is_silent:
        out[:len(audio_chunk)] = audio_chunk
        decay = 0.6
        for b_idx in range(1, beats + 1):
            start = b_idx * delay_samples
            end   = start + len(audio_chunk)
            if end <= len(out):
                out[start:end] += audio_chunk * (decay ** b_idx)

    # Return tail only (strip the original snippet)
    return out[len(audio_chunk):]


# ---------------------------------------------------------------------------
# White noise riser
# ---------------------------------------------------------------------------

def generate_white_noise_sweep(length_samples, sr):
    """Synthesised HPF-swept white noise riser for pre-drop tension."""
    noise = np.random.normal(0, 1, length_samples).astype(np.float32)
    fade_in  = np.linspace(0.0, 1.0, length_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, length_samples, dtype=np.float32)
    envelope = fade_in * fade_out * 4.0
    nyq = 0.5 * sr
    out_noise  = np.zeros_like(noise)
    chunk_size = 2048
    for i in range(0, len(noise), chunk_size):
        end      = min(i + chunk_size, len(noise))
        progress = i / max(1, len(noise))
        cutoff   = min(400 + progress * 8000, nyq * 0.99)
        b, a = signal.butter(2, cutoff / nyq, btype='low')
        chunk = noise[i:end]
        if end - i > 12:
            out_noise[i:end] = signal.filtfilt(b, a, chunk)
    return (out_noise * envelope * 0.12).astype(np.float32)


# ---------------------------------------------------------------------------
# Sidechain duck
# ---------------------------------------------------------------------------

def apply_sidechain_duck(instrumental, vocal, sr, hop=512):
    """
    Duck 'instrumental' whenever 'vocal' is loud.
    Works on mono arrays; caller should apply per-channel if stereo.
    """
    mono_vocal = _to_mono(vocal)
    mono_inst  = _to_mono(instrumental)
    vocal_rms  = librosa.feature.rms(y=mono_vocal, frame_length=2048, hop_length=hop)[0]
    if np.max(vocal_rms) > 0:
        vocal_rms = vocal_rms / np.max(vocal_rms)
    duck_gain = np.ones(len(mono_inst), dtype=np.float32)
    for i, r in enumerate(vocal_rms):
        start = i * hop
        end   = min(start + hop, len(mono_inst))
        if r > 0.05:
            duck_gain[start:end] = 0.3
    smooth_samples = int(0.01 * sr)
    duck_gain = uniform_filter1d(duck_gain, size=smooth_samples)
    return mono_inst * duck_gain


# ---------------------------------------------------------------------------
# HPF sweep (reveal effect)
# ---------------------------------------------------------------------------

def apply_hpf_sweep(audio, sr, start_freq=2000, end_freq=80, num_chunks=32):
    """
    Gradually sweep an HPF from start_freq down to end_freq.
    Used to 'reveal' Track B (e.g. vocals sweeping in).
    Handles mono and stereo.
    """
    if len(audio) == 0:
        return audio.copy()

    out        = np.zeros_like(audio)
    chunk_size = max(1, len(audio) // num_chunks)
    freqs      = np.logspace(np.log10(max(start_freq, 1)),
                             np.log10(max(end_freq,   1)), num_chunks)
    nyq = 0.5 * sr

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx   = start_idx + chunk_size if i < num_chunks - 1 else len(audio)
        chunk     = audio[start_idx:end_idx]
        freq      = freqs[i]
        norm_cut  = freq / nyq

        if 0.0 < norm_cut < 1.0:
            b, a = signal.butter(4, norm_cut, btype='high', analog=False)
            out[start_idx:end_idx] = _apply_filter_to_audio(chunk, b, a)
        else:
            out[start_idx:end_idx] = chunk

    return out


# ---------------------------------------------------------------------------
# Loop source detection (Whisper + onset fallback)
# ---------------------------------------------------------------------------

def find_best_loop_source(track_data, drop_sample, bpm, sr):
    """
    Use Whisper to find the last cleanly-bounded word before the drop.
    Falls back to onset detection if Whisper unavailable or finds nothing.
    """
    vocal_stem = _to_mono(track_data["vocals"])
    samples_per_beat   = int((60.0 / bpm) * sr)
    max_word_duration  = 60.0 / bpm

    search_start = max(0, drop_sample - (16 * samples_per_beat))
    search_end   = drop_sample
    search_audio = vocal_stem[search_start:search_end]

    try:
        whisper_sr          = 16000
        search_region_16k   = librosa.resample(search_audio, orig_sr=sr, target_sr=whisper_sr)
        import whisper
        model  = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, search_region_16k,
                                    language="en", word_timestamps=True)
        words  = [w for seg in result.get("segments", [])
                    for w in seg.get("words", [])]

        valid = []
        for w in words:
            ws = search_start + int(w["start"] * sr)
            we = search_start + int(w["end"]   * sr)
            dur = w["end"] - w["start"]
            if 0.1 <= dur <= max_word_duration and we <= drop_sample:
                valid.append((w, ws, we, dur))

        if valid:
            valid.sort(key=lambda x: x[3], reverse=True)
            _, ws, we, dur = valid[0]
            print(f"  Whisper: '{valid[0][0]['word'].strip()}' "
                  f"{valid[0][0]['start']:.2f}s–{valid[0][0]['end']:.2f}s")
            return ws, we

        print(f"  Whisper found {len(words)} words; none met criteria.")

    except Exception as e:
        print(f"  Whisper failed: {e}")

    print("  Falling back to onset detection…")
    onsets = librosa.onset.onset_detect(
        y=search_audio, sr=sr, units='samples', backtrack=True)
    if len(onsets) == 0:
        return search_start, search_start + samples_per_beat

    onset_sample = search_start + onsets[-1]
    return onset_sample, onset_sample + samples_per_beat


# ---------------------------------------------------------------------------
# Instrumental intro finder
# ---------------------------------------------------------------------------

def find_instrumental_intro(track_data, sr):
    """
    Find a region in Track B's intro that has instrumental energy but few vocals.
    Falls back to 32 beats before the drop.
    """
    drop_idx = track_data.get("drop_idx", 0)
    beats    = track_data.get("beats", np.array([]))
    bpm      = track_data.get("bpm", 120)
    spb      = int((60.0 / bpm) * sr)

    for s in track_data.get("segments", []):
        if "intro" in s["label"].lower():
            start = s["start_sample"]
            end   = min(s["end_sample"], start + int(20 * sr))
            if end > start + sr:
                vocal_chunk = _to_mono(track_data["vocals"][start:end])
                inst_chunk  = (_to_mono(track_data["drums"][start:end]) +
                               _to_mono(track_data["other"][start:end]))
                vocal_rms = np.mean(librosa.feature.rms(y=vocal_chunk)) if len(vocal_chunk) > 0 else 0
                inst_rms  = np.mean(librosa.feature.rms(y=inst_chunk))  if len(inst_chunk)  > 0 else 0
                if inst_rms > 0.005 and vocal_rms < 0.02:
                    print(f"  Instrumental intro at sample {start} "
                          f"(inst_rms={inst_rms:.4f}, vocal_rms={vocal_rms:.4f})")
                    return start

    if len(beats) > 0 and drop_idx > 0:
        fallback = int(beats[max(0, drop_idx - 32)])
        print(f"  No clean intro — falling back to 32 beats before drop: {fallback}")
        return fallback
    return 0


# ---------------------------------------------------------------------------
# Vocal cutoff finder (Whisper + beat-snap)
# ---------------------------------------------------------------------------

def find_vocal_cutoff_in_buildup(track_data, build_start, drop_sample, sr):
    """
    Find the last complete word in the 8 beats before the drop, then snap
    the cutoff to the nearest beat boundary.
    Falls back to 2 beats before the drop.
    """
    if build_start >= drop_sample:
        return drop_sample

    beats = track_data["beats"]
    spb   = int((60.0 / track_data["bpm"]) * sr)

    search_start = max(build_start, drop_sample - 8 * spb)
    search_end   = drop_sample
    search_audio = _to_mono(track_data["vocals"][search_start:search_end])

    try:
        whisper_sr        = 16000
        search_region_16k = librosa.resample(search_audio, orig_sr=sr, target_sr=whisper_sr)
        import whisper
        model  = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, search_region_16k,
                                    language="en", word_timestamps=True)
        words  = [w for seg in result.get("segments", [])
                    for w in seg.get("words", [])]

        valid = []
        for w in words:
            ws = search_start + int(w["start"] * sr)
            we = search_start + int(w["end"]   * sr)
            if we <= drop_sample - spb:
                valid.append((w, ws, we))

        if valid:
            last_word, _, w_end = valid[-1]
            print(f"  Vocal cutoff: '{last_word['word'].strip()}' ending at {w_end}")
            if len(beats) > 0:
                nearest = int(beats[np.argmin(np.abs(beats - w_end))])
                snapped = int(np.clip(nearest, build_start, drop_sample - spb))
                print(f"  Snapped to beat at {snapped}")
                return snapped
            return w_end

    except Exception as e:
        print(f"  Whisper vocal cutoff failed: {e}")

    fallback = max(build_start, drop_sample - 2 * spb)
    print(f"  Vocal cutoff fallback: 2 beats before drop at {fallback}")
    return fallback
