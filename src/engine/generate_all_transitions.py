"""
generate_all_transitions.py — Cloudy DJ 2.0

Generates Bass Swap and Treble Swap transitions between two tracks.

All audio is processed in stereo (N, 2) float32 throughout.
Time-stretching is applied to the MIXED signal (not individual stems) to
preserve phase coherence between stems.  Beat grids after warping are
computed mathematically from BPM — NOT re-detected from the smeared audio.
Loudness matching uses LUFS (BS.1770 approximation) instead of raw RMS.
"""

import os
import json
import librosa
import numpy as np
import soundfile as sf
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.engine import dsp_utils
from src.engine.loop_roll import generate_loop_roll

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRATE_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "library", "crate.json")
)

STRUCT_SR    = 22050
STEM_SR      = 44100
STRUCT_SCALE = STEM_SR / STRUCT_SR

MAX_WARP_RATE = 2.0    # librosa's phase vocoder breaks above 2×
MIN_WARP_RATE = 0.5    # and below 0.5×


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_audio(path):
    """Load any audio file as stereo (N, 2) float32."""
    y, sr = sf.read(path, dtype='float32')
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)   # mono → stereo
    elif y.shape[1] > 2:
        y = y[:, :2]                   # surround → take L+R only
    return y, sr


def _to_mono(audio):
    """Return a 1-D view for algorithms that require mono."""
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)


def normalize(audio, headroom=0.85):
    """Peak-normalize stereo or mono audio to headroom."""
    peak = np.max(np.abs(audio))
    if peak > headroom:
        audio = audio * (headroom / peak)
    return audio


# ---------------------------------------------------------------------------
# Track loading
# ---------------------------------------------------------------------------

def detect_key(audio, sr):
    chroma     = librosa.feature.chroma_cqt(y=_to_mono(audio), sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    classes    = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return classes[np.argmax(chroma_sum)]


def detect_first_drop(bass_mono, beats, sr, bpm):
    """Bass RMS contrast scan between 30 s and 120 s."""
    bass_rms      = librosa.feature.rms(y=bass_mono, frame_length=4096, hop_length=1024)[0]
    smoothed_rms  = np.convolve(bass_rms, np.ones(43) / 43, mode='valid')
    beats_per_sec = bpm / 60.0
    frames_per_sec = sr / 1024.0
    frames_8_bars = int(32 / beats_per_sec * frames_per_sec)

    best_frame, max_contrast = 0, 0
    start_f = int(30  * frames_per_sec)
    end_f   = int(120 * frames_per_sec)

    for i in range(start_f, min(end_f, len(smoothed_rms))):
        pre  = np.mean(smoothed_rms[max(0, i - frames_8_bars):i])
        post = np.mean(smoothed_rms[i:min(len(smoothed_rms), i + frames_8_bars)])
        contrast = post - pre
        if contrast > max_contrast:
            max_contrast = contrast
            best_frame   = i

    if best_frame == 0:
        return np.argmin(np.abs(beats - (60 * sr)))  # fallback: 60 s

    drop_sample = librosa.frames_to_samples(best_frame, hop_length=1024)
    return np.argmin(np.abs(beats - drop_sample))


def get_track_data(crate, track_name):
    full_name = next(k for k in crate.keys() if track_name in k)
    data      = crate[full_name]
    stems     = data["stems"]

    vocals, sr = load_audio(stems["vocals"])
    drums,  _  = load_audio(stems["drums"])
    bass,   _  = load_audio(stems["bass"])
    other,  _  = load_audio(stems["other"])

    mono = _to_mono(drums + bass + other)
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=data["bpm"], units='samples')

    key          = detect_key(other + bass, sr)
    drop_beat_idx = detect_first_drop(_to_mono(bass), beats, sr, data["bpm"])

    segs_raw = data.get("segments", [])
    segments = [
        {
            "label":        s["label"],
            "start_sample": int(s["start_sample"] * STRUCT_SCALE),
            "end_sample":   int(s["end_sample"]   * STRUCT_SCALE),
            "energy":       s["energy"],
        }
        for s in segs_raw
    ]

    return {
        "name":     full_name,
        "bpm":      data["bpm"],
        "key":      key,
        "vocals":   vocals,
        "drums":    drums,
        "bass":     bass,
        "other":    other,
        "beats":    beats,
        "drop_idx": drop_beat_idx,
        "sr":       sr,
        "segments": segments,
    }


def find_segment(track_data, label):
    for s in track_data["segments"]:
        if s["label"] == label:
            return s
    return None


# ---------------------------------------------------------------------------
# Phase-coherent time warping  (Bug 1 & 2 fix)
# ---------------------------------------------------------------------------

def apply_warping(a, b_bpm):
    """
    Time-stretch Track A to match Track B's BPM.

    KEY CHANGES vs old code:
      1. We stretch the FULL MIX once to measure the actual warp ratio,
         then apply that same ratio to each stem individually.
         This keeps all stems phase-aligned with each other.
      2. Beat grid after warping is computed MATHEMATICALLY from b_bpm,
         not re-detected from the phase-smeared audio.
      3. Drop sample is scaled by the measured warp ratio, then snapped
         to the nearest mathematical beat.

    Edge cases:
      - rate == 1.0  → skip stretching entirely
      - rate > 2.0   → clamped with warning
      - rate < 0.5   → clamped with warning
    """
    rate = a["bpm"] / b_bpm
    sr   = a["sr"]

    if abs(rate - 1.0) < 0.001:
        # Same BPM — just copy everything, no stretching needed
        return {**a, "bpm": b_bpm}

    if rate > MAX_WARP_RATE:
        print(f"  WARNING: warp rate {rate:.2f} clamped to {MAX_WARP_RATE} "
              f"(A={a['bpm']} BPM, B={b_bpm} BPM)")
        rate = MAX_WARP_RATE
    elif rate < MIN_WARP_RATE:
        print(f"  WARNING: warp rate {rate:.2f} clamped to {MIN_WARP_RATE}")
        rate = MIN_WARP_RATE

    # ------------------------------------------------------------------
    # Step 1: Stretch the full mix once to find the ACTUAL ratio
    # (librosa's STFT may not produce exactly len/rate samples)
    # ------------------------------------------------------------------
    mix_mono     = _to_mono(a["vocals"] + a["drums"] + a["bass"] + a["other"])
    mix_warped   = librosa.effects.time_stretch(mix_mono, rate=rate)
    actual_rate  = len(mix_mono) / len(mix_warped)  # true ratio after vocoder

    # ------------------------------------------------------------------
    # Step 2: Stretch each stem with the SAME actual_rate
    # Because actual_rate came from one vocoder call, all stems will now
    # be the same length and phase-aligned when re-summed.
    # ------------------------------------------------------------------
    def stretch_stem(stem_audio):
        mono    = _to_mono(stem_audio)
        warped  = librosa.effects.time_stretch(mono, rate=actual_rate)
        return np.stack([warped, warped], axis=1).astype(np.float32)

    vocals_w = stretch_stem(a["vocals"])
    drums_w  = stretch_stem(a["drums"])
    bass_w   = stretch_stem(a["bass"])
    other_w  = stretch_stem(a["other"])

    # Hard-truncate all stems to the same length to prevent cumulative drift
    # (the vocoder is non-deterministic — each call can differ by ±50 samples)
    min_stem_len = min(len(vocals_w), len(drums_w), len(bass_w), len(other_w))
    vocals_w = vocals_w[:min_stem_len]
    drums_w  = drums_w[:min_stem_len]
    bass_w   = bass_w[:min_stem_len]
    other_w  = other_w[:min_stem_len]

    # ------------------------------------------------------------------
    # Step 3: Beat grid — MATHEMATICAL, not audio-detected
    # ------------------------------------------------------------------
    spb_warped  = int((60.0 / b_bpm) * sr)
    num_beats   = len(a["beats"])
    beats_warped = np.array([i * spb_warped for i in range(num_beats)], dtype=np.int64)

    # ------------------------------------------------------------------
    # Step 4: Drop index — scale original drop sample, snap to nearest beat
    # ------------------------------------------------------------------
    orig_drop_sample   = int(a["beats"][a["drop_idx"]])
    warped_drop_sample = int(orig_drop_sample / actual_rate)
    drop_idx_warped    = int(np.argmin(np.abs(beats_warped - warped_drop_sample)))

    # Edge case: if track is too short after warping (< 4 beats), warn
    min_samples = 4 * spb_warped
    if len(vocals_w) < min_samples:
        print(f"  WARNING: track '{a['name']}' is only {len(vocals_w)/sr:.1f}s after warping — "
              f"may produce a very short transition.")

    # Scale segment boundaries
    segments_warped = [
        {**s,
         "start_sample": int(s["start_sample"] / actual_rate),
         "end_sample":   int(s["end_sample"]   / actual_rate)}
        for s in a.get("segments", [])
    ]

    return {
        "name":     a["name"],
        "bpm":      b_bpm,
        "key":      a["key"],
        "vocals":   vocals_w,
        "drums":    drums_w,
        "bass":     bass_w,
        "other":    other_w,
        "beats":    beats_warped,
        "drop_idx": drop_idx_warped,
        "sr":       sr,
        "segments": segments_warped,
    }


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _lufs_gain(a_section, b_section, sr):
    """
    Compute a safe linear gain to apply to b_section so its LUFS matches
    a_section.  Returns a scalar in [0.25, 1.0].
    """
    a_lufs = dsp_utils.measure_lufs(a_section, sr)
    b_lufs = dsp_utils.measure_lufs(b_section, sr)

    # Fallback: if the measurement section is silence, LUFS = floor → gain = 1.0
    if a_lufs <= -60 or b_lufs <= -60:
        return 1.0

    return dsp_utils.compute_gain_match(a_lufs, b_lufs)


def _safe_beat(beats, idx):
    """Return the sample at beat index idx, clamped to array bounds."""
    return int(beats[min(max(idx, 0), len(beats) - 1)])


def _write_len(out, start, stem, stem_start, stem_end):
    """How many samples we can safely write without exceeding buffers."""
    available_out  = len(out) - start
    available_stem = min(stem_end, len(stem)) - stem_start
    return max(0, min(available_out, available_stem))


# ---------------------------------------------------------------------------
# T1 — Bass Swap Transition
# ---------------------------------------------------------------------------

def generate_bass_swap_transition(track_a_str, track_b_str, out_name):
    """
    Bass Swap with Whisper vocal awareness:
      • A plays including its drop.
      • 8 beats AFTER A's drop: A's bass hard-cuts to 0.
        B's bass comes in FULL immediately.
      • A's vocals are cut at the last complete word before the swap (Whisper).
      • B's vocals enter at B's first detected word (Whisper).
      • A's mids/highs fade out over 16 beats (equal-power curve).
      • B's mids/highs fade in over 16 beats (equal-power curve).
      • White noise sweep masks the swap point.
    """
    print(f"\n[Bass Swap] {track_a_str} → {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    a_drop  = _safe_beat(a["beats"], a["drop_idx"])
    b_drop  = _safe_beat(b["beats"], b["drop_idx"])

    # ------------------------------------------------------------------
    # Swap anchor: 8 beats AFTER A's drop
    # ------------------------------------------------------------------
    swap_beat_offset = 8
    max_offset = (len(a["bass"]) - a_drop) // spb - 2
    swap_beat_offset = min(swap_beat_offset, max(max_offset, 0))

    a_swap = _safe_beat(a["beats"], a["drop_idx"] + swap_beat_offset)
    b_swap = max(b_drop, b["beats"][0] if len(b["beats"]) > 0 else 0)

    # ------------------------------------------------------------------
    # Whisper vocal boundaries
    # ------------------------------------------------------------------
    # A: find last complete word before swap → cut vocals there
    a_vocal_cut = dsp_utils.find_vocal_cutoff_in_buildup(
        a, max(0, a_swap - 8 * spb), a_swap, sr)
    print(f"  A vocal cut at sample {a_vocal_cut} ({a_vocal_cut/sr:.2f}s)")

    # B: find first vocal onset after B's swap point
    b_vocal_entry = dsp_utils.find_vocal_entry(b["vocals"], b_swap, sr)
    print(f"  B vocal entry at sample {b_vocal_entry} ({b_vocal_entry/sr:.2f}s)")

    pre_len   = int(10 * sr)
    blend_len = 16 * spb
    post_len  = int(10 * sr)
    total_len = pre_len + blend_len + post_len

    out = np.zeros((total_len, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Pre: 10 s of A at full energy
    # ------------------------------------------------------------------
    a_pre_start  = max(0, a_swap - pre_len)
    actual_pre   = a_swap - a_pre_start
    out_pre_off  = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_swap)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]
        # A vocals: play only up to the Whisper cut point
        vocal_end_in_pre = min(a_vocal_cut, a_swap)
        if vocal_end_in_pre > a_pre_start:
            wl = _write_len(out, out_pre_off, a["vocals"], a_pre_start, vocal_end_in_pre)
            vocal_chunk = a["vocals"][a_pre_start:a_pre_start + wl].copy()
            # 100ms fade out to avoid a hard snap/pop at the cut point
            fade_samples = min(int(0.1 * sr), len(vocal_chunk))
            if fade_samples > 0:
                fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
                vocal_chunk[-fade_samples:] *= fade
            out[out_pre_off:out_pre_off + wl] += vocal_chunk

    # Add echo tail to A's cut vocal
    if a_vocal_cut > spb:
        vocal_snip = a["vocals"][a_vocal_cut - spb:a_vocal_cut]
        echo_tail  = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
        # Find where a_vocal_cut lands in the output buffer
        out_vocal_cut = pre_len - (a_swap - a_vocal_cut)
        tail_len = min(len(echo_stereo), total_len - out_vocal_cut)
        if tail_len > 0 and out_vocal_cut >= 0:
            out[out_vocal_cut:out_vocal_cut + tail_len] += echo_stereo[:tail_len] * 0.7

    # White noise sweep centred on the swap point to mask the cut
    noise_len    = min(int(1.5 * spb), pre_len // 2)
    noise_mono   = dsp_utils.generate_white_noise_sweep(noise_len, sr)
    noise_stereo = np.stack([noise_mono, noise_mono], axis=1)
    noise_start  = max(0, pre_len - noise_len // 2)
    noise_end    = min(noise_start + noise_len, total_len)
    nw           = noise_end - noise_start
    if nw > 0:
        out[noise_start:noise_end] += noise_stereo[:nw]

    # ------------------------------------------------------------------
    # Blend zone: 16 beats — equal-power crossfade
    # ------------------------------------------------------------------
    n = blend_len
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)[:, None]  # equal-power out: cos²-like energy
    fade_in  = np.sin(t)[:, None]  # equal-power in:  sin²-like energy

    # A mids/highs: fade out (bass is ZERO — that's the swap)
    for stem in ["drums", "other"]:
        src_end = min(a_swap + n, len(a[stem]))
        wl      = src_end - a_swap
        if wl > 0:
            out[pre_len:pre_len + wl] += a[stem][a_swap:src_end] * fade_out[:wl]
    # A vocals already cut by Whisper — don't add them in the blend

    # LUFS gain match: measure A's full-energy drop vs B's drop
    a_measure = a["bass"][max(0, a_swap - 4 * spb):a_swap] + \
                a["drums"][max(0, a_swap - 4 * spb):a_swap]
    b_measure = b["bass"][b_swap:min(b_swap + 4 * spb, len(b["bass"]))] + \
                b["drums"][b_swap:min(b_swap + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_measure, b_measure, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    # B bass: FULL from beat 0 (the swap itself)
    b_bass_end = min(b_swap + n, len(b["bass"]))
    bbl        = b_bass_end - b_swap
    if bbl > 0:
        out[pre_len:pre_len + bbl] += b["bass"][b_swap:b_bass_end] * gain

    # B drums/other: fade in over 16 beats
    for stem in ["drums", "other"]:
        src_end = min(b_swap + n, len(b[stem]))
        wl      = src_end - b_swap
        if wl > 0:
            out[pre_len:pre_len + wl] += b[stem][b_swap:src_end] * fade_in[:wl] * gain

    # B vocals: enter at Whisper-detected word boundary
    b_vocal_delay_samples = max(0, b_vocal_entry - b_swap)
    b_vocal_blend_len     = max(0, n - b_vocal_delay_samples)
    if b_vocal_blend_len > 0:
        # Gentle 2-beat fade-in at the vocal entry point
        v_fade_len = min(2 * spb, b_vocal_blend_len)
        v_fade = np.concatenate([
            np.linspace(0.0, 1.0, v_fade_len, dtype=np.float32),
            np.ones(b_vocal_blend_len - v_fade_len, dtype=np.float32)
        ])[:, None]
        src_start = b_vocal_entry
        src_end   = min(src_start + b_vocal_blend_len, len(b["vocals"]))
        wl        = src_end - src_start
        if wl > 0:
            out_start = pre_len + b_vocal_delay_samples
            out[out_start:out_start + wl] += b["vocals"][src_start:src_end] * v_fade[:wl] * gain

    # ------------------------------------------------------------------
    # Post: 10 s of B after the swap
    # ------------------------------------------------------------------
    b_post_start = b_swap + blend_len
    b_post_end   = min(b_post_start + post_len, len(b["bass"]))
    actual_post  = b_post_end - b_post_start
    if actual_post > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, pre_len + blend_len, b[stem], b_post_start, b_post_end)
            out[pre_len + blend_len:pre_len + blend_len + wl] += \
                b[stem][b_post_start:b_post_start + wl] * gain

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# T2 — Treble Swap (Outro → Intro)
# ---------------------------------------------------------------------------

def generate_treble_swap_transition(track_a_str, track_b_str, out_name):
    """
    Treble Swap — fixed in Round 2:
      • Equal-power crossfade curves (no −6 dB mid-dip)
      • LUFS measured against A's highest-energy section, not its quiet outro
      • b_post_start = b_blend_start + blend_len (not b_blend_start)
      • B's drums HPF'd at 250 Hz to strip kick sub
    """
    print(f"\n[Treble Swap] {track_a_str} → {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    # ------------------------------------------------------------------
    # Find blend start points
    # ------------------------------------------------------------------
    a_outro_seg = find_segment(a, "outro")
    if a_outro_seg:
        a_blend_start = int(a_outro_seg["start_sample"])
    else:
        a_blend_start = max(0, len(a["drums"]) - 32 * spb)
    a_blend_start = int(np.clip(a_blend_start, 0, len(a["drums"]) - 2))

    b_blend_start = dsp_utils.find_instrumental_intro(b, sr)

    blend_beats = 16
    blend_len   = blend_beats * spb

    # ------------------------------------------------------------------
    # LUFS gain: measure A's HIGHEST-ENERGY section (drop), not outro
    # The old code measured A's quiet outro vs B's punchy intro → 0.425× brutal ducking
    # ------------------------------------------------------------------
    a_drop = _safe_beat(a["beats"], a["drop_idx"])
    a_m_start = max(0, a_drop)
    a_m_end   = min(a_drop + 4 * spb, len(a["drums"]))
    b_m_end   = min(b_blend_start + blend_len, len(b["drums"]))
    a_measure = (_to_mono(a["drums"][a_m_start:a_m_end]) +
                 _to_mono(a["bass"] [a_m_start:a_m_end]))
    b_measure = (_to_mono(b["drums"][b_blend_start:b_m_end]) +
                 _to_mono(b["bass"] [b_blend_start:b_m_end]))
    gain = _lufs_gain(a_measure, b_measure, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}× (ref: A's drop, not outro)")

    pre_len  = int(10 * sr)
    post_len = int(10 * sr)
    total    = pre_len + blend_len + post_len
    out      = np.zeros((total, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Pre: 10 s of A
    # ------------------------------------------------------------------
    a_pre_start = max(0, a_blend_start - pre_len)
    actual_pre  = a_blend_start - a_pre_start
    out_pre_off = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_blend_start)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # ------------------------------------------------------------------
    # Blend zone — equal-power crossfade
    # ------------------------------------------------------------------
    a_end = min(a_blend_start + blend_len, len(a["drums"]))
    a_act = a_end - a_blend_start
    b_end = min(b_blend_start + blend_len, len(b["drums"]))
    b_act = b_end - b_blend_start
    n     = min(a_act, b_act, blend_len)

    # Equal-power curves: constant energy throughout the blend
    t        = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    ep_out   = np.cos(t)[:, None]   # A fades out
    ep_in    = np.sin(t)[:, None]   # B fades in

    if n > 0:
        # A: all stems fade out with equal-power curve
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len:pre_len + n] += a[stem][a_blend_start:a_blend_start + n] * ep_out

        # B drums: HPF'd at 250 Hz to strip kick sub, fade in
        b_drums_section = b["drums"][b_blend_start:b_blend_start + n]
        b_drums_hpf     = dsp_utils.apply_hpf(b_drums_section, sr, cutoff_hz=250)
        out[pre_len:pre_len + n] += b_drums_hpf * ep_in * gain

        # B other: fade in
        out[pre_len:pre_len + n] += b["other"][b_blend_start:b_blend_start + n] * ep_in * gain

        # B vocals: HPF reveal sweep (2000→200 Hz over the blend)
        b_vocals_section = b["vocals"][b_blend_start:b_blend_start + n]
        b_vocals_revealed = dsp_utils.apply_hpf_sweep(b_vocals_section, sr,
                                                       start_freq=2000, end_freq=200,
                                                       num_chunks=16)
        out[pre_len:pre_len + n] += b_vocals_revealed * ep_in * gain

        # B bass is deliberately muted during the blend to avoid clashing with A's bass.
        # It will slam in immediately after the blend.

    # Echo tail from A's last vocal
    vocal_snip = a["vocals"][max(0, a_blend_start - spb):a_blend_start]
    echo_tail  = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
    tail_len   = min(len(echo_tail), blend_len)
    if tail_len > 0:
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
        out[pre_len:pre_len + tail_len] += echo_stereo[:tail_len] * 0.4

    # ------------------------------------------------------------------
    # Post: 10 s of B AFTER the blend (BUG FIX: was replaying from intro)
    # ------------------------------------------------------------------
    b_post_start = b_blend_start + blend_len   # ← FIXED (was b_blend_start)
    b_post_end   = min(b_post_start + post_len, len(b["drums"]))
    actual_post  = b_post_end - b_post_start

    if actual_post > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, pre_len + blend_len, b[stem], b_post_start, b_post_end)
            out[pre_len + blend_len:pre_len + blend_len + wl] += \
                b[stem][b_post_start:b_post_start + wl] * gain

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# T3 — Drop Swap + Echo Out + Tension Gap
# ---------------------------------------------------------------------------

def generate_drop_swap(track_a_str, track_b_str, out_name):
    """
    Hard-cut A before its drop, 1-beat tension gap with reverb tail,
    then B's drop SLAMS in.  Stereo (N,2) throughout.
    """
    print(f"\n[Drop Swap] {track_a_str} → {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    a_drop = _safe_beat(a["beats"], a["drop_idx"])
    b_drop = _safe_beat(b["beats"], b["drop_idx"])

    gap_len  = spb              # 1-beat tension gap
    pre_len  = int(10 * sr)
    post_len = int(10 * sr)
    total    = pre_len + gap_len + post_len
    out      = np.zeros((total, 2), dtype=np.float32)

    # Pre: 10 s of A up to drop
    a_pre_start = max(0, a_drop - pre_len)
    actual_pre  = a_drop - a_pre_start
    off         = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, off, a[stem], a_pre_start, a_drop)
            out[off:off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # ------------------------------------------------------------------
    # Tension gap: 1 beat of white noise riser + reverb tail from A
    # ------------------------------------------------------------------
    last_beat  = max(0, a_drop - spb)
    vocal_snip = a["vocals"][last_beat:a_drop]
    reverb_tail = dsp_utils.generate_reverb_tail(vocal_snip, sr, decay_seconds=1.0)
    
    # Generate the EDM tension riser
    riser = dsp_utils.generate_tension_riser(gap_len, sr)
    riser_stereo = np.stack([riser, riser], axis=1)
    
    rt_len = min(len(reverb_tail), gap_len)
    if rt_len > 0:
        if reverb_tail.ndim == 1:
            reverb_stereo = np.stack([reverb_tail, reverb_tail], axis=1)
        else:
            reverb_stereo = reverb_tail
        out[pre_len:pre_len + rt_len] += reverb_stereo[:rt_len] * 0.5
        
    out[pre_len:pre_len + gap_len] += riser_stereo * 0.7

    # LUFS match: Measure A's Drop vs B's Drop
    a_m = a["bass"][a_drop:min(a_drop + 4 * spb, len(a["bass"]))] + \
          a["drums"][a_drop:min(a_drop + 4 * spb, len(a["drums"]))]
    b_m = b["bass"][b_drop:min(b_drop + 4 * spb, len(b["bass"]))] + \
          b["drums"][b_drop:min(b_drop + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_m, b_m, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    # Post: B from its drop (after the gap)
    b_start_out = pre_len + gap_len
    b_end       = min(b_drop + post_len, len(b["bass"]))
    wl          = b_end - b_drop
    if wl > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[b_start_out:b_start_out + wl] += b[stem][b_drop:b_drop + wl] * gain

    # Echo tail from A's last vocal over B
    echo_tail = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
    tail_len  = min(len(echo_tail), post_len)
    if tail_len > 0:
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
        out[b_start_out:b_start_out + tail_len] += echo_stereo[:tail_len] * 0.3

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# T4 — Loop Roll Tension (Rewritten)
# ---------------------------------------------------------------------------

def generate_loop_roll_tension(track_a_str, track_b_str, out_name):
    """
    Loop roll on A's last vocal word for 8 beats → B's drop.
    Now uses the rewritten loop_roll.py which:
      - Pitch-preserves accelerations via time_stretch
      - Applies HPF sweep for tension
      - Includes reverb buildup
      - Plays A's drums underneath
      - Has a tension gap silence before the drop
    Returns stereo (N,2) directly.
    """
    print(f"\n[Loop Roll] {track_a_str} → {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]

    a_drop = _safe_beat(a["beats"], a["drop_idx"])
    b_drop = _safe_beat(b["beats"], b["drop_idx"])
    a_build_start = _safe_beat(a["beats"], max(0, a["drop_idx"] - 32))
    spb = int((60.0 / b["bpm"]) * sr)

    buildup_len    = a_drop - a_build_start
    roll_len       = 8 * spb
    norm_build_len = max(0, buildup_len - roll_len)

    pre_len  = int(10 * sr)
    post_len = int(10 * sr)
    total    = pre_len + buildup_len + post_len
    out      = np.zeros((total, 2), dtype=np.float32)

    # Pre
    a_pre_start = max(0, a_build_start - pre_len)
    actual_pre  = a_build_start - a_pre_start
    off         = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, off, a[stem], a_pre_start, a_build_start)
            out[off:off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # Buildup (A plays fully for the entire buildup)
    if buildup_len > 0:
        for stem in ["drums", "other", "vocals", "bass"]:
            wl = _write_len(out, pre_len, a[stem], a_build_start, a_drop)
            out[pre_len:pre_len + wl] += a[stem][a_build_start:a_build_start + wl]
            
        # Bass fade out in the last 4 beats
        bass_fade_len = min(buildup_len, 4 * spb)
        if bass_fade_len > 0:
            fade_start = pre_len + buildup_len - bass_fade_len
            fade = np.linspace(1.0, 0.0, bass_fade_len, dtype=np.float32)[:, None]
            # Replace the bass in the last 4 beats with faded bass
            a_bass_section = a["bass"][a_drop - bass_fade_len:a_drop]
            out[fade_start:fade_start + bass_fade_len] -= a_bass_section # subtract original
            out[fade_start:fade_start + bass_fade_len] += a_bass_section * fade # add faded

    # ------------------------------------------------------------------
    # Loop roll — NEW: passes drum stem, returns stereo directly
    # ------------------------------------------------------------------
    word_s, word_e = dsp_utils.find_best_loop_source(
        {"vocals": a["vocals"]}, a_drop, b["bpm"], sr)

    roll_stereo = generate_loop_roll(
        vocal_stem=a["vocals"],
        drum_stem=a["drums"],
        bpm=b["bpm"],
        sr=sr,
        drop_sample_idx=a_drop,
        num_beats=8,
        exact_word_start=word_s,
        exact_word_end=word_e
    )
    roll_start = pre_len + norm_build_len
    roll_wl    = min(len(roll_stereo), roll_len)
    if roll_wl > 0:
        out[roll_start:roll_start + roll_wl] += roll_stereo[:roll_wl]

    # LUFS match: Measure A's Drop vs B's Drop
    a_m = a["bass"][a_drop:min(a_drop + 4 * spb, len(a["bass"]))] + \
          a["drums"][a_drop:min(a_drop + 4 * spb, len(a["drums"]))]
    b_m = b["bass"][b_drop:min(b_drop + 4 * spb, len(b["bass"]))] + \
          b["drums"][b_drop:min(b_drop + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_m, b_m, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    # B post-drop
    b_end      = min(b_drop + post_len, len(b["bass"]))
    wl         = b_end - b_drop
    post_start = pre_len + buildup_len
    if wl > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[post_start:post_start + wl] += b[stem][b_drop:b_drop + wl] * gain

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Edit these to match tracks in your crate.json
    TRACK_A = "Tiesto_Secrets"
    TRACK_B = "James_Hype_Wild"

    a_tag = TRACK_A.split()[-1]
    b_tag = TRACK_B.split()[-1]

    generate_bass_swap_transition(
        TRACK_A, TRACK_B,
        f"1_BassSwap_{a_tag}_to_{b_tag}.wav"
    )
    generate_treble_swap_transition(
        TRACK_A, TRACK_B,
        f"2_TrebleSwap_{a_tag}_to_{b_tag}.wav"
    )
    generate_drop_swap(
        TRACK_A, TRACK_B,
        f"3_DropSwap_{a_tag}_to_{b_tag}.wav"
    )
    generate_loop_roll_tension(
        TRACK_A, TRACK_B,
        f"4_LoopRoll_{a_tag}_to_{b_tag}.wav"
    )

    print("\nAll transitions generated — check the output/ folder.")
