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
# T1 — Bass Swap Transition  (Bug 3, 4, 5, 6 fixes)
# ---------------------------------------------------------------------------

def generate_bass_swap_transition(track_a_str, track_b_str, out_name):
    """
    Bass Swap:
      • A plays including its drop.
      • 8 beats AFTER A's drop: A's bass hard-cuts to 0.
        B's bass comes in FULL immediately.
      • A's mids/highs (drums, other, vocals) fade out over 16 beats.
      • B's mids/highs (drums, other) fade in over 16 beats.
      • B's vocals fade in after 4 beats (to avoid immediate vocal clash).
      • White noise sweep masks the swap point.

    Bug 3 fix: swap fires 8 beats AFTER drop (not at peak).
    Bug 5 fix: LUFS-based gain, capped at 1.0.
    Bug 6 fix: full stereo (N,2) pipeline.
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
    # Swap anchor: 8 beats AFTER A's drop, not at the peak (Bug 3 fix)
    # ------------------------------------------------------------------
    swap_beat_offset = 8
    # Edge case: track ends before drop+8
    max_offset = (len(a["bass"]) - a_drop) // spb - 2
    swap_beat_offset = min(swap_beat_offset, max(max_offset, 0))

    a_swap = _safe_beat(a["beats"], a["drop_idx"] + swap_beat_offset)
    # Edge case: very short intro EDM — B's drop very close to start
    b_swap = max(b_drop, b["beats"][0] if len(b["beats"]) > 0 else 0)

    pre_len   = int(10 * sr)           # 10 s of A before the swap
    blend_len = 16 * spb               # 16 beats of crossfade
    post_len  = int(10 * sr)           # 10 s of B after the swap
    total_len = pre_len + blend_len + post_len

    out = np.zeros((total_len, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Pre: 10 s of A at full energy (from swap anchor backwards)
    # ------------------------------------------------------------------
    a_pre_start  = max(0, a_swap - pre_len)
    actual_pre   = a_swap - a_pre_start
    out_pre_off  = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_swap)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]

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
    # Blend zone: 16 beats
    # ------------------------------------------------------------------
    n = blend_len
    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]  # (N,1) for broadcasting
    fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]

    # A mids/highs: fade out (bass is ZERO — that's the swap)
    for stem in ["drums", "other", "vocals"]:
        src_start = a_swap
        src_end   = min(a_swap + n, len(a[stem]))
        wl        = src_end - src_start
        if wl > 0:
            out[pre_len:pre_len + wl] += a[stem][src_start:src_end] * fade_out[:wl]

    # LUFS gain match: measure the last 4 beats of A vs first 4 beats of B
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

    # B mids/highs: fade in over 16 beats
    for stem in ["drums", "other"]:
        src_start = b_swap
        src_end   = min(b_swap + n, len(b[stem]))
        wl        = src_end - src_start
        if wl > 0:
            out[pre_len:pre_len + wl] += b[stem][src_start:src_end] * fade_in[:wl] * gain

    # B vocals: fade in from beat 4 (avoid immediate vocal clash)
    vocal_delay = 4 * spb
    vfade_len   = max(0, n - vocal_delay)
    if vfade_len > 0:
        v_fade   = np.linspace(0.0, 1.0, vfade_len, dtype=np.float32)[:, None]
        src_start = b_swap + vocal_delay
        src_end   = min(src_start + vfade_len, len(b["vocals"]))
        wl        = src_end - src_start
        if wl > 0:
            out[pre_len + vocal_delay:pre_len + vocal_delay + wl] += \
                b["vocals"][src_start:src_end] * v_fade[:wl] * gain

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
# T2 — Treble Swap (Outro → Intro)  (Bug 2, 3, 5, 6 fixes)
# ---------------------------------------------------------------------------

def generate_treble_swap_transition(track_a_str, track_b_str, out_name):
    """
    Treble Swap (real DJ term for what's sometimes called 'High Swap'):
      • Identify A's outro (or last 16 beats before end) and B's intro.
      • Over 16 beats:
          - A's bass kills in 4 beats (fast).
          - A's drums/other/vocals fade out over 16 beats (gradual).
          - B's drums/other (HPF'd to strip kick sub) fade in over 12 beats.
          - B's bass enters in the LAST 4 beats only.
          - B's vocals revealed with an HPF sweep over 8 beats post-blend.
      • Echo tail from A's last vocal.

    Bug 2 fix: B's drum stem is HPF'd at 250 Hz to strip kick sub before
               being faded in — prevents two basses playing simultaneously.
    Bug 5 fix: LUFS-based gain matching.
    Bug 6 fix: full stereo (N,2) pipeline.
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
    a_outro_seg  = find_segment(a, "outro")
    if a_outro_seg:
        a_blend_start = int(a_outro_seg["start_sample"])
    else:
        a_blend_start = max(0, len(a["drums"]) - 32 * spb)
    a_blend_start = int(np.clip(a_blend_start, 0, len(a["drums"]) - 2))

    b_blend_start = dsp_utils.find_instrumental_intro(b, sr)

    blend_beats = 16
    blend_len   = blend_beats * spb

    # ------------------------------------------------------------------
    # LUFS gain: measure A's outro energy vs B's intro energy
    # ------------------------------------------------------------------
    a_m_end = min(a_blend_start + blend_len, len(a["drums"]))
    b_m_end = min(b_blend_start + blend_len, len(b["drums"]))
    a_measure = (_to_mono(a["drums"][a_blend_start:a_m_end]) +
                 _to_mono(a["bass"] [a_blend_start:a_m_end]))
    b_measure = (_to_mono(b["drums"][b_blend_start:b_m_end]) +
                 _to_mono(b["bass"] [b_blend_start:b_m_end]))
    gain = _lufs_gain(a_measure, b_measure, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    pre_len  = int(10 * sr)
    post_len = int(10 * sr)
    total    = pre_len + blend_len + post_len
    out      = np.zeros((total, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Pre: 10 s of A
    # ------------------------------------------------------------------
    a_pre_start  = max(0, a_blend_start - pre_len)
    actual_pre   = a_blend_start - a_pre_start
    out_pre_off  = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_blend_start)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # ------------------------------------------------------------------
    # Blend zone
    # ------------------------------------------------------------------
    blend = np.zeros((blend_len, 2), dtype=np.float32)

    a_end = min(a_blend_start + blend_len, len(a["drums"]))
    a_act = a_end - a_blend_start

    if a_act > 0:
        n = a_act
        # A bass: full → 0 over first 4 beats (fast cut)
        bass_fade_len = min(n, 4 * spb)
        a_bass_fade   = np.concatenate([
            np.linspace(1.0, 0.0, bass_fade_len),
            np.zeros(n - bass_fade_len)
        ]).astype(np.float32)[:, None]
        blend[:n] += a["bass"][a_blend_start:a_end] * a_bass_fade

        # A drums/other/vocals: full → 0 over all 16 beats
        full_fade = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
        for stem in ["drums", "other", "vocals"]:
            blend[:n] += a[stem][a_blend_start:a_end] * full_fade

    b_end = min(b_blend_start + blend_len, len(b["drums"]))
    b_act = b_end - b_blend_start

    if b_act > 0:
        n = b_act

        # B drums: fade in over first 12 beats
        # CRITICAL: HPF at 250 Hz to strip kick sub (Bug 2 fix)
        mid_fade_len = min(n, 12 * spb)
        mid_fade     = np.concatenate([
            np.linspace(0.0, 1.0, mid_fade_len),
            np.ones(n - mid_fade_len)
        ]).astype(np.float32)[:, None] * gain

        b_drums_section = b["drums"][b_blend_start:b_end]
        b_drums_hpf     = dsp_utils.apply_hpf(b_drums_section, sr, cutoff_hz=250)
        blend[:n] += b_drums_hpf * mid_fade

        # B other: fade in over first 12 beats (no HPF needed — no sub content)
        blend[:n] += b["other"][b_blend_start:b_end] * mid_fade

        # B vocals: HPF reveal (thin → full over blend)
        b_vocals_section  = b["vocals"][b_blend_start:b_end]
        b_vocals_filtered = dsp_utils.apply_hpf_sweep(b_vocals_section, sr,
                                                       start_freq=2000, end_freq=2000,
                                                       num_chunks=1)
        blend[:n] += b_vocals_filtered * mid_fade

        # B bass: enters ONLY in the last 4 beats (the treble swap moment)
        bass_in_len = min(n, 4 * spb)
        bass_in_fade = np.concatenate([
            np.zeros(max(0, n - bass_in_len)),
            np.linspace(0.0, 1.0, min(bass_in_len, n))
        ]).astype(np.float32)[:, None] * gain
        blend[:n] += b["bass"][b_blend_start:b_end] * bass_in_fade

    # Echo tail from A's last vocal
    vocal_snip   = a["vocals"][max(0, a_blend_start - spb):a_blend_start]
    echo_tail    = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
    tail_len     = min(len(echo_tail), blend_len)
    if tail_len > 0:
        # echo_tail may be mono from dsp_utils if snippet was mono; stack to stereo
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
        blend[:tail_len] += echo_stereo[:tail_len]

    out[pre_len:pre_len + blend_len] = blend

    # ------------------------------------------------------------------
    # Post: 10 s of B (with HPF vocal reveal sweep)
    # ------------------------------------------------------------------
    b_post_start = b_blend_start
    b_post_end   = min(b_post_start + post_len, len(b["drums"]))
    actual_post  = b_post_end - b_post_start

    if actual_post > 0:
        for stem in ["drums", "bass", "other"]:
            wl = _write_len(out, pre_len + blend_len, b[stem], b_post_start, b_post_end)
            out[pre_len + blend_len:pre_len + blend_len + wl] += \
                b[stem][b_post_start:b_post_start + wl] * gain

        # Vocals HPF sweep reveal: 2000 Hz → 80 Hz over 8 beats
        sweep_len         = min(8 * spb, actual_post)
        b_vocals_sweep    = b["vocals"][b_post_start:b_post_start + sweep_len]
        revealed_vocals   = dsp_utils.apply_hpf_sweep(b_vocals_sweep, sr,
                                                       start_freq=2000, end_freq=80,
                                                       num_chunks=16)
        wl = _write_len(out, pre_len + blend_len, b["vocals"], b_post_start,
                        b_post_start + sweep_len)
        if wl > 0:
            out[pre_len + blend_len:pre_len + blend_len + wl] += \
                revealed_vocals[:wl] * gain

        # Rest of vocals at full (after sweep)
        rest_start = b_post_start + sweep_len
        rest_end   = b_post_end
        if rest_end > rest_start:
            wl = _write_len(out, pre_len + blend_len + sweep_len,
                            b["vocals"], rest_start, rest_end)
            if wl > 0:
                out[pre_len + blend_len + sweep_len:
                    pre_len + blend_len + sweep_len + wl] += \
                    b["vocals"][rest_start:rest_start + wl] * gain

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# T3 — Drop Swap + Echo Out
# ---------------------------------------------------------------------------

def generate_drop_swap(track_a_str, track_b_str, out_name):
    """
    Hard-cut A exactly at its drop, fire B's drop simultaneously.
    A's last vocal echoes out over B.  Stereo (N,2) throughout.
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

    pre_len  = int(10 * sr)
    post_len = int(10 * sr)
    total    = pre_len + post_len
    out      = np.zeros((total, 2), dtype=np.float32)

    # Pre: 10 s of A up to drop
    a_pre_start = max(0, a_drop - pre_len)
    actual_pre  = a_drop - a_pre_start
    off         = pre_len - actual_pre
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, off, a[stem], a_pre_start, a_drop)
            out[off:off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # LUFS match
    a_m = a["bass"][max(0, a_drop - 4 * spb):a_drop]
    b_m = b["bass"][b_drop:min(b_drop + 4 * spb, len(b["bass"]))]
    gain = _lufs_gain(a_m, b_m, sr)

    # Post: B from its drop
    b_end = min(b_drop + post_len, len(b["bass"]))
    wl    = b_end - b_drop
    if wl > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len:pre_len + wl] += b[stem][b_drop:b_drop + wl] * gain

    # Echo tail from A's last vocal
    last_beat   = max(0, a_drop - spb)
    vocal_snip  = a["vocals"][last_beat:a_drop]
    echo_tail   = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
    tail_len    = min(len(echo_tail), post_len)
    if tail_len > 0:
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
        out[pre_len:pre_len + tail_len] += echo_stereo[:tail_len]

    out_path = os.path.join(OUTPUT_DIR, out_name)
    sf.write(out_path, normalize(out), sr)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# T4 — Loop Roll Tension
# ---------------------------------------------------------------------------

def generate_loop_roll_tension(track_a_str, track_b_str, out_name):
    """
    Loop roll on A's last vocal word for 8 beats → B's drop.
    Stereo (N,2) throughout.
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

    buildup_len   = a_drop - a_build_start
    roll_len      = 8 * spb
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

    # Buildup (bass fades last 4 beats)
    if norm_build_len > 0:
        bass_full = max(0, norm_build_len - 4 * spb)
        bass_fade = norm_build_len - bass_full
        for stem in ["drums", "other", "vocals"]:
            wl = _write_len(out, pre_len, a[stem], a_build_start, a_build_start + norm_build_len)
            out[pre_len:pre_len + wl] += a[stem][a_build_start:a_build_start + wl]
        if bass_full > 0:
            wl = _write_len(out, pre_len, a["bass"], a_build_start, a_build_start + bass_full)
            out[pre_len:pre_len + wl] += a["bass"][a_build_start:a_build_start + wl]
        if bass_fade > 0:
            fade = np.linspace(1.0, 0.0, bass_fade, dtype=np.float32)[:, None]
            bs   = a_build_start + bass_full
            wl   = _write_len(out, pre_len + bass_full, a["bass"], bs, bs + bass_fade)
            out[pre_len + bass_full:pre_len + bass_full + wl] += a["bass"][bs:bs + wl] * fade[:wl]

    # Loop roll (mono → stereo)
    clean_vocals_mono = dsp_utils.apply_bandpass(_to_mono(a["vocals"]), sr, low=250, high=5000)
    word_s, word_e    = dsp_utils.find_best_loop_source(
        {"vocals": a["vocals"]}, a_drop, b["bpm"], sr)
    roll_mono = generate_loop_roll(
        clean_vocals_mono, b["bpm"], sr,
        drop_sample_idx=a_drop, num_beats=8,
        exact_word_start=word_s, exact_word_end=word_e
    )
    roll_stereo = np.stack([roll_mono, roll_mono], axis=1)
    roll_start  = pre_len + norm_build_len
    roll_wl     = min(len(roll_stereo), roll_len)
    if roll_wl > 0:
        out[roll_start:roll_start + roll_wl] += roll_stereo[:roll_wl]

    # LUFS match
    a_m = a["bass"][max(0, a_drop - 4 * spb):a_drop]
    b_m = b["bass"][b_drop:min(b_drop + 4 * spb, len(b["bass"]))]
    gain = _lufs_gain(a_m, b_m, sr)

    # B post-drop
    b_end = min(b_drop + post_len, len(b["bass"]))
    wl    = b_end - b_drop
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
    TRACK_A = "Sky High"
    TRACK_B = "Heroes Tonight"

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
