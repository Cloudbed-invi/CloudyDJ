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
from scipy.ndimage import uniform_filter1d

# Try to import from src.engine instead of assuming we're running locally
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


def normalize(audio):
    """
    Passes the audio through the new Master Soft Clipper to prevent 
    transition volume surges without crushing the entire song.
    """
    return dsp_utils.apply_master_limiter(audio)


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
    hop = 1024
    rms = librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=hop)[0]
    
    # 2. Smooth the RMS envelope
    smooth_len = int((60.0 / bpm) * sr / hop) # 1 beat smoothing
    if smooth_len > 0:
        rms = uniform_filter1d(rms, size=smooth_len)

    mean_bass = np.mean(rms) + 1e-9
    frames_per_sec = sr / hop
    start_frame = int(30 * frames_per_sec)
    end_frame   = int(120 * frames_per_sec)

    best_frame, max_contrast = 0, 0
    window = smooth_len * 32
    for frame_idx in range(max(window, start_frame), min(end_frame, len(rms) - window)):
        pre_mean  = np.mean(rms[max(0, frame_idx - window):frame_idx])
        post_mean = np.mean(rms[frame_idx:frame_idx + window])
        contrast  = post_mean - pre_mean
        if contrast > max_contrast and post_mean > 0.03:
            max_contrast = contrast
            best_frame   = frame_idx

    confidence = max_contrast / mean_bass

    if best_frame == 0 or confidence < 0.5:
        return np.argmin(np.abs(beats - (60 * sr))), False

    drop_sample = librosa.frames_to_samples(best_frame, hop_length=hop)
    return np.argmin(np.abs(beats - drop_sample)), True


def get_track_data(crate, track_name):
    full_name = next(k for k in crate.keys() if track_name in k)
    data      = crate[full_name]
    stems     = data["stems"]

    vocals, sr = load_audio(stems["vocals"])
    vocals = dsp_utils.clean_vocal_stem(vocals, sr)
    drums,  _  = load_audio(stems["drums"])
    bass,   _  = load_audio(stems["bass"])
    other,  _  = load_audio(stems["other"])

    mono = _to_mono(drums + bass + other)
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=data["bpm"], units='samples')

    key = detect_key(other + bass, sr)
    drop_beat_idx, drop_confident = detect_first_drop(_to_mono(bass), beats, sr, data["bpm"])
    
    print(f"Loaded '{track_name}' (BPM: {data['bpm']}, Key: {key})")
    print(f"  Drop at beat {drop_beat_idx} (Confident: {drop_confident})")

    return {
        "name":           full_name,
        "bpm":            data["bpm"],
        "key":            key,
        "vocals":         vocals,
        "drums":          drums,
        "bass":           bass,
        "other":          other,
        "beats":          beats,
        "drop_idx":       drop_beat_idx,
        "drop_confident": drop_confident,
        "sr":             sr,
        "segments":       data.get("segments", []),
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
    rate = b_bpm / a["bpm"]
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
        """Time-stretch a stem while preserving stereo width."""
        if stem_audio.ndim == 2 and stem_audio.shape[1] == 2:
            left  = librosa.effects.time_stretch(stem_audio[:, 0].astype(np.float32), rate=actual_rate)
            right = librosa.effects.time_stretch(stem_audio[:, 1].astype(np.float32), rate=actual_rate)
            n = min(len(left), len(right))  # L/R can differ by ~1 sample
            return np.stack([left[:n], right[:n]], axis=1).astype(np.float32)
        mono = _to_mono(stem_audio)
        warped = librosa.effects.time_stretch(mono, rate=actual_rate)
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
    
    if num_beats > 0:
        # Scale each ORIGINAL detected beat sample by the measured actual_rate,
        # instead of extrapolating a rigid grid from beat[0] + b_bpm spacing.
        # This preserves real timing/groove from the source and can't drift,
        # because it never assumes a["bpm"] or the grid was perfectly uniform.
        beats_warped = np.round(a["beats"].astype(np.float64) / actual_rate).astype(np.int64)
    else:
        beats_warped = np.array([], dtype=np.int64)

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
        **a,
        "bpm":      b_bpm,
        "vocals":   vocals_w,
        "drums":    drums_w,
        "bass":     bass_w,
        "other":    other_w,
        "beats":    beats_warped,
        "drop_idx": drop_idx_warped,
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

def generate_bass_swap_transition(track_a_str, track_b_str, out_name, mode="with_vocals_with_fx"):
    """
    Superhuman AI Bass Swap with Auto-Nudge and HPF Tension Sweep:
      • Anchor: B's Drop is aligned to 16 beats AFTER A's Drop.
      • Tension Build (16 beats): A's bass stem receives an HPF sweep (20Hz -> 250Hz).
      • Pre-Blend: B's drums, vocals, and leads fade in (sidechained), but B's bass is MUTED.
      • Nudging: B is phase-aligned to A using waveform cross-correlation at the swap point.
      • The Swap: Exactly at the 16th beat, A's bass cuts, B's bass slams in at 100%.
      • Post-Swap: A's highs/mids fade out over the next 16 beats.
    """
    print(f"\n[Bass Swap] {track_a_str} → {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    
    if not a_orig.get("drop_confident") or not b.get("drop_confident"):
        print(f"  SKIPPED: {out_name} — no confident drop on one side (Bass Swap isn't a good fit here)")
        return
        
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    a_swap = _safe_beat(a["beats"], a["drop_idx"] + 16)
    
    # B's intro anchor. We want 16 beats of build + 32 beats of post-drop
    blend_len      = 16 * spb
    post_len       = 32 * spb
    post_blend_len = 16 * spb # For tails
    b_swap         = int(b["beats"][b["drop_idx"]])

    # Auto-Nudge B to A's exact phase. Sum drums and bass to lock the low end
    max_nudge = int(0.05 * sr)
    a_sync_chunk = a["drums"][a_swap:a_swap + spb] + a["bass"][a_swap:a_swap + spb]
    b_sync_chunk = b["drums"][b_swap:b_swap + spb] + b["bass"][b_swap:b_swap + spb]
    nudge = dsp_utils.calculate_phase_nudge(a_sync_chunk, b_sync_chunk, sr, max_nudge_samples=max_nudge)
    b_swap -= nudge
    print(f"  Auto-Nudge: shifting Track B by {nudge} samples ({nudge/sr*1000:.1f}ms)")
    if abs(nudge) >= max_nudge:
        print(f"  WARNING: nudge clamped at ±{max_nudge} samples — likely still misaligned")

    # Whisper vocal boundaries
    if "with_vocals" in mode:
        # A_vocal_cut represents the absolute cutoff. 
        # Instead of cutting exactly at a_swap, we allow A's vocal to continue over the drop.
        a_vocal_cut, _ = dsp_utils.find_vocal_cutoff_in_buildup(a, max(0, a_swap - 8 * spb), a_swap + 16 * spb, sr)
        b_vocal_entry, _ = dsp_utils.find_vocal_entry(b["vocals"], b_swap, sr)
        
        # Vocal Clash Prevention: The Baton Pass
        # We must ensure A's vocal does NOT overlap with B's vocal.
        if b_vocal_entry is not None:
            # relative offset of B's vocal after the drop (e.g. +3.4s)
            b_entry_rel = b_vocal_entry - b_swap 
            
            # Max allowed A cut is exactly 1 beat BEFORE B's vocal starts
            max_a_cut = a_swap + b_entry_rel - spb
            
            if a_vocal_cut > max_a_cut:
                print(f"  Vocal Clash Prevention: cutting A's vocal early to avoid B's entry!")
                a_vocal_cut = max_a_cut
    else:
        a_vocal_cut = 0
        b_vocal_entry = None

    print(f"  A vocal cut at sample {a_vocal_cut} ({a_vocal_cut/sr:.2f}s)")
    if b_vocal_entry is not None:
        print(f"  B vocal entry at sample {b_vocal_entry} ({b_vocal_entry/sr:.2f}s)")
    else:
        print(f"  B vocal entry at sample None")

    pre_len   = int(10 * sr)
    blend_len = 16 * spb       # 16 beat tension build before the swap
    post_len  = 16 * spb + int(10 * sr) # 16 beat post-blend + 10s tail
    total_len = pre_len + blend_len + post_len

    out = np.zeros((total_len, 2), dtype=np.float32)
    swap_out_idx = pre_len + blend_len

    # ------------------------------------------------------------------
    # Pre (10s of A before the tension build starts)
    # ------------------------------------------------------------------
    a_build_start = a_swap - blend_len
    a_pre_start = max(0, a_build_start - pre_len)
    actual_pre = a_build_start - a_pre_start
    out_pre_off = pre_len - actual_pre
    
    if actual_pre > 0:
        for stem in ["drums", "bass", "other"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_build_start)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]
        
        # A vocals
        if a_vocal_cut > a_pre_start:
            v_end = min(a_vocal_cut, a_build_start)
            wl = _write_len(out, out_pre_off, a["vocals"], a_pre_start, v_end)
            out[out_pre_off:out_pre_off + wl] += a["vocals"][a_pre_start:a_pre_start + wl]

    # ------------------------------------------------------------------
    # The Tension Build (16 beats leading up to the swap)
    # A's Bass is HPF Swept. B fades in (Bass MUTED).
    # ------------------------------------------------------------------
    a_build_end = min(a_swap, len(a["bass"]))
    wl_build = a_build_end - a_build_start
    
    if wl_build > 0:
        # A drums, other (untouched)
        for stem in ["drums", "other"]:
            out[pre_len:pre_len + wl_build] += a[stem][a_build_start:a_build_end]
            
        # A vocals (up to cut)
        v_start = max(a_build_start, 0)
        v_end = min(a_vocal_cut, a_build_end)
        if v_end > v_start:
            v_wl = v_end - v_start
            v_out_idx = pre_len + (v_start - a_build_start)
            out[v_out_idx:v_out_idx + v_wl] += a["vocals"][v_start:v_end]
            
        # A Bass HPF Tension Sweep (20Hz -> 250Hz)
        a_bass_build = a["bass"][a_build_start:a_build_end]
        a_bass_swept = dsp_utils.apply_hpf_sweep(a_bass_build, sr, start_freq=20, end_freq=250, num_chunks=16)
        out[pre_len:pre_len + wl_build] += a_bass_swept

    # LUFS match
    a_measure = a["bass"][max(0, a_swap - 4 * spb):a_swap] + a["drums"][max(0, a_swap - 4 * spb):a_swap]
    b_measure = b["bass"][max(0, b_swap):min(b_swap + 4 * spb, len(b["bass"]))] + b["drums"][max(0, b_swap):min(b_swap + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_measure, b_measure, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    # ------------------------------------------------------------------
    # The Tension Build (16 beats leading up to the swap)
    # ------------------------------------------------------------------
    # A fades out (cosine), B fades in (sine)
    b_build_start = b_swap - blend_len
    a_build_start = a_swap - blend_len
    
    # ------------------------------------------------------------------
    # Calculate B's variance post-drop (Sparse vs Dense rule)
    # ------------------------------------------------------------------
    b_post_other = b["other"][b_swap:b_swap + int(16 * spb)]
    if len(b_post_other) > int(0.05 * sr):
        mono_b = np.mean(b_post_other, axis=1) if b_post_other.ndim == 2 else b_post_other
        window_size = int(0.05 * sr)
        num_windows = len(mono_b) // window_size
        reshaped = mono_b[:num_windows * window_size].reshape(num_windows, window_size)
        rms_windows = np.sqrt(np.mean(reshaped**2, axis=1))
        b_variance = float(np.var(rms_windows))
    else:
        b_variance = 0.0
        
    is_dense = b_variance < 0.0015
    print(f"  Track B Variance: {b_variance:.6f} -> {'Dense (Hard Swap)' if is_dense else 'Sparse (Gentle Swap)'}")
    
    t_in = np.linspace(0.0, np.pi / 2, blend_len, dtype=np.float32)
    fade_in = np.sin(t_in)[:, None]
    
    if is_dense:
        # Dense (Hard Swap): A stays at 100% to keep energy, B builds up underneath
        a_fade_build = np.ones((blend_len, 1), dtype=np.float32)
    else:
        # Sparse (Gentle Swap): A dips slightly to 70% to make room for B's sweep
        a_fade_build = (0.7 + 0.3 * np.cos(t_in))[:, None]
    
    # A's build-up. Drums and Other fade according to rule.
    for stem in ["drums", "other"]:
        a_src_start = max(0, a_build_start)
        a_src_end = min(a_swap, len(a[stem]))
        wl_a = a_src_end - a_src_start
        if wl_a > 0:
            out_idx = pre_len + (a_src_start - a_build_start)
            chunk = np.copy(a[stem][a_src_start:a_src_end])
            
            if stem == "drums":
                # Apply HPF to Track A's drums during build-up to carve out space and build tension
                chunk = dsp_utils.apply_hpf(chunk, sr, 100)
                
            out[out_idx:out_idx + wl_a] += chunk * a_fade_build[-wl_a:]
            
    # Vocals (play until cut)
    a_src_start = max(0, a_build_start)
    a_src_end = min(a_vocal_cut, len(a["vocals"]))
    wl_a_vocal = a_src_end - a_src_start
    if wl_a_vocal > 0:
        out_idx = pre_len + (a_src_start - a_build_start)
        out[out_idx:out_idx + wl_a_vocal] += a["vocals"][a_src_start:a_src_end]
            
    # A's bass plays until 1 beat before the drop (anticipation vacuum). 50ms micro-fade out to prevent pop.
    a_bass_start = max(0, a_build_start)
    a_bass_end = max(0, min(a_swap - spb, len(a["bass"])))
    wl_a_bass = a_bass_end - a_bass_start
    if wl_a_bass > 0:
        out_idx = pre_len + (a_bass_start - a_build_start)
        chunk = np.copy(a["bass"][a_bass_start:a_bass_end])
        
        fade_len = int(0.05 * sr)  # 50ms
        if fade_len > wl_a_bass:
            fade_len = wl_a_bass
            
        if fade_len > 0:
            curve = np.cos(np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32))
            chunk[-fade_len:] *= curve[:, np.newaxis]
                
        out[out_idx:out_idx + wl_a_bass] += chunk
    
    for stem in ["drums", "other", "vocals"]:
        # If no vocals, skip vocals stem completely
        if stem == "vocals" and "no_vocals" in mode:
            continue
            
        b_src_start = max(0, b_build_start)
        b_src_end = min(b_swap, len(b[stem]))
        wl_b = b_src_end - b_src_start
        if wl_b > 0:
            out_idx = pre_len + (b_src_start - b_build_start)
            chunk = np.copy(b[stem][b_src_start:b_src_end])
            
            if stem == "vocals":
                v_mask_start = max(0, b_vocal_entry - b_src_start) if b_vocal_entry else wl_b
                if v_mask_start > 0 and v_mask_start < wl_b:
                    chunk[:v_mask_start] = 0.0
                elif v_mask_start >= wl_b:
                    chunk[:] = 0.0
                out[out_idx:out_idx + wl_b] += chunk * gain
            else:
                if stem == "drums":
                    chunk = dsp_utils.apply_hpf(chunk, sr, 100)
                elif stem == "other" and "with_fx" in mode:
                    chunk = dsp_utils.apply_lpf_sweep(chunk, sr, start_freq=500, end_freq=20000, num_chunks=16)
                out[out_idx:out_idx + wl_b] += chunk * fade_in[-wl_b:] * gain

    # ------------------------------------------------------------------
    # Echo tail for A's vocal
    # ------------------------------------------------------------------
    if a_vocal_cut > spb and "with_fx" in mode and "with_vocals" in mode:
        vocal_snip = a["vocals"][a_vocal_cut - spb:a_vocal_cut]
        echo_tail  = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
            
        # The out index is relative to where A's vocal cut happened.
        # Note: a_vocal_cut could now be PAST a_swap.
        if a_vocal_cut <= a_swap:
            # Cut happened before/at the drop
            out_vocal_cut = pre_len + blend_len - (a_swap - a_vocal_cut)
        else:
            # Cut happened after the drop
            out_vocal_cut = pre_len + blend_len + (a_vocal_cut - a_swap)
            
        tail_len = min(len(echo_stereo), total_len - out_vocal_cut)
        if tail_len > 0 and out_vocal_cut >= 0:
            out[out_vocal_cut:out_vocal_cut + tail_len] += echo_stereo[:tail_len] * 1.0

    # ------------------------------------------------------------------
    # Wash-out Echo Tail for Track A
    # We apply this to both Hard and Gentle Swaps now to mask the dry cut.
    # ------------------------------------------------------------------
    if a_swap > 4 * spb and "with_fx" in mode:
        # Grab a larger snippet (8 beats) of the synths/leads right before the drop
        wash_snip = a["other"][a_swap - 8 * spb : a_swap]
        wash_tail = dsp_utils.generate_echo_tail(wash_snip, sr, a["bpm"], beats=8)
        if wash_tail.ndim == 1:
            wash_tail = np.stack([wash_tail, wash_tail], axis=1)
        tail_len = min(len(wash_tail), total_len - swap_out_idx)
        if tail_len > 0:
            # Filter it so it doesn't clash with B's leads, just sits in the background
            wash_tail = dsp_utils.apply_lpf(wash_tail, sr, 1500)
            wash_tail = dsp_utils.apply_hpf(wash_tail, sr, 300)
            out[swap_out_idx:swap_out_idx + tail_len] += wash_tail[:tail_len] * 0.6

    # B's bass drops exactly at the swap. 5ms micro-fade in to prevent zero-crossing pop.
    b_bass_start = max(0, b_swap)
    b_bass_end = min(b_swap + post_len, len(b["bass"]))
    wl_b_bass = b_bass_end - b_bass_start
    if wl_b_bass > 0:
        out_idx = swap_out_idx + (b_bass_start - b_swap)
        chunk = np.copy(b["bass"][b_bass_start:b_bass_end])
        
        fade_len = int(0.005 * sr)  # 5ms
        if fade_len > wl_b_bass:
            fade_len = wl_b_bass
            
        if fade_len > 0:
            curve = np.sin(np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32))
            chunk[:fade_len] *= curve[:, np.newaxis]
                
        out[out_idx:out_idx + wl_b_bass] += chunk * gain

    # ------------------------------------------------------------------
    # The Swap & Post-Blend (16 or 32 beats fading A out, B at full power)
    # ------------------------------------------------------------------
    if is_dense:
        post_blend_len = int(spb) # Just a tiny 1-beat fade to avoid clicks
        t_out = np.linspace(0.0, np.pi / 2, post_blend_len, dtype=np.float32)
        fade_out_post = np.cos(t_out)[:, None]
    else:
        post_blend_len = 32 * spb # Long 32-beat carry-over
        t_out = np.linspace(0.0, np.pi / 2, post_blend_len, dtype=np.float32)
        fade_out_post = np.cos(t_out)[:, None]
    
    # A fades out (drums, other) over post-swap
    for stem in ["drums", "other"]:
        a_post_start = max(0, a_swap)
        a_post_end = min(a_swap + post_blend_len, len(a[stem]))
        wl_a = a_post_end - a_post_start
        if wl_a > 0:
            out_idx = swap_out_idx + (a_post_start - a_swap)
            chunk = np.copy(a[stem][a_post_start:a_post_end])
            
            if not is_dense:
                # For Sparse transitions, apply HPF so only Treble carries over (no mid-mud)
                chunk = dsp_utils.apply_hpf(chunk, sr, 2000)
                
            out[out_idx:out_idx + wl_a] += chunk * fade_out_post[:wl_a]
            
    # If A's vocal carries over the drop, we add it here in the post-swap section
    if "with_vocals" in mode and a_vocal_cut > a_swap:
        a_vocal_post_start = a_swap
        a_vocal_post_end = a_vocal_cut
        wl_a_vocal_post = a_vocal_post_end - a_vocal_post_start
        if wl_a_vocal_post > 0:
            out_idx = swap_out_idx
            chunk = np.copy(a["vocals"][a_vocal_post_start:a_vocal_post_end])
            out[out_idx:out_idx + wl_a_vocal_post] += chunk

    # B slams in full power (Drums, Other, Vocals)
    for stem in ["drums", "other", "vocals"]:
        if stem == "vocals" and "no_vocals" in mode:
            continue
            
        b_post_start = max(0, b_swap)
        b_post_end = min(b_swap + post_len, len(b[stem]))
        wl_b = b_post_end - b_post_start
        if wl_b > 0:
            out_idx = swap_out_idx + (b_post_start - b_swap)
            chunk = np.copy(b[stem][b_post_start:b_post_end])
            
            if stem == "vocals":
                v_mask_start = max(0, b_vocal_entry - b_post_start) if b_vocal_entry is not None else wl_b
                if v_mask_start > 0 and v_mask_start < wl_b:
                    chunk[:v_mask_start] = 0.0
                elif v_mask_start >= wl_b:
                    chunk[:] = 0.0
                    
            out[out_idx:out_idx + wl_b] += chunk * gain

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

    gap_len  = 0              # No gap, instant cut
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

    # ------------------------------------------------------------------
    # Background tension riser under A's last 8 beats
    # ------------------------------------------------------------------
    riser_beats = min(8, int(actual_pre / spb))
    if riser_beats > 0:
        riser_samples = riser_beats * spb
        riser = dsp_utils.generate_tension_riser(riser_samples, sr)
        riser_stereo = np.stack([riser, riser], axis=1)
        riser_start = pre_len - riser_samples
        out[riser_start:pre_len] += riser_stereo * 0.7

    # Echo tail from A's last vocal
    last_beat  = max(0, a_drop - spb)
    vocal_snip = a["vocals"][last_beat:a_drop]

    # LUFS match: Measure A's Drop vs B's Drop
    a_m = a["bass"][a_drop:min(a_drop + 4 * spb, len(a["bass"]))] + \
          a["drums"][a_drop:min(a_drop + 4 * spb, len(a["drums"]))]
    b_m = b["bass"][b_drop:min(b_drop + 4 * spb, len(b["bass"]))] + \
          b["drums"][b_drop:min(b_drop + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_m, b_m, sr)
    print(f"  LUFS gain applied to B: {gain:.3f}×")

    # Post: B from its drop (instant slam)
    b_start_out = pre_len
    b_end       = min(b_drop + post_len, len(b["bass"]))
    wl          = b_end - b_drop
    if wl > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[b_start_out:b_start_out + wl] += b[stem][b_drop:b_drop + wl] * gain

    # Add Impact Downlifter on B's drop
    downlifter = dsp_utils.generate_impact_downlifter(int(8 * spb), sr)
    dl_stereo  = np.stack([downlifter, downlifter], axis=1)
    dl_len     = min(len(dl_stereo), post_len)
    if dl_len > 0:
        out[b_start_out:b_start_out + dl_len] += dl_stereo[:dl_len]

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
            
        # HPF Sweep on A's main stems over the 8-beat roll to make room
        if roll_len > 0:
            sweep_start = pre_len + buildup_len - roll_len
            # Apply HPF sweep from 20Hz up to 2000Hz over the 8 beats
            for stem in ["drums", "other", "bass", "vocals"]:
                a_section = a[stem][a_drop - roll_len:a_drop]
                a_hpf = dsp_utils.apply_hpf_sweep(a_section, sr, start_freq=20, end_freq=2000, num_chunks=16)
                # Replace the original with the swept version
                out[sweep_start:sweep_start + roll_len] -= a_section
                out[sweep_start:sweep_start + roll_len] += a_hpf

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
        # Layer loop roll at 0.7 volume so it doesn't clip
        out[roll_start:roll_start + roll_wl] += roll_stereo[:roll_wl] * 0.7

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
            
    # Add Impact Downlifter on B's drop
    downlifter = dsp_utils.generate_impact_downlifter(int(8 * spb), sr)
    dl_stereo  = np.stack([downlifter, downlifter], axis=1)
    dl_len     = min(len(dl_stereo), post_len)
    if dl_len > 0:
        out[post_start:post_start + dl_len] += dl_stereo[:dl_len]

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
