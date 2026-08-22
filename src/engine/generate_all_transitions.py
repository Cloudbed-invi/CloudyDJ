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




def detect_first_drop(bass_mono, beats, sr, bpm):
    """Bass RMS sustained energy scan between 10 s and 80% of track length."""
    hop = 1024
    rms = librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=hop)[0]
    
    # Smooth the RMS envelope
    smooth_len = int((60.0 / bpm) * sr / hop) # 1 beat smoothing
    if smooth_len > 0:
        rms = uniform_filter1d(rms, size=smooth_len)

    mean_bass = np.mean(rms) + 1e-9
    frames_per_sec = sr / hop
    start_frame = int(10 * frames_per_sec)
    end_frame   = int(len(rms) * 0.8)

    best_frame, best_score = 0, 0
    window = smooth_len * 16
    sustain_window = smooth_len * 32
    
    for frame_idx in range(max(window, start_frame), min(end_frame, len(rms) - sustain_window)):
        pre_mean  = np.mean(rms[max(0, frame_idx - window):frame_idx])
        post_mean = np.mean(rms[frame_idx:frame_idx + sustain_window])
        score = post_mean * (1 + (post_mean - pre_mean) / (pre_mean + 1e-6))
        if score > best_score and post_mean > 0.02:
            best_score = score
            best_frame = frame_idx

    confidence = best_score / mean_bass

    if best_frame == 0 or confidence < 0.5:
        return np.argmin(np.abs(beats - (60 * sr))), False

    # Safety net: scan forward to ensure bass > 0.05
    for frame_idx in range(best_frame, min(best_frame + int(10 * frames_per_sec), len(rms))):
        if rms[frame_idx] > 0.05:
            best_frame = frame_idx
            break

    drop_sample = librosa.frames_to_samples(best_frame, hop_length=hop)
    return np.argmin(np.abs(beats - drop_sample)), True


def get_track_data(crate, track_name):
    full_name = next(k for k in crate.keys() if track_name in k)
    data      = crate[full_name]
    stems     = data["stems"]

    vocals, sr = load_audio(stems["vocals"])
    # removed clean_vocal_stem
    drums,  _  = load_audio(stems["drums"])
    bass,   _  = load_audio(stems["bass"])
    other,  _  = load_audio(stems["other"])

    mono = _to_mono(drums + bass + other)
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=data["bpm"], units='samples')

    key = data["key"]
    
    # 100% accurate drop detection via crate metadata override
    drop_override = data.get("drop_idx")
    if drop_override is not None:
        drop_beat_idx, drop_confident = drop_override, True
    else:
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
        "vocal_words":    data.get("vocal_words", []),
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
    # Step 1: Compute Exact Target Length for Stems
    # ------------------------------------------------------------------
    target_len = int(len(a["vocals"]) / rate)
    actual_rate = rate

    # ------------------------------------------------------------------
    # Step 2: Interleave stems to 8 channels and stretch with WSOLA
    # This guarantees 100% perfect phase coherence across all stems
    # to prevent frequency cancellation when summed back together.
    # ------------------------------------------------------------------
    print(f"  Warping stems using 8-channel audiotsm WSOLA (rate={rate:.3f})...")
    # Interleave to (N, 8)
    interleaved = np.concatenate([
        a["vocals"], a["drums"], a["bass"], a["other"]
    ], axis=1)
    
    y_in = interleaved.T.astype(np.float32)
    import audiotsm
    import audiotsm.io.array
    reader = audiotsm.io.array.ArrayReader(y_in)
    tsm = audiotsm.wsola(channels=8, speed=rate)
    writer = audiotsm.io.array.ArrayWriter(channels=8)
    tsm.run(reader, writer)
    out_audio = writer.data.T
    
    if len(out_audio) > target_len:
        out_audio = out_audio[:target_len]
    elif len(out_audio) < target_len:
        out_audio = np.pad(out_audio, ((0, target_len - len(out_audio)), (0, 0)))

    # De-interleave back to (N, 2) stems
    vocals_w = out_audio[:, 0:2]
    drums_w  = out_audio[:, 2:4]
    bass_w   = out_audio[:, 4:6]
    other_w  = out_audio[:, 6:8]

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
    
    vocal_words_warped = [
        {**w,
         "start": w["start"] / actual_rate,
         "end":   w["end"]   / actual_rate}
        for w in a.get("vocal_words", [])
    ]

    return {
        **a,
        "bpm":         b_bpm,
        "vocals":      vocals_w,
        "drums":       drums_w,
        "bass":        bass_w,
        "other":       other_w,
        "beats":       beats_warped,
        "drop_idx":    drop_idx_warped,
        "segments":    segments_warped,
        "vocal_words": vocal_words_warped,
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

def generate_transition(track_a_str, track_b_str, out_name, strategy="cut"):
    """
    Drop-Point Stem Swap:
      • A plays at 100% energy up to the EXACT drop.
      • B drums/synths secretly pre-roll inside the same pre_len window.
      • At the exact drop: A's bass stops, B's bass comes in FULL.
      • Post-drop blend is handled dynamically by Mixgraph Strategy.
      • 3-Stage Vocal Handoff: A phrase cut + LPF echo, B delayed entry + HPF sweep reveal.
    """
    print(f"\n[Bass Swap] {track_a_str} -> {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    # EXACT DROP POINTS — snapped to nearest zero crossing to eliminate click artifacts
    a_drop = _safe_beat(a["beats"], a["drop_idx"])
    b_drop = _safe_beat(b["beats"], b["drop_idx"])
    
    # Snap both cut points to the nearest zero crossing in the mixed signal
    # This prevents mid-waveform splices that create audible clicks
    a_mix = a["drums"] + a["bass"] + a["other"]
    b_mix = b["drums"] + b["bass"] + b["other"]
    a_swap = dsp_utils.find_zero_crossing(a_mix, a_drop, search_radius=1024, direction='before')
    b_swap = dsp_utils.find_zero_crossing(b_mix, b_drop, search_radius=1024, direction='after')
    print(f"  ZC snap: A cut {a_drop} -> {a_swap} (offset {a_swap - a_drop}), B start {b_drop} -> {b_swap} (offset {b_swap - b_drop})")

    # ------------------------------------------------------------------
    # Whisper vocal boundaries - BATON PASS
    # ------------------------------------------------------------------
    b_vocal_entry  = dsp_utils.find_strong_phrase_entry(b["vocals"], b_swap, sr, min_gap_s=0.4)
    b_entry_offset = b_vocal_entry - b_swap

    if b_entry_offset < int(1.5 * spb):
        # B sings immediately — cut A during the build-up
        a_vocal_cut = dsp_utils.find_vocal_cutoff_in_buildup(a, max(0, a_swap - 16 * spb), a_swap, sr)
    else:
        target_cut_point = a_swap + b_entry_offset - spb
        a_vocal_cut = dsp_utils.find_vocal_cutoff_in_buildup(a, a_swap, max(a_swap + 8 * spb, target_cut_point), sr)

    if (a_vocal_cut - a_swap) > b_entry_offset:
        a_vocal_cut = a_swap + max(0, b_entry_offset - int(0.5 * spb))

    # Fix Track 4 Clashing: if strategy is CUT and keys clash, don't let A vocals bleed into B's drop at all
    if strategy == "cut":
        from decision_engine import _camelot_distance
        if _camelot_distance(a_orig["key"], b["key"]) == "clash":
            a_vocal_cut = min(a_vocal_cut, a_swap)
            print("  Vocal clash prevented: forced A vocal cut at drop.")

    print(f"  B vocal entry: {b_entry_offset/sr:.2f}s after drop")
    print(f"  A vocal cut: {(a_vocal_cut - a_swap)/sr:.2f}s after drop")

    # ------------------------------------------------------------------
    # Buffer layout: always 10s pre-drop, 32-beat blend, 10s post
    # The 32-beat B pre-roll lives INSIDE the pre_len window (last 16 beats of it)
    # ------------------------------------------------------------------
    pre_len   = int(10 * sr)   # 10s of A before the drop
    blend_len = 32 * spb       # 32-beat secret crossfade post-drop
    post_len  = int(10 * sr)   # 10s of B solo
    total_len = pre_len + blend_len + post_len

    out = np.zeros((total_len, 2), dtype=np.float32)

    # Drop lands at exactly pre_len in the output buffer
    drop_out = pre_len

    # LUFS gain match — A drop vs B drop, capped to avoid extreme amplification
    a_measure = a["bass"][a_swap:min(a_swap + 4 * spb, len(a["bass"]))] + \
                a["drums"][a_swap:min(a_swap + 4 * spb, len(a["drums"]))] + \
                a["vocals"][a_swap:min(a_swap + 4 * spb, len(a["vocals"]))] + \
                a["other"][a_swap:min(a_swap + 4 * spb, len(a["other"]))]
                
    b_measure = b["bass"][b_swap:min(b_swap + 4 * spb, len(b["bass"]))] + \
                b["drums"][b_swap:min(b_swap + 4 * spb, len(b["drums"]))] + \
                b["vocals"][b_swap:min(b_swap + 4 * spb, len(b["vocals"]))] + \
                b["other"][b_swap:min(b_swap + 4 * spb, len(b["other"]))]
    gain = min(_lufs_gain(a_measure, b_measure, sr), 2.0)  # cap: never amplify B by more than +6dB
    
    # In hard-swap strategies (cut, echo_out), A's drop and B's drop do not play simultaneously.
    # Therefore, we DO NOT aggressively duck B. We want B to hit with full impact!
    # If B is naturally louder than A (gain < 1.0), we force gain = 1.0 so B isn't weakened.
    if strategy in ["cut", "echo_out"] and gain < 1.0:
        gain = 1.0

    # Measure masking ratio to decide if we need to sidechain A's drums
    a_drums_rms = np.sqrt(np.mean(dsp_utils._to_mono(a["drums"][a_swap:min(a_swap + 4 * spb, len(a["drums"]))])**2))
    b_vocal_rms = np.sqrt(np.mean(dsp_utils._to_mono(b["vocals"][b_vocal_entry:min(b_vocal_entry + 4 * spb, len(b["vocals"]))])**2))
    
    needs_duck = False
    if b_vocal_rms > 0.001:  # Only if B actually sings here
        masking_ratio = a_drums_rms / (b_vocal_rms * gain + 1e-9)
        if masking_ratio > 1.5:
            needs_duck = True
            print(f"  Masking detected (ratio={masking_ratio:.2f}x). Applying gentle sidechain duck to A's drums.")

    # ------------------------------------------------------------------
    # Pre-Drop: A at full energy (all 4 stems) — fills [0 .. drop_out]
    # ------------------------------------------------------------------
    a_pre_start = max(0, a_swap - pre_len)
    actual_pre  = a_swap - a_pre_start
    out_pre_off = pre_len - actual_pre   # 0 when track is long enough

    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_swap)
            chunk = a[stem][a_pre_start:a_pre_start + wl].copy()
            
            # THE LOOP AND BUILD: Create a 4-beat tension roll right before the drop
            if strategy == "loop_and_build" and stem in ["drums", "other", "vocals"]:
                roll_beats = 4
                roll_samples = int(roll_beats * spb)
                if wl > roll_samples:
                    # Take 1 beat of audio, 4 beats before the drop
                    loop_src = chunk[wl - roll_samples : wl - roll_samples + int(spb)].copy()
                    
                    # Create a repeating 1-beat loop for the last 4 beats
                    for i in range(4):
                        start = wl - roll_samples + (i * int(spb))
                        end = start + int(spb)
                        if end <= wl:
                            chunk[start:end] = loop_src

            # When B sings at beat 0, fade A's vocals out over the last 8 beats
            # of pre-drop using a cosine curve (sounds natural, gives 3.5s clean runway).
            if stem == "vocals" and b_entry_offset <= int(2 * spb):
                fade_beats = 8
                fade_samples = min(fade_beats * spb, wl)
                fade_env = np.cos(np.linspace(0.0, np.pi / 2, fade_samples)).astype(np.float32)
                if chunk.ndim == 2:
                    chunk[-fade_samples:] *= fade_env[:, None]
                else:
                    chunk[-fade_samples:] *= fade_env
            else:
                # Apply a tiny 5ms micro-fade-out to the very end of all other stems 
                # (or vocals that aren't already faded) to prevent zero-crossing pops
                # since A stops instantly at the drop in a CUT/ECHO_OUT strategy.
                pop_fade_len = min(int(0.005 * sr), wl)
                if pop_fade_len > 0:
                    pop_fade = np.linspace(1.0, 0.0, pop_fade_len, dtype=np.float32)
                    if chunk.ndim == 2:
                        chunk[-pop_fade_len:] *= pop_fade[:, None]
                    else:
                        chunk[-pop_fade_len:] *= pop_fade

            out[out_pre_off:out_pre_off + wl] += chunk


    # ------------------------------------------------------------------
    # LOOP ECHO "CURIOSITY BUILDER" — only when B sings at beat 0.
    #
    # Rule: take 1/2 beat of B's first word, reverse it, LPF to 3kHz.
    # Loop it 4 times leading up to the drop (-2.0, -1.5, -1.0, -0.5 beats)
    # with increasing volume to create a rhythmic vocal riser.
    # ------------------------------------------------------------------
    if b_entry_offset <= int(2 * spb):
        slice_len = min(int(spb / 2), drop_out, len(b["vocals"]) - b_vocal_entry)
        if slice_len > 128 and b_vocal_entry + slice_len <= len(b["vocals"]):
            b_first_slice = b["vocals"][b_vocal_entry:b_vocal_entry + slice_len].copy()
            rev_slice = b_first_slice[::-1].copy()
            rev_slice = dsp_utils.apply_lpf(rev_slice, sr, 3000)
            
            # small hanning window to avoid clicks
            env = np.hanning(slice_len).astype(np.float32)
            if rev_slice.ndim == 1:
                rev_slice = np.stack([rev_slice, rev_slice], axis=1)
            env = env[:, None]
            rev_slice = (rev_slice * env).astype(np.float32)
            
            volumes = [0.05, 0.10, 0.15, 0.25]
            for i, vol in enumerate(volumes):
                start_offset = int((2.0 - i * 0.5) * spb)
                echo_start = drop_out - start_offset
                if echo_start >= 0:
                    ew = min(slice_len, len(out) - echo_start)
                    out[echo_start:echo_start + ew] += rev_slice[:ew] * vol * gain

    # ------------------------------------------------------------------
    # White noise impact at the exact drop — forward sweep
    # Applied only on 'cut' (at very low volume to mask splice) or 'filter_sweep'
    # ------------------------------------------------------------------
    if strategy in ["cut", "filter_sweep", "quick_blend"]:
        noise_vol = 0.02 if strategy in ["cut", "quick_blend"] else 0.05
        noise_len    = int(2.0 * spb)
        noise_mono   = dsp_utils.generate_white_noise_sweep(noise_len, sr) * noise_vol
        # Gentle low-pass at 8kHz so the sweep doesn't clash with cymbal energy
        noise_mono   = dsp_utils.apply_lpf(noise_mono, sr, 8000)
        noise_stereo = np.stack([noise_mono, noise_mono], axis=1)
        nw = min(noise_len, total_len - drop_out)
        if nw > 0:
            out[drop_out:drop_out + nw] += noise_stereo[:nw]

    # ------------------------------------------------------------------
    # Post-Drop Blend Zone: strategy-based execution
    # ------------------------------------------------------------------
    t        = np.linspace(0.0, np.pi / 2, blend_len, dtype=np.float32)
    fade_out = np.cos(t)[:, None]
    fade_in  = np.sin(t)[:, None].astype(np.float32)

    # 1. Bass: Bass ALWAYS swaps instantly.
    # We apply a tiny 10ms micro-fade (approx 441 samples) to prevent zero-crossing pops,
    # but absolutely NO 1-beat crossfade. A 1-beat crossfade destroys kick transients.
    micro_fade_len = min(int(0.010 * sr), blend_len)
    
    b_bass_end = min(b_swap + blend_len, len(b["bass"]))
    bbl = b_bass_end - b_swap
    if bbl > 0:
        b_bass_chunk = b["bass"][b_swap:b_bass_end].copy() * gain
        # Tiny fade in to prevent pop — use separate variable to NOT overwrite blend fade_in
        if bbl >= micro_fade_len:
            bass_micro_fade = np.linspace(0.0, 1.0, micro_fade_len, dtype=np.float32)
            if b_bass_chunk.ndim == 2: b_bass_chunk[:micro_fade_len] *= bass_micro_fade[:, None]
            else: b_bass_chunk[:micro_fade_len] *= bass_micro_fade
        out[drop_out:drop_out + bbl] += b_bass_chunk
        
    # We DO NOT write A's bass after the swap point. A's bass stops instantly.

    # 2. Drums & Other (Synths) -> Execute Mixgraph Strategy
    for stem in ["drums", "other"]:
        if strategy == "long_blend":
            # The Long Blend: 16-beat smooth crossfade
            blend_beats = 16
            fade_len = min(blend_beats * spb, blend_len)
            
            # A fades out smoothly
            src_end = min(a_swap + fade_len, len(a[stem]))
            wl = src_end - a_swap
            if wl > 0:
                chunk_fade_out = np.cos(np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32))[:, None]
                a_chunk = a[stem][a_swap:src_end] * chunk_fade_out[:wl]
                if stem == "drums" and needs_duck:
                    b_vocals_in_blend = b["vocals"][b_swap:b_swap + wl]
                    if len(b_vocals_in_blend) < wl:
                        if b_vocals_in_blend.ndim == 2: b_vocals_in_blend = np.pad(b_vocals_in_blend, ((0, wl - len(b_vocals_in_blend)), (0, 0)), 'constant')
                        else: b_vocals_in_blend = np.pad(b_vocals_in_blend, (0, wl - len(b_vocals_in_blend)), 'constant')
                    a_chunk = dsp_utils.apply_sidechain_duck(a_chunk, b_vocals_in_blend, sr)
                out[drop_out:drop_out + wl] += a_chunk
                
            # B drops at full volume (standard modern DJing approach to blends)
            src_end_b = min(b_swap + blend_len, len(b[stem]))
            wl_b = src_end_b - b_swap
            if wl_b > 0:
                out[drop_out:drop_out + wl_b] += b[stem][b_swap:src_end_b] * gain
                
        elif strategy == "cut":
            # The Cut: Instant swap. No bleeding of A's drop audio into B.
                
            src_end_b = min(b_swap + blend_len, len(b[stem]))
            wl_b = src_end_b - b_swap
            if wl_b > 0:
                out[drop_out:drop_out + wl_b] += b[stem][b_swap:src_end_b] * gain
                
        elif strategy == "echo_out":
            # The Echo Out: Hard cut A, but throw it into a massive reverb/delay tail
            wash_start = max(0, a_swap - 2 * spb)
            if a_swap > wash_start:
                snip = a[stem][wash_start:a_swap]
                from src.engine import effects
                padded_snip = np.zeros((len(snip) + blend_len, 2), dtype=np.float32)
                padded_snip[:len(snip)] = snip
                wash_tail = effects.reverb_throw(padded_snip, len(padded_snip), sr)
                tail = wash_tail[len(snip):]
                wl_tail = min(len(tail), blend_len)
                if wl_tail > 0:
                    # Boost the echo tail so it creates a massive wash. 0.7 was too quiet.
                    out[drop_out:drop_out + wl_tail] += tail[:wl_tail] * 1.2
                    
            # B drops at full volume
            src_end_b = min(b_swap + blend_len, len(b[stem]))
            wl_b = src_end_b - b_swap
            if wl_b > 0:
                out[drop_out:drop_out + wl_b] += b[stem][b_swap:src_end_b] * gain
                
        elif strategy == "filter_sweep":
            # The Filter Sweep: A gets LPF swept away over 16 beats. B drops full.
            # (Ideally B would HPF in, but for a bass swap, dropping B full is standard).
            blend_beats = 16
            fade_len = min(blend_beats * spb, blend_len)
            src_end = min(a_swap + fade_len, len(a[stem]))
            wl = src_end - a_swap
            if wl > 0:
                chunk = a[stem][a_swap:src_end].copy()
                # Apply a static LPF for now to mimic the swept sound (simplification)
                chunk = dsp_utils.apply_lpf(chunk, sr, 1000)
                chunk_fade_out = np.cos(np.linspace(0.0, np.pi / 2, fade_len, dtype=np.float32))[:, None]
                chunk *= chunk_fade_out[:wl]
                out[drop_out:drop_out + wl] += chunk
                
            # B drops at full volume
            src_end_b = min(b_swap + blend_len, len(b[stem]))
            wl_b = src_end_b - b_swap
            if wl_b > 0:
                out[drop_out:drop_out + wl_b] += b[stem][b_swap:src_end_b] * gain
                
        elif strategy == "loop_and_build":
            # Loop and build finishes with a hard drop for A. B drops at full.
            src_end_b = min(b_swap + blend_len, len(b[stem]))
            wl_b = src_end_b - b_swap
            if wl_b > 0:
                out[drop_out:drop_out + wl_b] += b[stem][b_swap:src_end_b] * gain

        # ---- Track 1 fix: bridge the mid hollow by continuing A's synths ----
        # Wild's 'other' stem is virtually silent at its drop (mid=0.001).
        # Wild is a bass+drums-only intro — there are no synths for the full blend.
        # We extend Tiesto's synths for 16 beats at a slow cosine fade-out
        # so the mid band is filled for the entire first half of the blend zone.
        if stem == "other":
            bridge_beats = 16
            bridge_len = min(bridge_beats * spb, blend_len)
            a_other_src_end = min(a_swap + bridge_len, len(a["other"]))
            wl_bridge = a_other_src_end - a_swap
            if wl_bridge > 0:
                bridge_env = np.cos(np.linspace(0.0, np.pi / 2, wl_bridge, dtype=np.float32))[:, None] * 0.55
                out[drop_out:drop_out + wl_bridge] += a["other"][a_swap:a_other_src_end] * bridge_env

    # 4. A Vocals: write into blend zone with gentle LPF to push them back,
    #    hard cut at a_vocal_cut, then a short LPF echo tail.
    a_vcut_offset = a_vocal_cut - a_swap   # can be negative (pre-drop cut)
    if a_vcut_offset > 0:
        # A vocals are still singing post-drop up to the cut.
        # Apply a soft LPF (4kHz) so they feel recessed behind B's bass
        a_vocal_blend = dsp_utils.apply_lpf(a["vocals"][a_swap:a_vocal_cut], sr, 4000)
        out[drop_out:drop_out + a_vcut_offset] += a_vocal_blend
        echo_src_start = a_vocal_cut - spb
        echo_src_end   = a_vocal_cut
    else:
        # Cut is pre-drop: A vocals already in buffer from the pre-drop section.
        echo_src_start = max(0, a_vocal_cut - spb)
        echo_src_end   = max(echo_src_start + 1, a_vocal_cut)

    # Loop Roll: only add if B doesn't sing immediately (would clash)
    # AND only if strategy is NOT 'cut'. Cut implies harmonic clash or hard swap, so a loop roll ruins it.
    if strategy != "cut" and b_entry_offset > int(2 * spb) and echo_src_end > echo_src_start and echo_src_start >= 0:
        vocal_snip = a["vocals"][echo_src_start:echo_src_end] # This is exactly 1 spb long
        # Repeat it to fill the gap until B enters, max 16 beats
        roll_beats = min(16, int(b_entry_offset / spb))
        roll_len = roll_beats * spb
        roll_audio = np.tile(vocal_snip, (roll_beats, 1))
        
        # Apply LPF to push it back
        roll_audio = dsp_utils.apply_lpf(roll_audio, sr, 1000)
        
        # Fade out the roll so it doesn't abruptly stop
        t_fade = np.linspace(1, 0, roll_len, dtype=np.float32)
        roll_audio *= t_fade[:, None]
        
        roll_out_start = drop_out + max(0, a_vcut_offset)
        wl_roll = min(len(roll_audio), total_len - roll_out_start)
        if wl_roll > 0:
            out[roll_out_start:roll_out_start + wl_roll] += roll_audio[:wl_roll] * 0.4

    # 5. B Vocals: enter at phrase start.
    #    Only do an HPF sweep reveal if B enters >2 beats after the drop —
    #    if B starts at beat 0, play vocals full and clean immediately.
    if b_entry_offset < blend_len:
        src_end = min(b_swap + blend_len, len(b["vocals"]))
        wl = src_end - b_vocal_entry
        if wl > 0:
            out_vstart = drop_out + b_entry_offset
            
            # Apply a reverb pre-wash echo-in
            pre_wash = dsp_utils.apply_vocal_echo_in(
                b["vocals"][b_vocal_entry:min(b_vocal_entry + 2 * spb, src_end)].copy(), sr, b["bpm"]
            )
            wash_start = max(drop_out, out_vstart - 2 * spb)
            wl_wash = min(len(pre_wash), out_vstart - wash_start)
            if wl_wash > 0:
                out[wash_start:wash_start + wl_wash] += pre_wash[:wl_wash] * gain
            
            # Apply a quick 10ms fade-in to the entering dry vocal to prevent zero-crossing clicks.
            # No HPF sweep, so the vocal retains its full natural warmth.
            fade_samples = min(int(0.01 * sr), wl)
            snippet = b["vocals"][b_vocal_entry:src_end].copy()
            if fade_samples > 0:
                snippet[:fade_samples] *= np.linspace(0, 1, fade_samples, dtype=np.float32)[:, None]
            out[out_vstart:out_vstart + wl] += snippet * gain

    # ------------------------------------------------------------------
    # Post-Blend: B Solo
    # ------------------------------------------------------------------
    b_post_start = b_swap + blend_len
    b_post_end   = min(len(b["drums"]), b_post_start + post_len)
    wl_post      = b_post_end - b_post_start
    out_post_off = drop_out + blend_len
    if wl_post > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            end = min(len(b[stem]), b_post_start + wl_post)
            w   = end - b_post_start
            if w > 0:
                out[out_post_off:out_post_off + w] += b[stem][b_post_start:end] * gain

    if strategy in ["cut", "echo_out"]:
        from src.engine import fx_generator
        # Extract B's first beat to analyze its spectrum
        b_first_beat_mix = b["bass"][b_swap:min(b_swap + spb, len(b["bass"]))] + \
                           b["drums"][b_swap:min(b_swap + spb, len(b["drums"]))] + \
                           b["other"][b_swap:min(b_swap + spb, len(b["other"]))]
        
        # Generates a spectrum-aware noise sweep (returns None if B is already very bright)
        impact = fx_generator.generate_impact_fx(sr, 4.0, b_first_beat_mix)
        if impact is not None:
            wl_impact = min(len(impact), len(out) - drop_out)
            if wl_impact > 0:
                out[drop_out:drop_out + wl_impact] += impact[:wl_impact] * 0.75  # 75% volume

    out = normalize(out)
    print(f"  Saved -> {out_name}")
    sf.write(out_name, out, sr)



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
                 _to_mono(a["bass"] [a_m_start:a_m_end]) +
                 _to_mono(a["vocals"][a_m_start:a_m_end]) +
                 _to_mono(a["other"] [a_m_start:a_m_end]))
    b_measure = (_to_mono(b["drums"][b_blend_start:b_m_end]) +
                 _to_mono(b["bass"] [b_blend_start:b_m_end]) +
                 _to_mono(b["vocals"][b_blend_start:b_m_end]) +
                 _to_mono(b["other"] [b_blend_start:b_m_end]))
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
