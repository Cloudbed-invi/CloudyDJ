import os
import json
import librosa
import numpy as np
import soundfile as sf
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.engine.loop_roll import generate_loop_roll
from src.engine import dsp_utils

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "output"))
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

CRATE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "library", "crate.json"))

STRUCT_SR = 22050
STEM_SR   = 44100
STRUCT_SCALE = STEM_SR / STRUCT_SR


def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    return y, sr


def normalize(audio, headroom=0.85):
    peak = np.max(np.abs(audio))
    if peak > headroom:
        audio = audio * (headroom / peak)
    return audio


def calculate_energy(audio):
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    active = rms[rms > 0.01]
    return float(np.mean(active)) if len(active) > 0 else 0.001


def calculate_local_energy(track_data, start_sample, end_sample):
    """Measure combined RMS energy of ALL stems in a specific region."""
    if start_sample >= end_sample:
        return 0.001
    total = np.zeros(end_sample - start_sample, dtype=np.float32)
    for stem in ["drums", "bass", "other", "vocals"]:
        total += track_data[stem][start_sample:end_sample]
    rms = librosa.feature.rms(y=total, frame_length=2048, hop_length=512)[0]
    active = rms[rms > 0.01]
    return float(np.mean(active)) if len(active) > 0 else 0.001


def detect_key(audio, sr):
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitch_classes[np.argmax(chroma_sum)]


def detect_first_drop(bass, beats, sr, bpm):
    bass_rms = librosa.feature.rms(y=bass, frame_length=4096, hop_length=1024)[0]
    smoothed_rms = np.convolve(bass_rms, np.ones(43) / 43, mode='valid')
    beats_per_sec = bpm / 60.0
    frames_per_sec = sr / 1024.0
    frames_8_bars = int(32 / beats_per_sec * frames_per_sec)
    best_drop_frame = 0
    max_contrast = 0
    start_frame = int(30 * frames_per_sec)
    end_frame = int(120 * frames_per_sec)
    for i in range(start_frame, min(end_frame, len(smoothed_rms))):
        pre_energy = np.mean(smoothed_rms[max(0, i - frames_8_bars):i])
        post_energy = np.mean(smoothed_rms[i:min(len(smoothed_rms), i + frames_8_bars)])
        contrast = post_energy - pre_energy
        if contrast > max_contrast:
            max_contrast = contrast
            best_drop_frame = i
    if best_drop_frame == 0:
        return np.argmin(np.abs(beats - (60 * sr)))
    drop_sample = librosa.frames_to_samples(best_drop_frame, hop_length=1024)
    return np.argmin(np.abs(beats - drop_sample))


def get_track_data(crate, track_name):
    full_name = next(k for k in crate.keys() if track_name in k)
    data = crate[full_name]
    stems = data["stems"]
    vocals, sr = load_audio(stems["vocals"])
    drums, _ = load_audio(stems["drums"])
    bass, _ = load_audio(stems["bass"])
    other, _ = load_audio(stems["other"])
    mono = drums + bass + other
    _, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=data["bpm"], units='samples')
    key = detect_key(other + bass, sr)
    drop_beat_idx = detect_first_drop(bass, beats, sr, data["bpm"])
    segs_raw = data.get("segments", [])
    segments = [
        {
            "label":        s["label"],
            "start_sample": int(s["start_sample"] * STRUCT_SCALE),
            "end_sample":   int(s["end_sample"] * STRUCT_SCALE),
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


def apply_warping(a, b_bpm):
    rate = a["bpm"] / b_bpm
    a_warped = {
        "bpm":    b_bpm,
        "sr":     a["sr"],
        "vocals": librosa.effects.time_stretch(a["vocals"], rate=rate),
        "drums":  librosa.effects.time_stretch(a["drums"],  rate=rate),
        "bass":   librosa.effects.time_stretch(a["bass"],   rate=rate),
        "other":  librosa.effects.time_stretch(a["other"],  rate=rate),
    }
    mono = a_warped["drums"] + a_warped["bass"] + a_warped["other"]
    _, beats = librosa.beat.beat_track(y=mono, sr=a["sr"], bpm=b_bpm, units='samples')
    a_warped["beats"] = beats
    a_drop_sample_orig = a["beats"][a["drop_idx"]]
    a_drop_sample_warped = int(a_drop_sample_orig / rate)
    a_warped["drop_idx"] = np.argmin(np.abs(beats - a_drop_sample_warped))
    a_warped["segments"] = [
        {**s,
         "start_sample": int(s["start_sample"] / rate),
         "end_sample":   int(s["end_sample"] / rate),
         }
        for s in a.get("segments", [])
    ]
    return a_warped


# ---------------------------------------------------------------------------
# 1 & 2 – Outro → Intro (EQ Staggered Crossfade, not a volume fade)
# ---------------------------------------------------------------------------

def generate_outro_to_intro(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    a_orig = get_track_data(crate, track_a_str)
    b = get_track_data(crate, track_b_str)
    a = apply_warping(a_orig, b["bpm"])
    sr = a["sr"]
    spb = int((60.0 / b["bpm"]) * sr)

    # T1 FIX: Use proper outro for A, and energetic intro for B (not silence)
    a_outro_seg = find_segment(a, "outro")
    a_blend_start = int(a_outro_seg["start_sample"]) if a_outro_seg else max(0, len(a["drums"]) - 32 * spb)
    a_blend_start = max(0, min(a_blend_start, len(a["drums"]) - 2))

    b_blend_start = dsp_utils.find_instrumental_intro(b, sr)

    blend_beats = 16
    blend_len = blend_beats * spb

    # Energy normalization: match A's outro energy for B's intro
    a_e = calculate_local_energy(a, a_blend_start, a_blend_start + blend_len)
    b_e = calculate_local_energy(b, b_blend_start, b_blend_start + blend_len)
    gain_ratio = min(max(a_e / (b_e + 0.0001), 0.6), 1.8)

    pre_len = int(10 * sr)
    post_len = int(10 * sr)
    out = np.zeros(pre_len + blend_len + post_len, dtype=np.float32)

    # Pre: 10s of A solo
    a_pre_start = max(0, a_blend_start - pre_len)
    actual_pre = a_blend_start - a_pre_start
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len - actual_pre:pre_len] += a[stem][a_pre_start:a_blend_start]

    # T1/T2 FIX: Staggered EQ crossfade (real DJ style):
    # - A's bass fades out beats 1-4
    # - A's mids/highs fade out beats 1-16
    # - B's mids/highs fade in beats 1-12
    # - B's bass fades in beats 12-16
    blend = np.zeros(blend_len, dtype=np.float32)
    
    a_end = min(a_blend_start + blend_len, len(a["drums"]))
    a_act = a_end - a_blend_start
    b_end = min(b_blend_start + blend_len, len(b["drums"]))
    b_act = b_end - b_blend_start

    if a_act > 0:
        n = a_act
        # A bass: full -> 0 over first 4 beats
        bass_fade_len = min(n, 4 * spb)
        bass_fade = np.concatenate([
            np.linspace(1.0, 0.0, bass_fade_len),
            np.zeros(n - bass_fade_len)
        ]).astype(np.float32)
        blend[:n] += a["bass"][a_blend_start:a_end] * bass_fade

        # A drums/other/vocals: full -> 0 over all 16 beats
        full_fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
        blend[:n] += a["drums"][a_blend_start:a_end] * full_fade
        blend[:n] += a["other"][a_blend_start:a_end] * full_fade
        blend[:n] += a["vocals"][a_blend_start:a_end] * full_fade

    if b_act > 0:
        n = b_act
        # B mids/drums: 0 -> full over first 12 beats
        mid_fade_len = min(n, 12 * spb)
        mid_fade = np.concatenate([
            np.linspace(0.0, 1.0, mid_fade_len),
            np.ones(n - mid_fade_len)
        ]).astype(np.float32) * gain_ratio
        blend[:n] += b["drums"][b_blend_start:b_end] * mid_fade
        blend[:n] += b["other"][b_blend_start:b_end] * mid_fade
        
        # Vocals: HPF reveal. During the blend, they are heavily filtered (thin)
        b_vocals_blend = b["vocals"][b_blend_start:b_end]
        b_vocals_filtered = dsp_utils.apply_hpf_sweep(b_vocals_blend, sr, start_freq=2000, end_freq=2000, num_chunks=1)
        blend[:n] += b_vocals_filtered * mid_fade

        # B bass: 0 -> full over beats 12-16
        bass_in_len = min(n, 4 * spb)
        bass_in_fade = np.concatenate([
            np.zeros(max(0, n - bass_in_len)),
            np.linspace(0.0, 1.0, min(bass_in_len, n))
        ]).astype(np.float32) * gain_ratio
        blend[:n] += b["bass"][b_blend_start:b_end] * bass_in_fade

    # Echo tail from A's last vocal
    vocal_snip = a["vocals"][max(0, a_blend_start - spb):a_blend_start]
    echo_tail = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
    tail_len = min(len(echo_tail), blend_len)
    if tail_len > 0:
        blend[:tail_len] += echo_tail[:tail_len]

    out[pre_len:pre_len + blend_len] = blend

    # Post: 10s of B solo
    # EXTENDED MIX TRICK: Repeat the same 16 beats of B's intro we just blended, but this time WITH vocals!
    b_post_start = b_blend_start
    b_post_end = min(b_post_start + post_len, len(b["drums"]))
    actual_post = b_post_end - b_post_start
    if actual_post > 0:
        for stem in ["drums", "bass", "other"]:
            out[pre_len + blend_len:pre_len + blend_len + actual_post] += b[stem][b_post_start:b_post_end] * gain_ratio
            
        # Vocals HPF Sweep Reveal: Sweep down from 2000Hz to 80Hz over the first 8 beats
        sweep_len = min(8 * spb, actual_post)
        crossfade_len = min(2 * spb, actual_post - sweep_len) if actual_post > sweep_len else 0
        total_sweep_len = sweep_len + crossfade_len
        
        if total_sweep_len > 0:
            b_vocals_sweep = b["vocals"][b_post_start:b_post_start + total_sweep_len]
            revealed_vocals = dsp_utils.apply_hpf_sweep(b_vocals_sweep, sr, start_freq=2000, end_freq=80, num_chunks=16)
            
            # Apply fade out to the last 2 beats of the sweep, and fade in to the raw vocal
            if crossfade_len > 0:
                fade_out = np.linspace(1.0, 0.0, crossfade_len, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, crossfade_len, dtype=np.float32)
                revealed_vocals[sweep_len:] *= fade_out
                
                raw_crossfade = b["vocals"][b_post_start + sweep_len:b_post_start + total_sweep_len] * fade_in
                revealed_vocals[sweep_len:] += raw_crossfade

            out[pre_len + blend_len:pre_len + blend_len + total_sweep_len] += revealed_vocals * gain_ratio
            
        # Rest of vocals at full
        if actual_post > total_sweep_len:
            out[pre_len + blend_len + total_sweep_len:pre_len + blend_len + actual_post] += b["vocals"][b_post_start + total_sweep_len:b_post_end] * gain_ratio

    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out), sr)


# ---------------------------------------------------------------------------
# 2 – High Energy Drop Swap
# ---------------------------------------------------------------------------

def generate_high_energy_drop_swap(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    a_orig, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    a = apply_warping(a_orig, b["bpm"])
    sr = a["sr"]

    a_drop_beat = int(a["beats"][a["drop_idx"]])
    b_drop_beat = int(b["beats"][b["drop_idx"]])
    a_build_start_idx = max(0, a["drop_idx"] - 32)
    a_build_start = int(a["beats"][a_build_start_idx])
    spb = int((60.0 / a["bpm"]) * sr)

    pre_len = int(10 * sr)
    post_len = int(10 * sr)
    buildup_len = a_drop_beat - a_build_start

    out_length = pre_len + buildup_len + post_len
    out_audio = np.zeros(out_length, dtype=np.float32)

    a_pre_start = max(0, a_build_start - pre_len)
    actual_pre = a_build_start - a_pre_start
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out_audio[pre_len - actual_pre:pre_len] += a[stem][a_pre_start:a_build_start]

    # T2: Bass stays FULL for first 28 beats, only fades in final 4 beats
    out_audio[pre_len:pre_len + buildup_len] += a["drums"][a_build_start:a_drop_beat]
    out_audio[pre_len:pre_len + buildup_len] += a["other"][a_build_start:a_drop_beat]
    out_audio[pre_len:pre_len + buildup_len] += a["vocals"][a_build_start:a_drop_beat]

    # Bass: full for 28 beats, fade to 0 in final 4 beats only
    bass_full_len = max(0, buildup_len - 4 * spb)
    bass_fade_len = buildup_len - bass_full_len
    if bass_full_len > 0:
        out_audio[pre_len:pre_len + bass_full_len] += a["bass"][a_build_start:a_build_start + bass_full_len]
    if bass_fade_len > 0:
        bass_fade = np.linspace(1.0, 0.0, bass_fade_len, dtype=np.float32)
        out_audio[pre_len + bass_full_len:pre_len + buildup_len] += (
            a["bass"][a_build_start + bass_full_len:a_drop_beat] * bass_fade
        )

    # Echo tail on last vocal beat before drop
    vocal_scan_region = a["vocals"][max(0, a_drop_beat - 4 * spb):a_drop_beat]
    vocal_energy = calculate_energy(vocal_scan_region)
    echo_beats = 4 if vocal_energy > 0.01 else 2
    
    last_beat_start = max(0, a_drop_beat - spb)
    vocal_snippet = a["vocals"][last_beat_start:a_drop_beat]
    echo_tail = dsp_utils.generate_echo_tail(vocal_snippet, sr, a["bpm"], beats=echo_beats)

    # Local energy comparison: Last 4 beats of A vs First 4 beats of B
    a_local_e = calculate_local_energy(a, max(0, a_drop_beat - 4 * spb), a_drop_beat)
    b_local_e = calculate_local_energy(b, b_drop_beat, min(b_drop_beat + 4 * spb, len(b["drums"])))
    gain_ratio = min(max(a_local_e / (b_local_e + 0.001), 0.5), 1.5)

    b_end = min(b_drop_beat + post_len, len(b["drums"]))
    write_len = b_end - b_drop_beat
    post_start_idx = pre_len + buildup_len

    # B drops: all stems (vocals fade in over 4 beats)
    vocal_fade_len = min(4 * spb, write_len)
    vocal_fade = np.concatenate([
        np.linspace(0.0, 1.0, vocal_fade_len),
        np.ones(write_len - vocal_fade_len)
    ]).astype(np.float32)
    
    out_audio[post_start_idx:post_start_idx + write_len] += (
        b["drums"][b_drop_beat:b_end] +
        b["bass"][b_drop_beat:b_end] * gain_ratio +
        b["other"][b_drop_beat:b_end] +
        b["vocals"][b_drop_beat:b_end] * vocal_fade
    )
    tail_len = min(len(echo_tail), write_len)
    if tail_len > 0:
        out_audio[post_start_idx:post_start_idx + tail_len] += echo_tail[:tail_len]

    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out_audio), sr)


# ---------------------------------------------------------------------------
# 3 – High Energy Vocal Buildup Cut
# ---------------------------------------------------------------------------

def generate_high_energy_vocal_buildup_blend(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    a_orig, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    a = apply_warping(a_orig, b["bpm"])
    sr = a["sr"]
    spb = int((60.0 / b["bpm"]) * sr)

    a_drop_beat = int(a["beats"][a["drop_idx"]])
    b_drop_beat = int(b["beats"][b["drop_idx"]])
    a_build_start_idx = max(0, a["drop_idx"] - 32)
    a_build_start = int(a["beats"][a_build_start_idx])

    # T3 FIX: Use full-track RMS to find peak energy before cutting
    a_vocal_cut = dsp_utils.find_vocal_cutoff_in_buildup(a, a_build_start, a_drop_beat, sr)
    
    buildup_len = a_vocal_cut - a_build_start
    
    if buildup_len <= 0:
        # Fallback to full buildup
        buildup_len = a_drop_beat - a_build_start
        a_vocal_cut = a_drop_beat

    a_local_e = calculate_local_energy(a, max(0, a_vocal_cut - 4 * spb), a_vocal_cut)
    b_local_e = calculate_local_energy(b, b_drop_beat, min(b_drop_beat + 4 * spb, len(b["drums"])))
    gain_ratio = min(max(a_local_e / (b_local_e + 0.001), 0.5), 1.5)

    pre_len = int(10 * sr)
    post_len = int(10 * sr)
    out = np.zeros(pre_len + buildup_len + post_len, dtype=np.float32)

    a_pre_start = max(0, a_build_start - pre_len)
    actual_pre = a_build_start - a_pre_start
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len - actual_pre:pre_len] += a[stem][a_pre_start:a_build_start]

    # Buildup up to the exact vocal cut (all stems cut early!)
    out[pre_len:pre_len + buildup_len] += a["drums"][a_build_start:a_vocal_cut]
    out[pre_len:pre_len + buildup_len] += a["other"][a_build_start:a_vocal_cut]
    out[pre_len:pre_len + buildup_len] += a["vocals"][a_build_start:a_vocal_cut]
    
    bass_full_len = max(0, buildup_len - 4 * spb)
    bass_fade_len = buildup_len - bass_full_len
    if bass_full_len > 0:
        out[pre_len:pre_len + bass_full_len] += a["bass"][a_build_start:a_build_start + bass_full_len]
    if bass_fade_len > 0:
        bass_fade = np.linspace(1.0, 0.0, bass_fade_len, dtype=np.float32)
        out[pre_len + bass_full_len:pre_len + buildup_len] += (
            a["bass"][a_build_start + bass_full_len:a_vocal_cut] * bass_fade
        )

    # B drops EXACTLY at the cut point!
    b_end = min(b_drop_beat + post_len, len(b["drums"]))
    write_len = b_end - b_drop_beat
    post_start_idx = pre_len + buildup_len

    out[post_start_idx:post_start_idx + write_len] += (
        b["drums"][b_drop_beat:b_end] * gain_ratio +
        b["bass"][b_drop_beat:b_end] * gain_ratio +
        b["other"][b_drop_beat:b_end] * gain_ratio +
        b["vocals"][b_drop_beat:b_end] * gain_ratio
    )

    # Echo tail on last vocal
    last_beat_start = max(0, a_vocal_cut - spb)
    vocal_snippet = a["vocals"][last_beat_start:a_vocal_cut]
    echo_tail = dsp_utils.generate_echo_tail(vocal_snippet, sr, a["bpm"], beats=4)
    tail_len = min(len(echo_tail), post_len)
    if tail_len > 0:
        out[post_start_idx:post_start_idx + tail_len] += echo_tail[:tail_len]

    # B drops immediately at vocal cutoff (T3 FIX: removed duplicate write)
    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out), sr)


# ---------------------------------------------------------------------------
# 5 – Loop Roll Tension (Whisper Edition)
# ---------------------------------------------------------------------------

def generate_loop_roll_tension(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    a_orig, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    a = apply_warping(a_orig, b["bpm"])
    sr = a["sr"]

    a_drop_beat = int(a["beats"][a["drop_idx"]])
    b_drop_beat = int(b["beats"][b["drop_idx"]])
    a_build_start = int(a["beats"][max(0, a["drop_idx"] - 32)])
    spb = int((60.0 / b["bpm"]) * sr)

    buildup_len = a_drop_beat - a_build_start
    roll_len = 8 * spb
    norm_build_len = max(0, buildup_len - roll_len)

    pre_len = int(10 * sr)
    post_len = int(10 * sr)
    out = np.zeros(pre_len + buildup_len + post_len, dtype=np.float32)

    a_pre_start = max(0, a_build_start - pre_len)
    actual_pre = a_build_start - a_pre_start
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len - actual_pre:pre_len] += a[stem][a_pre_start:a_build_start]

    # First 24 beats: full track plays (bass fades in last 4 beats)
    spb_a = spb
    bass_full_len = max(0, norm_build_len - 4 * spb_a)
    bass_fade_len = norm_build_len - bass_full_len

    if norm_build_len > 0:
        out[pre_len:pre_len + norm_build_len] += a["drums"][a_build_start:a_build_start + norm_build_len]
        out[pre_len:pre_len + norm_build_len] += a["other"][a_build_start:a_build_start + norm_build_len]
        out[pre_len:pre_len + norm_build_len] += a["vocals"][a_build_start:a_build_start + norm_build_len]

        if bass_full_len > 0:
            out[pre_len:pre_len + bass_full_len] += a["bass"][a_build_start:a_build_start + bass_full_len]
        if bass_fade_len > 0:
            bf = np.linspace(1.0, 0.0, bass_fade_len, dtype=np.float32)
            out[pre_len + bass_full_len:pre_len + norm_build_len] += (
                a["bass"][a_build_start + bass_full_len:a_build_start + norm_build_len] * bf
            )

    # Final 8 beats: loop roll — apply bandpass to vocal ONLY for loop generation
    clean_vocals = dsp_utils.apply_bandpass(a["vocals"], sr, low=250, high=5000)
    word_s, word_e = dsp_utils.find_best_loop_source(a, a_drop_beat, b["bpm"], sr)
    
    loop_roll_audio = generate_loop_roll(
        clean_vocals, b["bpm"], sr,
        drop_sample_idx=a_drop_beat,
        num_beats=8,
        exact_word_start=word_s,
        exact_word_end=word_e
    )

    roll_start = pre_len + norm_build_len
    roll_write_len = min(len(loop_roll_audio), roll_len)
    if roll_write_len > 0:
        out[roll_start:roll_start + roll_write_len] += loop_roll_audio[:roll_write_len]

    # Energy normalization for B drop vs A buildup
    a_local_e = calculate_local_energy(a, max(0, a_drop_beat - 4 * spb), a_drop_beat)
    b_local_e = calculate_local_energy(b, b_drop_beat, min(b_drop_beat + 4 * spb, len(b["drums"])))
    gain_ratio = min(max(a_local_e / (b_local_e + 0.001), 0.5), 1.5)

    # Ramp up loop roll volume smoothly into B's drop gain ratio
    if roll_write_len > 0:
        roll_ramp_len = min(2 * spb, roll_write_len)
        ramp = np.linspace(1.0, gain_ratio, roll_ramp_len, dtype=np.float32)
        out[roll_start + roll_write_len - roll_ramp_len:roll_start + roll_write_len] *= ramp
    b_end = min(b_drop_beat + post_len, len(b["drums"]))
    write_len = b_end - b_drop_beat
    post_start = pre_len + buildup_len
    out[post_start:post_start + write_len] += (
        b["drums"][b_drop_beat:b_end] +
        b["bass"][b_drop_beat:b_end] * gain_ratio +
        b["other"][b_drop_beat:b_end] +
        b["vocals"][b_drop_beat:b_end]
    )

    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out), sr)


# ---------------------------------------------------------------------------
# 6 – True Bass Swap (swap happens BEFORE A's drop, not after)
# ---------------------------------------------------------------------------

def generate_bass_swap_transition(track_a_str, track_b_str, out_name):
    """
    Real DJ Bass Swap:
    1. A plays full (including its drop)
    2. 16 beats AFTER A's drop: A's bass is CUT. B's bass starts FULL.
       A's mids/highs fade out. B's mids/highs fade in.
    3. B's drop: full energy

    BUT we now add a noise sweep at the swap point to mask the cut.
    Also: we play 10s of A at its DROP energy (not post-drop silence).
    """
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    a_orig, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    a = apply_warping(a_orig, b["bpm"])
    sr = a["sr"]
    spb = int((60.0 / b["bpm"]) * sr)

    a_drop = int(a["beats"][a["drop_idx"]])
    b_drop = int(b["beats"][b["drop_idx"]])

    # T6 FIX: The 10s pre-section starts DURING A's drop (full energy), not post-drop
    # The swap happens at a_drop + 10s... but ensure it's a beat boundary
    a_swap_anchor = a_drop  # swap happens at A's drop itself
    b_swap_anchor = max(0, b_drop - 16 * spb)  # B starts 16 beats before its drop

    pre_len = int(10 * sr)
    blend_len = 16 * spb
    post_len = int(10 * sr)
    out_length = pre_len + blend_len + post_len
    out = np.zeros(out_length, dtype=np.float32)

    # 10s of A playing at its drop (full energy, all stems)
    a_pre_start = max(0, a_swap_anchor - pre_len)
    actual_pre = a_swap_anchor - a_pre_start
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len - actual_pre:pre_len] += a[stem][a_pre_start:a_swap_anchor]

    # White noise sweep centered at the swap point
    noise_len = min(int(1.5 * spb), pre_len // 2)
    noise = dsp_utils.generate_white_noise_sweep(noise_len, sr)
    noise_start = pre_len - noise_len // 2
    noise_start = max(0, noise_start)
    noise_end = min(noise_start + noise_len, len(out))
    actual_noise = noise_end - noise_start
    if actual_noise > 0:
        out[noise_start:noise_end] += noise[:actual_noise]

    # 16-beat blend: A mids/highs fade out, B mids/highs fade in, B bass full immediately
    a_blend_end = min(a_swap_anchor + blend_len, len(a["drums"]))
    actual_blend = a_blend_end - a_swap_anchor
    b_blend_end = min(b_swap_anchor + blend_len, len(b["drums"]))
    b_blend_actual = b_blend_end - b_swap_anchor

    if actual_blend > 0:
        n = actual_blend
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        # A mids/highs fade out (drums = high energy, other = mids, vocals = mids/highs)
        out[pre_len:pre_len + n] += a["drums"][a_swap_anchor:a_blend_end] * fade_out
        out[pre_len:pre_len + n] += a["other"][a_swap_anchor:a_blend_end] * fade_out
        out[pre_len:pre_len + n] += a["vocals"][a_swap_anchor:a_blend_end] * fade_out
        # A's bass = ZERO (hard cut)

    if b_blend_actual > 0:
        n = min(b_blend_actual, actual_blend)
        fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
        # B mids/highs fade in
        out[pre_len:pre_len + n] += b["drums"][b_swap_anchor:b_swap_anchor + n] * fade_in
        out[pre_len:pre_len + n] += b["other"][b_swap_anchor:b_swap_anchor + n] * fade_in
        out[pre_len:pre_len + n] += b["vocals"][b_swap_anchor:b_swap_anchor + n] * fade_in
        # B's bass = FULL immediately (that's the swap!)
        out[pre_len:pre_len + n] += b["bass"][b_swap_anchor:b_swap_anchor + n]

    # Post: 10s of B playing at/after its drop
    b_post_start = b_swap_anchor + blend_len  # this lands exactly at B's drop
    b_post_end = min(b_post_start + post_len, len(b["drums"]))
    actual_post = b_post_end - b_post_start
    if actual_post > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            out[pre_len + blend_len:pre_len + blend_len + actual_post] += b[stem][b_post_start:b_post_end]

    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out), sr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_a1 = "Sky High"
    t_b1 = "Heroes Tonight"
    t_a2 = "On & On"
    t_b2 = "Blank"

    # The 4 Core Techniques
    generate_outro_to_intro(t_a1, t_b1, f"1_Outro_to_Intro_{t_a1.split()[-1]}_to_{t_b1.split()[-1]}.wav")
    generate_high_energy_drop_swap(t_a1, t_b1, f"2_HighEnergy_DropSwap_{t_a1.split()[-1]}_to_{t_b1.split()[-1]}.wav")
    generate_high_energy_vocal_buildup_blend(t_a2, t_b2, f"3_Vocal_Buildup_Cut_{t_a2.split()[-1]}_to_{t_b2.split()[-1]}.wav")
    generate_loop_roll_tension(t_a2, t_b2, f"4_Loop_Roll_Tension_{t_a2.split()[-1]}_to_{t_b2.split()[-1]}.wav")

    print("Batch generation complete! Check the output folder.")
