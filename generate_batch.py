import os
import json
import librosa
import numpy as np
import soundfile as sf

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
CRATE_FILE = os.path.join(os.path.dirname(__file__), "library", "crate.json")

def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    if len(y.shape) > 1: y = np.mean(y, axis=1) # force mono
    return y, sr

def normalize(audio):
    peak = np.max(np.abs(audio))
    if peak > 0: return audio / peak
    return audio

def calculate_energy(audio):
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    return np.mean(rms[rms > 0.01]) if len(rms[rms > 0.01]) > 0 else 0.001

def detect_key(audio, sr):
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_sum = np.sum(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return pitch_classes[np.argmax(chroma_sum)]

def detect_first_drop(bass, beats, sr, bpm):
    # A drop usually happens between 30s and 120s
    # It's defined by a huge jump in BASS energy that sustains.
    bass_rms = librosa.feature.rms(y=bass, frame_length=4096, hop_length=1024)[0]
    smoothed_rms = np.convolve(bass_rms, np.ones(43)/43, mode='valid') # ~1 sec smoothing
    
    # We want to find the point where the energy is significantly higher than the previous 8 bars
    beats_per_sec = bpm / 60.0
    frames_per_sec = sr / 1024.0
    frames_8_bars = int(32 / beats_per_sec * frames_per_sec)
    
    best_drop_frame = 0
    max_contrast = 0
    
    # Start looking only after 30 seconds! Intros can be loud.
    start_frame = int(30 * frames_per_sec)
    end_frame = int(120 * frames_per_sec)
    
    for i in range(start_frame, min(end_frame, len(smoothed_rms))):
        # Average energy of the 8 bars BEFORE this frame
        pre_energy = np.mean(smoothed_rms[max(0, i-frames_8_bars):i])
        # Average energy of the 8 bars AFTER this frame
        post_energy = np.mean(smoothed_rms[i:min(len(smoothed_rms), i+frames_8_bars)])
        
        contrast = post_energy - pre_energy
        if contrast > max_contrast:
            max_contrast = contrast
            best_drop_frame = i
            
    if best_drop_frame == 0:
        return np.argmin(np.abs(beats - (60 * sr))) # Fallback
        
    drop_sample = librosa.frames_to_samples(best_drop_frame, hop_length=1024)
    # Snap to nearest beat
    drop_beat_idx = np.argmin(np.abs(beats - drop_sample))
    return drop_beat_idx

def get_track_data(crate, track_name):
    full_name = next(k for k in crate.keys() if track_name in k)
    data = crate[full_name]
    stems = data["stems"]
    
    vocals, sr = load_audio(stems["vocals"])
    drums, _ = load_audio(stems["drums"])
    bass, _ = load_audio(stems["bass"])
    other, _ = load_audio(stems["other"])
    
    mono = drums + bass + other
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=data["bpm"], units='samples')
    
    key = detect_key(other + bass, sr)
    drop_beat_idx = detect_first_drop(bass, beats, sr, data["bpm"])
    
    print(f"[{track_name}] Detected Key: {key} | First Drop: {beats[drop_beat_idx]/sr:.2f}s")
    
    return {
        "name": full_name,
        "bpm": data["bpm"],
        "key": key,
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "other": other,
        "beats": beats,
        "drop_idx": drop_beat_idx,
        "sr": sr
    }

def find_vocal_cut_point(vocal_stem, target_sample, sr, hop_length=512):
    rms = librosa.feature.rms(y=vocal_stem, frame_length=2048, hop_length=hop_length)[0]
    if np.max(rms) > 0: rms = rms / np.max(rms)
    target_frame = min(librosa.samples_to_frames(target_sample, hop_length=hop_length), len(rms) - 1)
    
    if rms[target_frame] < 0.05: return target_sample
    for f in range(target_frame, 0, -1):
        if rms[f] < 0.05: return librosa.frames_to_samples(f, hop_length=hop_length)
    return target_sample

def find_vocal_end_point(vocal_stem, start_sample, sr, hop_length=512):
    rms = librosa.feature.rms(y=vocal_stem, frame_length=2048, hop_length=hop_length)[0]
    if np.max(rms) > 0: rms = rms / np.max(rms)
    start_frame = min(librosa.samples_to_frames(start_sample, hop_length=hop_length), len(rms) - 1)
    
    silence_count = 0
    for f in range(start_frame, len(rms)):
        if rms[f] < 0.05:
            silence_count += 1
            if silence_count > 10: return librosa.frames_to_samples(f - 10, hop_length=hop_length)
        else: silence_count = 0
    return len(vocal_stem)

def find_vocal_start_point(vocal_stem, min_start_sample, sr, hop_length=512):
    rms = librosa.feature.rms(y=vocal_stem, frame_length=2048, hop_length=hop_length)[0]
    if np.max(rms) > 0: rms = rms / np.max(rms)
    start_frame = min(librosa.samples_to_frames(min_start_sample, hop_length=hop_length), len(rms) - 1)
    
    for f in range(start_frame, len(rms)):
        if rms[f] > 0.05: return librosa.frames_to_samples(f, hop_length=hop_length)
    return len(vocal_stem)

def generate_outro_to_intro(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f: crate = json.load(f)
    a, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    sr = a["sr"]
    
    samples_per_beat = int((60.0 / a["bpm"]) * sr)
    blend_length = 32 * samples_per_beat
    
    # Track A Outro (Target exactly 32 beats before drums end)
    target_a_end = len(a["drums"]) - blend_length - (samples_per_beat * 4)
    a_transition_beat = a["beats"][np.argmin(np.abs(a["beats"] - target_a_end))]
    
    # Track B Intro (First beat)
    b_start_beat = b["beats"][0] if len(b["beats"]) > 0 else 0
    
    # Get physical lengths
    actual_blend_len = min(blend_length, len(b["drums"]) - b_start_beat, len(a["drums"]) - a_transition_beat)
    preview_start = max(0, a_transition_beat - (15 * sr))
    preview_end = a_transition_beat + actual_blend_len + (15 * sr)
    
    out_length = preview_end - preview_start
    out_audio = np.zeros(out_length, dtype=np.float32)
    
    a_pre_end = a_transition_beat - preview_start
    out_audio[:a_pre_end] += a["drums"][preview_start:a_transition_beat] + a["bass"][preview_start:a_transition_beat] + a["other"][preview_start:a_transition_beat] + a["vocals"][preview_start:a_transition_beat]
    
    # Original Linear Crossfade
    fade_out = np.linspace(1.0, 0.0, actual_blend_len)
    fade_in = np.linspace(0.0, 1.0, actual_blend_len)
    
    blend_zone = np.zeros(actual_blend_len, dtype=np.float32)
    blend_zone += a["bass"][a_transition_beat:a_transition_beat+actual_blend_len] * fade_out
    blend_zone += a["drums"][a_transition_beat:a_transition_beat+actual_blend_len] * fade_out
    blend_zone += a["other"][a_transition_beat:a_transition_beat+actual_blend_len] * fade_out
    blend_zone += a["vocals"][a_transition_beat:a_transition_beat+actual_blend_len] * fade_out
    
    b_blend_end = b_start_beat + actual_blend_len
    blend_zone += b["bass"][b_start_beat:b_blend_end] * fade_in
    blend_zone += b["drums"][b_start_beat:b_blend_end] * fade_in
    blend_zone += b["other"][b_start_beat:b_blend_end] * fade_in
    blend_zone += b["vocals"][b_start_beat:b_blend_end] * fade_in
    
    out_audio[a_pre_end:a_pre_end+actual_blend_len] += blend_zone
    
    b_rest_start = b_start_beat + actual_blend_len
    b_rest_end = min(b_rest_start + (15 * sr), len(b["drums"]))
    write_len = b_rest_end - b_rest_start
    
    out_audio[a_pre_end+actual_blend_len:a_pre_end+actual_blend_len+write_len] += b["drums"][b_rest_start:b_rest_end] + b["bass"][b_rest_start:b_rest_end] + b["other"][b_rest_start:b_rest_end] + b["vocals"][b_rest_start:b_rest_end]
    
    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out_audio), sr)

def generate_high_energy_drop_swap(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f: crate = json.load(f)
    a, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    sr = a["sr"]
    
    # Master Phrase Alignment! We align Track A's 32-beat buildup to Track B's Drop
    a_drop_beat = a["beats"][a["drop_idx"]]
    b_drop_beat = b["beats"][b["drop_idx"]]
    
    # Buildup is EXACTLY 32 beats (1 phrase)
    a_build_start_idx = max(0, a["drop_idx"] - 32)
    a_build_start = a["beats"][a_build_start_idx]
    
    a_safe_vocal_cut = find_vocal_cut_point(a["vocals"], a_drop_beat - (5 * sr), sr)
    gain_ratio = min(max(calculate_energy(a["bass"]) / (calculate_energy(b["bass"]) + 0.001), 0.5), 2.0)
    
    out_length = (a_drop_beat - a_build_start) + (16 * sr)
    out_audio = np.zeros(out_length, dtype=np.float32)
    
    buildup_len = a_drop_beat - a_build_start
    hpf_washout_curve = np.linspace(1.0, 0.0, buildup_len) ** 2
    
    out_audio[:buildup_len] += a["drums"][a_build_start:a_drop_beat] 
    out_audio[:buildup_len] += a["bass"][a_build_start:a_drop_beat] * hpf_washout_curve
    out_audio[:buildup_len] += a["other"][a_build_start:a_drop_beat]
    
    vocal_play_length = a_safe_vocal_cut - a_build_start
    if vocal_play_length > 0: out_audio[:vocal_play_length] += a["vocals"][a_build_start:a_safe_vocal_cut]
    
    b_safe_vocal_start = find_vocal_start_point(b["vocals"], b_drop_beat + (5 * sr), sr)
    b_end = min(b_drop_beat + (16 * sr), len(b["drums"]))
    write_len = b_end - b_drop_beat
    
    out_audio[buildup_len:buildup_len+write_len] += b["drums"][b_drop_beat:b_end] + (b["bass"][b_drop_beat:b_end] * gain_ratio) + b["other"][b_drop_beat:b_end]
    
    b_vocal_delay = b_safe_vocal_start - b_drop_beat
    if b_vocal_delay < write_len: out_audio[buildup_len + b_vocal_delay : buildup_len + write_len] += b["vocals"][b_safe_vocal_start:b_end]
    
    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out_audio), sr)

def generate_high_energy_vocal_buildup_blend(track_a_str, track_b_str, out_name):
    print(f"Generating {out_name}...")
    with open(CRATE_FILE, 'r') as f: crate = json.load(f)
    a, b = get_track_data(crate, track_a_str), get_track_data(crate, track_b_str)
    sr = a["sr"]
    
    a_drop_beat = a["beats"][a["drop_idx"]]
    b_drop_beat = b["beats"][b["drop_idx"]]
    a_build_start_idx = max(0, a["drop_idx"] - 32)
    a_build_start = a["beats"][a_build_start_idx]
    
    a_vocal_end = find_vocal_end_point(a["vocals"], a_drop_beat, sr)
    a_vocal_bleed_length = a_vocal_end - a_drop_beat
    
    gain_ratio = min(max(calculate_energy(a["bass"]) / (calculate_energy(b["bass"]) + 0.001), 0.5), 2.0)
    
    out_length = (a_drop_beat - a_build_start) + (16 * sr)
    out_audio = np.zeros(out_length, dtype=np.float32)
    
    buildup_len = a_drop_beat - a_build_start
    hpf_washout_curve = np.linspace(1.0, 0.0, buildup_len) ** 2
    
    out_audio[:buildup_len] += a["drums"][a_build_start:a_drop_beat] + (a["bass"][a_build_start:a_drop_beat] * hpf_washout_curve) + a["other"][a_build_start:a_drop_beat]
    
    a_vocal_play_length = min(len(out_audio), a_vocal_end - a_build_start)
    if a_vocal_play_length > 0: out_audio[:a_vocal_play_length] += a["vocals"][a_build_start:a_build_start+a_vocal_play_length]
    
    b_end = min(b_drop_beat + (16 * sr), len(b["drums"]))
    write_len = b_end - b_drop_beat
    out_audio[buildup_len:buildup_len+write_len] += b["drums"][b_drop_beat:b_end] + (b["bass"][b_drop_beat:b_end] * gain_ratio) + b["other"][b_drop_beat:b_end]
    
    if a_vocal_bleed_length < write_len:
        out_audio[buildup_len + a_vocal_bleed_length : buildup_len + write_len] += b["vocals"][b_drop_beat + a_vocal_bleed_length : b_end]
    
    sf.write(os.path.join(OUTPUT_DIR, out_name), normalize(out_audio), sr)

if __name__ == "__main__":
    generate_outro_to_intro("Sky High", "Heroes Tonight", "1_LowEnergy_Outro-Intro_SkyHigh_to_HeroesTonight.wav")
    generate_outro_to_intro("Heroes Tonight", "Sky High", "2_LowEnergy_Outro-Intro_HeroesTonight_to_SkyHigh.wav")
    
    generate_high_energy_drop_swap("Sky High", "Heroes Tonight", "3_HighEnergy_Drop-Swap_SkyHigh_to_HeroesTonight.wav")
    generate_high_energy_vocal_buildup_blend("Heroes Tonight", "Sky High", "4_HighEnergy_Vocal-Buildup-Blend_HeroesTonight_to_SkyHigh.wav")
    print("Batch generation complete! Check the output folder.")
