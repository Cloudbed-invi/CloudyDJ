import os
import json
import librosa
import soundfile as sf
import numpy as np

CRATE_FILE = os.path.join(os.path.dirname(__file__), "library", "crate.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

def load_audio(path, sr=44100):
    data, samplerate = sf.read(path, dtype='float32')
    if samplerate != sr:
        data = librosa.resample(data.T, orig_sr=samplerate, target_sr=sr).T
    if data.ndim == 1:
        data = np.column_stack((data, data))
    return data, sr

def normalize(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio

def generate_echo_tail(audio_snippet, sr, bpm, beats=8):
    """Takes a small audio snippet (e.g. the last vocal word) and creates a pure echo tail."""
    delay_samples = int((60.0 / bpm) * sr)
    tail_length = delay_samples * beats
    out = np.zeros((tail_length, 2), dtype=np.float32)
    
    # Place original snippet at the start
    snip_len = min(len(audio_snippet), tail_length)
    out[:snip_len] = audio_snippet[:snip_len]
    
    # Generate bouncing delays
    feedback = 0.65
    for i in range(delay_samples, tail_length):
        out[i] += out[i - delay_samples] * feedback
        
    return out

def execute_echo_out_drop_swap(track_a_name, track_b_name, track_a_bpm, track_b_bpm):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
        
    print(f"Loading {track_a_name} stems...")
    a_vocals, sr = load_audio(crate[track_a_name]["stems"]["vocals"])
    a_drums, _ = load_audio(crate[track_a_name]["stems"]["drums"])
    a_bass, _ = load_audio(crate[track_a_name]["stems"]["bass"])
    a_other, _ = load_audio(crate[track_a_name]["stems"]["other"])
    
    print(f"Loading {track_b_name} stems...")
    b_vocals, _ = load_audio(crate[track_b_name]["stems"]["vocals"])
    b_drums, _ = load_audio(crate[track_b_name]["stems"]["drums"])
    b_bass, _ = load_audio(crate[track_b_name]["stems"]["bass"])
    b_other, _ = load_audio(crate[track_b_name]["stems"]["other"])
    
    print("Analyzing beat grids...")
    a_mono = np.mean(a_drums + a_bass + a_other, axis=1)
    _, a_beats = librosa.beat.beat_track(y=a_mono, sr=sr, bpm=track_a_bpm, units='samples')
    
    b_mono = np.mean(b_drums + b_bass + b_other, axis=1)
    _, b_beats = librosa.beat.beat_track(y=b_mono, sr=sr, bpm=track_b_bpm, units='samples')
    
    target_sample = 60 * sr
    a_transition_beat = a_beats[np.argmin(np.abs(a_beats - target_sample))]
    b_start_beat = b_beats[0] if len(b_beats) > 0 else 0
    
    samples_per_beat_a = int((60.0 / track_a_bpm) * sr)
    cut_point = a_transition_beat - (samples_per_beat_a * 8) # Cut bass 8 beats early!
    if cut_point < 0: cut_point = 0
    
    b_length = len(b_drums) - b_start_beat
    output_length = a_transition_beat + b_length
    
    out_audio = np.zeros((output_length, 2), dtype=np.float32)
    
    print("Mixing tracks (HARD CUT Echo-Out)...")
    
    # 1. Track A plays normally until 8 beats before drop
    out_audio[:cut_point] += a_drums[:cut_point]
    out_audio[:cut_point] += a_bass[:cut_point]
    out_audio[:cut_point] += a_other[:cut_point]
    out_audio[:cut_point] += a_vocals[:cut_point]
    
    # 2. Tension Zone (8 beats before drop) -> NO BASS OR KICK DRUMS
    # High-passed effect by just using Vocals and Other stems!
    out_audio[cut_point:a_transition_beat] += a_vocals[cut_point:a_transition_beat]
    out_audio[cut_point:a_transition_beat] += a_other[cut_point:a_transition_beat]
    
    # 3. Exactly at the transition beat -> HARD CUT TRACK A. (NO FADE!)
    # Generate a pure echo tail from the last 1 beat of Track A's vocals
    last_beat_start = a_transition_beat - samples_per_beat_a
    vocal_snippet = a_vocals[last_beat_start:a_transition_beat]
    echo_tail = generate_echo_tail(vocal_snippet, sr, track_a_bpm, beats=16) # 16 beats of echo ring out
    
    # 4. Drop Track B instantly!
    drop_end = a_transition_beat + b_length
    out_audio[a_transition_beat:drop_end] += b_drums[b_start_beat:]
    out_audio[a_transition_beat:drop_end] += b_bass[b_start_beat:]
    out_audio[a_transition_beat:drop_end] += b_other[b_start_beat:]
    out_audio[a_transition_beat:drop_end] += b_vocals[b_start_beat:]
    
    # 5. Layer the pure echo tail over Track B's drop
    tail_len = min(len(echo_tail), drop_end - a_transition_beat)
    out_audio[a_transition_beat:a_transition_beat+tail_len] += echo_tail[:tail_len]
    
    out_audio = normalize(out_audio)
    out_file = os.path.join(OUTPUT_DIR, "drop_swap_mix.wav")
    print(f"Exporting final mix to {out_file}...")
    sf.write(out_file, out_audio, sr)
    print("Done!")

def execute_smooth_eq_blend(track_a_name, track_b_name, track_a_bpm, track_b_bpm):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
        
    print(f"Loading {track_a_name} stems...")
    a_vocals, sr = load_audio(crate[track_a_name]["stems"]["vocals"])
    a_drums, _ = load_audio(crate[track_a_name]["stems"]["drums"])
    a_bass, _ = load_audio(crate[track_a_name]["stems"]["bass"])
    a_other, _ = load_audio(crate[track_a_name]["stems"]["other"])
    
    print(f"Loading {track_b_name} stems...")
    b_vocals, _ = load_audio(crate[track_b_name]["stems"]["vocals"])
    b_drums, _ = load_audio(crate[track_b_name]["stems"]["drums"])
    b_bass, _ = load_audio(crate[track_b_name]["stems"]["bass"])
    b_other, _ = load_audio(crate[track_b_name]["stems"]["other"])
    
    # Time-stretch Track B to match Track A perfectly if there is a tiny variance
    if abs(track_a_bpm - track_b_bpm) > 0.1:
        stretch_factor = track_b_bpm / track_a_bpm
        print(f"Time-stretching Track B by factor {stretch_factor:.3f}...")
        b_drums = librosa.effects.time_stretch(b_drums.T, rate=stretch_factor).T
        b_bass = librosa.effects.time_stretch(b_bass.T, rate=stretch_factor).T
        b_vocals = librosa.effects.time_stretch(b_vocals.T, rate=stretch_factor).T
        b_other = librosa.effects.time_stretch(b_other.T, rate=stretch_factor).T
    
    print("Analyzing beat grids...")
    a_mono = np.mean(a_drums + a_bass + a_other, axis=1)
    _, a_beats = librosa.beat.beat_track(y=a_mono, sr=sr, bpm=track_a_bpm, units='samples')
    
    b_mono = np.mean(b_drums + b_bass + b_other, axis=1)
    _, b_beats = librosa.beat.beat_track(y=b_mono, sr=sr, bpm=track_a_bpm, units='samples')
    
    # FLAWLESS DJ BLEND: Outro to Intro!
    # A true DJ blends the last 32 beats of Track A into the first 32 beats of Track B.
    # This prevents vocals from clashing because intros and outros are usually just beats.
    
    blend_beats = 32
    samples_per_beat = int((60.0 / track_a_bpm) * sr)
    blend_length = blend_beats * samples_per_beat
    
    # Find the beat closest to the end of Track A (minus the blend length)
    target_a_end = len(a_drums) - blend_length - (samples_per_beat * 4) # pad by 4 beats
    a_transition_beat = a_beats[np.argmin(np.abs(a_beats - target_a_end))]
    
    # Track B starts exactly on its first major beat
    b_start_beat = b_beats[0] if len(b_beats) > 0 else 0
    
    b_length = len(b_drums) - b_start_beat
    output_length = a_transition_beat + b_length
    
    out_audio = np.zeros((output_length, 2), dtype=np.float32)
    
    print("Mixing tracks (FLAWLESS OUTRO-TO-INTRO BLEND)...")
    
    # Track A plays normally until the transition beat
    out_audio[:a_transition_beat] += a_drums[:a_transition_beat]
    out_audio[:a_transition_beat] += a_bass[:a_transition_beat]
    out_audio[:a_transition_beat] += a_other[:a_transition_beat]
    out_audio[:a_transition_beat] += a_vocals[:a_transition_beat]
    
    # Crossfade Zone (32 beats)
    actual_blend_len = min(blend_length, len(b_drums) - b_start_beat, len(a_drums) - a_transition_beat)
    
    # True DJ EQ curve:
    # Track A's Bass drops out almost instantly (exponential fade)
    # Track B's Bass cuts in smoothly
    fade_out_bass = np.linspace(1.0, 0.0, actual_blend_len) ** 4
    fade_out_highs = np.linspace(1.0, 0.0, actual_blend_len)
    
    fade_in_bass = np.linspace(0.0, 1.0, actual_blend_len) ** 2
    fade_in_highs = np.linspace(0.0, 1.0, actual_blend_len)
    
    blend_zone = np.zeros((actual_blend_len, 2), dtype=np.float32)
    
    # Fade out A
    blend_zone += a_bass[a_transition_beat:a_transition_beat+actual_blend_len] * fade_out_bass.reshape(-1, 1)
    blend_zone += a_drums[a_transition_beat:a_transition_beat+actual_blend_len] * fade_out_highs.reshape(-1, 1)
    blend_zone += a_other[a_transition_beat:a_transition_beat+actual_blend_len] * fade_out_highs.reshape(-1, 1)
    blend_zone += a_vocals[a_transition_beat:a_transition_beat+actual_blend_len] * fade_out_highs.reshape(-1, 1)
    
    # Fade in B
    b_blend_start = b_start_beat
    b_blend_end = b_start_beat + actual_blend_len
    blend_zone += b_bass[b_blend_start:b_blend_end] * fade_in_bass.reshape(-1, 1)
    blend_zone += b_drums[b_blend_start:b_blend_end] * fade_in_highs.reshape(-1, 1)
    blend_zone += b_other[b_blend_start:b_blend_end] * fade_in_highs.reshape(-1, 1)
    # We DO NOT fade in B's vocals yet to prevent messy clashing! 
    # B's vocals will start when the crossfade finishes.
    
    out_audio[a_transition_beat:a_transition_beat+actual_blend_len] += blend_zone
    
    # Rest of Track B plays out (including its vocals now!)
    drop_end = a_transition_beat + b_length
    remaining_b = drop_end - (a_transition_beat + actual_blend_len)
    
    b_rest_start = b_start_beat + actual_blend_len
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_drums[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_bass[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_other[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_vocals[b_rest_start:]
    
    # Export ONLY the transition slice for quick testing
    # 15 seconds before transition, and 15 seconds after the blend
    preview_start = max(0, a_transition_beat - (15 * sr))
    preview_end = min(len(out_audio), a_transition_beat + actual_blend_len + (15 * sr))
    
    preview_audio = out_audio[preview_start:preview_end]
    preview_audio = normalize(preview_audio)
    
    out_file = os.path.join(OUTPUT_DIR, "smooth_blend_mix.wav")
    print(f"Exporting final mix preview to {out_file}...")
    sf.write(out_file, preview_audio, sr)
    print("Done!")

def execute_mashup_blend(track_a_name, track_b_name, track_a_bpm, track_b_bpm):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
        
    print(f"Loading {track_a_name} stems...")
    a_vocals, sr = load_audio(crate[track_a_name]["stems"]["vocals"])
    a_drums, _ = load_audio(crate[track_a_name]["stems"]["drums"])
    a_bass, _ = load_audio(crate[track_a_name]["stems"]["bass"])
    a_other, _ = load_audio(crate[track_a_name]["stems"]["other"])
    
    print(f"Loading {track_b_name} stems...")
    b_vocals, _ = load_audio(crate[track_b_name]["stems"]["vocals"])
    b_drums, _ = load_audio(crate[track_b_name]["stems"]["drums"])
    b_bass, _ = load_audio(crate[track_b_name]["stems"]["bass"])
    b_other, _ = load_audio(crate[track_b_name]["stems"]["other"])
    
    # Time-stretch Track B
    if abs(track_a_bpm - track_b_bpm) > 0.1:
        stretch_factor = track_b_bpm / track_a_bpm
        b_drums = librosa.effects.time_stretch(b_drums.T, rate=stretch_factor).T
        b_bass = librosa.effects.time_stretch(b_bass.T, rate=stretch_factor).T
        b_vocals = librosa.effects.time_stretch(b_vocals.T, rate=stretch_factor).T
        b_other = librosa.effects.time_stretch(b_other.T, rate=stretch_factor).T
        
    print("Analyzing beat grids...")
    a_mono = np.mean(a_drums + a_bass + a_other, axis=1)
    _, a_beats = librosa.beat.beat_track(y=a_mono, sr=sr, bpm=track_a_bpm, units='samples')
    
    b_mono = np.mean(b_drums + b_bass + b_other, axis=1)
    _, b_beats = librosa.beat.beat_track(y=b_mono, sr=sr, bpm=track_a_bpm, units='samples')
    
    # HIGH ENERGY MASHUP: 
    # Mix Track A's main Drop (60 seconds) with Track B's main Drop (60 seconds)!
    a_target = 60 * sr
    a_transition_beat = a_beats[np.argmin(np.abs(a_beats - a_target))]
    
    b_target = 60 * sr
    b_start_beat = b_beats[np.argmin(np.abs(b_beats - b_target))]
    
    samples_per_beat = int((60.0 / track_a_bpm) * sr)
    blend_beats = 64 # 16 Bars of absolute high-energy mashup!
    blend_length = blend_beats * samples_per_beat
    
    b_length = len(b_drums) - b_start_beat
    output_length = a_transition_beat + b_length
    out_audio = np.zeros((output_length, 2), dtype=np.float32)
    
    print("Mixing tracks (HIGH ENERGY MASHUP)...")
    
    # Track A plays normally until the transition beat
    out_audio[:a_transition_beat] += a_drums[:a_transition_beat]
    out_audio[:a_transition_beat] += a_bass[:a_transition_beat]
    out_audio[:a_transition_beat] += a_other[:a_transition_beat]
    out_audio[:a_transition_beat] += a_vocals[:a_transition_beat]
    
    # Mashup Zone
    actual_blend_len = min(blend_length, len(b_drums) - b_start_beat, len(a_drums) - a_transition_beat)
    blend_zone = np.zeros((actual_blend_len, 2), dtype=np.float32)
    
    # To prevent muddy clashes in high energy:
    # 1. We keep Track A's massive Bass and Drums at 100%
    blend_zone += a_drums[a_transition_beat:a_transition_beat+actual_blend_len]
    blend_zone += a_bass[a_transition_beat:a_transition_beat+actual_blend_len]
    
    # 2. We fade out Track A's Melody to make room
    fade_out = np.linspace(1.0, 0.0, actual_blend_len).reshape(-1, 1)
    blend_zone += a_other[a_transition_beat:a_transition_beat+actual_blend_len] * fade_out
    
    # 3. We bring in Track B's Vocals and Melody at 100% over Track A's beat!
    b_blend_start = b_start_beat
    b_blend_end = b_start_beat + actual_blend_len
    blend_zone += b_vocals[b_blend_start:b_blend_end]
    blend_zone += b_other[b_blend_start:b_blend_end]
    
    # We completely mute Track B's bass/drums to prevent galloping clashes during the mashup!
    out_audio[a_transition_beat:a_transition_beat+actual_blend_len] += blend_zone
    
    # Rest of Track B plays out
    drop_end = a_transition_beat + b_length
    remaining_b = drop_end - (a_transition_beat + actual_blend_len)
    
    b_rest_start = b_start_beat + actual_blend_len
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_drums[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_bass[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_other[b_rest_start:]
    out_audio[a_transition_beat+actual_blend_len:drop_end] += b_vocals[b_rest_start:]
    
    # Export slice
    preview_start = max(0, a_transition_beat - (15 * sr))
    preview_end = min(len(out_audio), a_transition_beat + actual_blend_len + (15 * sr))
    
    preview_audio = out_audio[preview_start:preview_end]
    preview_audio = normalize(preview_audio)
    
    out_file = os.path.join(OUTPUT_DIR, "mashup_mix.wav")
    print(f"Exporting final mix preview to {out_file}...")
    sf.write(out_file, preview_audio, sr)
    print("Done!")

if __name__ == "__main__":
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    tracks = list(crate.keys())
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "blend":
        house_tracks = [t for t in tracks if "Sky High" in t or "Heroes Tonight" in t]
        if len(house_tracks) >= 2:
            track_a = house_tracks[0]
            track_b = house_tracks[1]
            execute_smooth_eq_blend(track_a, track_b, crate[track_a]["bpm"], crate[track_b]["bpm"])
    elif len(sys.argv) > 1 and sys.argv[1] == "mashup":
        house_tracks = [t for t in tracks if "Sky High" in t or "Heroes Tonight" in t]
        if len(house_tracks) >= 2:
            track_a = house_tracks[0]
            track_b = house_tracks[1]
            execute_mashup_blend(track_a, track_b, crate[track_a]["bpm"], crate[track_b]["bpm"])
    else:
        if len(tracks) >= 2:
            execute_echo_out_drop_swap(tracks[0], tracks[1], crate[tracks[0]]["bpm"], crate[tracks[1]]["bpm"])
