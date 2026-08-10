import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from loop_roll import generate_loop_roll

sys.path.append("C:/Projects/Cloudy DJ 2.0")
from generate_batch import get_track_data

CRATE_FILE = "C:/Projects/Cloudy DJ 2.0/library/crate.json"
TECH_DIR = "C:/Projects/Cloudy DJ 2.0/techniques"
OUTPUT_FILE = "C:/Projects/Cloudy DJ 2.0/output/autonomous_mix.wav"

def load_track(title):
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    return get_track_data(crate, title), crate[title]["bpm"]

def load_technique(filename):
    with open(os.path.join(TECH_DIR, filename), 'r') as f:
        return json.load(f)

def multi_band_filter(audio, sr):
    nyq = 0.5 * sr
    b, a = signal.butter(4, 250 / nyq, btype='low')
    lows = signal.filtfilt(b, a, audio)
    b, a = signal.butter(4, [250 / nyq, 4000 / nyq], btype='band')
    mids = signal.filtfilt(b, a, audio)
    b, a = signal.butter(4, 4000 / nyq, btype='high')
    highs = signal.filtfilt(b, a, audio)
    return lows, mids, highs

def apply_eq_envelope(audio_stem, envelope, target_length):
    env_arr = np.array(envelope)
    x_old = np.linspace(0, 1, len(env_arr))
    x_new = np.linspace(0, 1, target_length)
    env_resampled = np.interp(x_new, x_old, env_arr)
    return audio_stem * env_resampled

def match_energy(buildup, target_audio, sr):
    buildup_peak_rms = np.max(librosa.feature.rms(y=buildup[-sr:], frame_length=2048, hop_length=512)[0])
    target_drop_rms = np.max(librosa.feature.rms(y=target_audio[:sr], frame_length=2048, hop_length=512)[0])
    if buildup_peak_rms == 0: return buildup
    gain_multiplier = target_drop_rms / buildup_peak_rms
    ramp = np.linspace(1.0, gain_multiplier, len(buildup))
    return buildup * ramp

def apply_autonomous_mix():
    print("Loading Tracks...")
    a_data, a_bpm = load_track("Tiesto - Secrets")
    b_data, b_bpm = load_track("James Hype - Wild")
    
    sr = 44100
    print("Loading Technique 2 (Wild -> Moon EQ Fade)...")
    tech = load_technique("extracted_transition_2.json")
    
    print(f"Time stretching Track A from {a_bpm} to {b_bpm} BPM...")
    rate = b_bpm / a_bpm
    a_inst = a_data["bass"] + a_data["drums"] + a_data["other"]
    a_inst_warped = librosa.effects.time_stretch(a_inst, rate=rate)
    a_vocals_warped = librosa.effects.time_stretch(a_data["vocals"], rate=rate)
    
    b_inst = b_data["bass"] + b_data["drums"] + b_data["other"]
    b_vocals = b_data["vocals"]
    
    a_full = a_inst_warped + a_vocals_warped
    b_full = b_inst + b_vocals
    
    # Find drops
    # Warping changes the number of samples, so we calculate drop_sample based on the original beats but warped
    a_drop_sample_orig = a_data["beats"][a_data["drop_idx"]]
    a_drop_sample = int(a_drop_sample_orig / rate)
    b_drop_sample = b_data["beats"][b_data["drop_idx"]]
    
    print(f"Track A Drop: {a_drop_sample/sr:.2f}s | Track B Drop: {b_drop_sample/sr:.2f}s")
    
    # We will transition so that Track B's drop happens EXACTLY 32 beats after Track A's drop.
    samples_per_beat = int((60.0 / b_bpm) * sr)
    transition_point_a = a_drop_sample + (32 * samples_per_beat)
    
    if transition_point_a > len(a_inst_warped):
        transition_point_a = a_drop_sample
    
    # Create the output master audio timeline
    total_length = transition_point_a + (len(b_full) - b_drop_sample) + int(10*sr)
    out_audio = np.zeros(total_length)
    
    # GENERATE LOOP ROLL ON TRACK A VOCALS FIRST (so we get exact length)
    print("Generating Loop Roll Tension Builder...")
    buildup = generate_loop_roll(a_vocals_warped, b_bpm, sr, transition_point_a)
    
    # Scale energy perfectly to match Track B's drop!
    b_drop_audio = b_full[b_drop_sample : b_drop_sample + int(5*sr)]
    buildup_scaled = match_energy(buildup, b_drop_audio, sr)
    
    # Calculate EXACT Buildup Start Time based on generated loop roll
    buildup_start = transition_point_a - len(buildup_scaled)
    
    # 1. Place Track A up to the buildup start
    out_audio[:buildup_start] += a_inst_warped[:buildup_start]
    out_audio[:buildup_start] += a_vocals_warped[:buildup_start]
    
    # 2. Track A Buildup (DJ turns down the BASS during the loop roll!)
    # We take Track A's instrumental during the 16 beats, split it, and fade the bass to 0!
    a_inst_buildup = a_inst_warped[buildup_start:transition_point_a]
    a_l, a_m, a_h = multi_band_filter(a_inst_buildup, sr)
    
    # Fade bass from 1.0 down to 0.0
    bass_fade = np.linspace(1.0, 0.0, len(a_l))
    # Fade mids slightly to make room for loop roll
    mid_fade = np.linspace(1.0, 0.5, len(a_m))
    
    a_inst_buildup_eq = (a_l * bass_fade) + (a_m * mid_fade) + a_h
    out_audio[buildup_start:transition_point_a] += a_inst_buildup_eq
    
    # 3. Add the Loop Roll (Vocals)
    out_audio[buildup_start : transition_point_a] += buildup_scaled
    
    # 4. EXACTLY AT THE DROP -> TRACK B!
    # Track B's intro is COMPLETELY skipped. It drops with maximum impact!
    b_length = len(b_full) - b_drop_sample
    out_audio[transition_point_a : transition_point_a + b_length] += b_full[b_drop_sample:]
    
    # Washout Track A's Synths (Top-End) so it doesn't just stop instantly
    a_top_end = a_data["drums"] + a_data["other"]
    a_top_end_warped = librosa.effects.time_stretch(a_top_end, rate=rate)
    wash_len = int(10 * sr)
    a_wash = a_top_end_warped[transition_point_a:min(transition_point_a+wash_len, len(a_top_end_warped))]
    if len(a_wash) > 0:
        wash_fade = np.linspace(0.5, 0.0, len(a_wash))
        out_audio[transition_point_a:transition_point_a+len(a_wash)] += (a_wash * wash_fade)
    
    # Save a 2-minute snippet around the transition so it's easy to listen to
    short_start = max(0, transition_point_a - (30 * sr))
    short_end = min(len(out_audio), transition_point_a + (60 * sr))
    
    short_audio = out_audio[short_start:short_end]
    
    # CRITICAL: NORMALIZE AUDIO TO PREVENT DIGITAL CLIPPING/DISTORTION!
    peak_amplitude = np.max(np.abs(short_audio))
    if peak_amplitude > 1.0:
        print(f"Audio was clipping heavily (Peak: {peak_amplitude:.2f})! Normalizing...")
        short_audio = short_audio / peak_amplitude
    
    print(f"Saving autonomous master mix to {OUTPUT_FILE}...")
    sf.write(OUTPUT_FILE, short_audio, sr)
    print("Done! Autonomous transition successful.")

if __name__ == "__main__":
    apply_autonomous_mix()
