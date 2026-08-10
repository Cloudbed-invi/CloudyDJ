import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa
from scipy import signal

sys.path.append("C:/Projects/Cloudy DJ 2.0")
from generate_batch import get_track_data

CRATE_FILE = "C:/Projects/Cloudy DJ 2.0/library/crate.json"
MIX_AUDIO = "C:/Projects/Cloudy DJ 2.0/output/mix_audio.wav"
TECH_DIR = "C:/Projects/Cloudy DJ 2.0/techniques"
os.makedirs(TECH_DIR, exist_ok=True)

def load_track(title):
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    return get_track_data(crate, title)

def find_offset(template, target, sr):
    """ Finds the sample offset and secondary correlation peaks to detect loops. """
    # Downsample for speed
    target_ds = librosa.resample(target, orig_sr=sr, target_sr=22050)
    template_ds = librosa.resample(template, orig_sr=sr, target_sr=22050)
    
    correlation = signal.correlate(target_ds, template_ds, mode='valid', method='fft')
    offset_ds = np.argmax(correlation)
    offset_samples = int(offset_ds * (sr / 22050))
    
    # Loop Detection: Look for periodic secondary peaks
    # If the track was looped, the correlation will have peaks every N beats
    peaks, properties = signal.find_peaks(correlation, height=np.max(correlation)*0.6, distance=22050)
    is_looped = len(peaks) > 3 # If there are many high peaks, it's highly repetitive or looped
    
    return offset_samples, is_looped

def multi_band_filter(audio, sr):
    """ Splits audio into Low, Mid, High bands using butterworth filters """
    nyq = 0.5 * sr
    
    # Low pass (0-250 Hz)
    b, a = signal.butter(4, 250 / nyq, btype='low')
    lows = signal.filtfilt(b, a, audio)
    
    # Band pass (250-4000 Hz)
    b, a = signal.butter(4, [250 / nyq, 4000 / nyq], btype='band')
    mids = signal.filtfilt(b, a, audio)
    
    # High pass (4000+ Hz)
    b, a = signal.butter(4, 4000 / nyq, btype='high')
    highs = signal.filtfilt(b, a, audio)
    
    return lows, mids, highs

def extract_multi_band_envelope(mix_window, a_stem, b_stem, sr):
    mix_l, mix_m, mix_h = multi_band_filter(mix_window, sr)
    a_l, a_m, a_h = multi_band_filter(a_stem, sr)
    b_l, b_m, b_h = multi_band_filter(b_stem, sr)
    
    # Calculate RMS for each band
    def get_rms(audio):
        rms = librosa.feature.rms(y=audio, frame_length=4096, hop_length=1024)[0]
        # Savitzky-Golay filter to smooth it
        return signal.savgol_filter(rms, window_length=11, polyorder=2)
        
    mix_rms_l = get_rms(mix_l)
    mix_rms_m = get_rms(mix_m)
    mix_rms_h = get_rms(mix_h)
    
    a_rms_l = get_rms(a_l)
    a_rms_m = get_rms(a_m)
    a_rms_h = get_rms(a_h)
    
    b_rms_l = get_rms(b_l)
    b_rms_m = get_rms(b_m)
    b_rms_h = get_rms(b_h)
    
    # Extract envelopes
    eps = 1e-6
    env_l = (mix_rms_l / (a_rms_l + b_rms_l + eps)).tolist()[::10]
    env_m = (mix_rms_m / (a_rms_m + b_rms_m + eps)).tolist()[::10]
    env_h = (mix_rms_h / (a_rms_h + b_rms_h + eps)).tolist()[::10]
    
    return env_l, env_m, env_h

def analyze_transition(name_a, name_b, index):
    print(f"\n--- Extracting Transition {index}: {name_a} -> {name_b} ---")
    a = load_track(name_a)
    b = load_track(name_b)
    
    global mix_y, mix_sr
    
    a_chunk = a["drums"][30*mix_sr:60*mix_sr]
    b_chunk = b["drums"][30*mix_sr:60*mix_sr]
    
    print("Correlating Track A...")
    a_offset, a_looped = find_offset(a_chunk, mix_y, mix_sr)
    a_start_in_mix = a_offset - (30 * mix_sr)
    
    print("Correlating Track B...")
    b_offset, b_looped = find_offset(b_chunk, mix_y, mix_sr)
    b_start_in_mix = b_offset - (30 * mix_sr)
    
    diff_samples = b_start_in_mix - a_start_in_mix
    print(f"Track A starts at {a_start_in_mix/mix_sr:.2f}s, Track B starts at {b_start_in_mix/mix_sr:.2f}s")
    print(f"Track B drops in {diff_samples/mix_sr:.2f}s after Track A")
    
    if a_looped: print("DETECTED: DJ Looped Track A during transition!")
    if b_looped: print("DETECTED: DJ Looped Track B during transition!")
    
    # Window analysis (30s before and 30s after B drops in)
    # The DJ usually mixes starting 30s before the new track drops, and finishes 30s after.
    # We will analyze a massive 60-second window!
    window_start = b_start_in_mix - (30 * mix_sr)
    window_end = b_start_in_mix + (30 * mix_sr)
    
    # If the mix file isn't long enough, clip it
    if window_start < 0: window_start = 0
    if window_end > len(mix_y): window_end = len(mix_y)
    
    mix_window = mix_y[window_start:window_end]
    
    # Mix together the pure instrumental stems (bass + drums + other)
    a_inst = a["bass"] + a["drums"] + a["other"]
    b_inst = b["bass"] + b["drums"] + b["other"]
    
    a_start_idx = window_start - a_start_in_mix
    b_start_idx = window_start - b_start_in_mix
    
    # Safe slicing
    a_inst_window = np.zeros(len(mix_window))
    b_inst_window = np.zeros(len(mix_window))
    
    if a_start_idx >= 0 and a_start_idx + len(mix_window) <= len(a_inst):
        a_inst_window = a_inst[a_start_idx : a_start_idx + len(mix_window)]
    if b_start_idx >= 0 and b_start_idx + len(mix_window) <= len(b_inst):
        b_inst_window = b_inst[b_start_idx : b_start_idx + len(mix_window)]
        
    print("Extracting Multi-Band EQ Envelopes...")
    env_l, env_m, env_h = extract_multi_band_envelope(mix_window, a_inst_window, b_inst_window, mix_sr)
    
    technique = {
        "source_mix": "YouTube DJ Mix",
        "track_out": name_a,
        "track_in": name_b,
        "duration": 60.0,
        "loop_out": a_looped,
        "loop_in": b_looped,
        "eq_automation": {
            "lows": env_l,
            "mids": env_m,
            "highs": env_h
        }
    }
    
    out_file = os.path.join(TECH_DIR, f"extracted_transition_{index}.json")
    with open(out_file, "w") as f:
        json.dump(technique, f, indent=4)
        
    print(f"Saved highly-advanced multi-band technique to {out_file}!")

if __name__ == "__main__":
    print("Loading Mix Audio (this takes a moment)...")
    global mix_y, mix_sr
    mix_y, mix_sr = sf.read(MIX_AUDIO, dtype='float32')
    if len(mix_y.shape) > 1: mix_y = mix_y.mean(axis=1)
    
    transitions = [
        ("ovc - Sprinter (KARMA Remix)", "James Hype - Wild"),
        ("James Hype - Wild", "Kasa Remixoff & Jony Like - Moon"),
        ("Kasa Remixoff & Jony Like - Moon", "Misericordie - Voodoo People (Rave Mix)")
    ]
    
    for idx, (t_a, t_b) in enumerate(transitions, 1):
        try:
            analyze_transition(t_a, t_b, idx)
        except Exception as e:
            print(f"Skipping Transition {idx} due to error: {e}")
