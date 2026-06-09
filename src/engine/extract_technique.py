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

def load_track(title):
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    return get_track_data(crate, title)

def find_offset(template, target, sr):
    """ Finds the sample offset of template within target using cross-correlation. """
    # Downsample for speed
    target_ds = librosa.resample(target, orig_sr=sr, target_sr=22050)
    template_ds = librosa.resample(template, orig_sr=sr, target_sr=22050)
    
    correlation = signal.correlate(target_ds, template_ds, mode='valid', method='fft')
    offset_ds = np.argmax(correlation)
    offset_samples = int(offset_ds * (sr / 22050))
    return offset_samples

def extract_technique():
    print("Loading Mix Audio...")
    mix_y, sr = sf.read(MIX_AUDIO, dtype='float32')
    if len(mix_y.shape) > 1: mix_y = mix_y.mean(axis=1)
    
    print("Loading Track A (Sprinter) and Track B (Wild)...")
    a = load_track("ovc - Sprinter (KARMA Remix)")
    b = load_track("James Hype - Wild")
    
    # Let's find exactly where Track A and Track B exist in the mix!
    # We use a 30 second chunk from the middle of their stems to avoid intro ambiguitiy
    a_chunk = a["drums"][30*sr:60*sr]
    b_chunk = b["drums"][30*sr:60*sr]
    
    print("Correlating Track A to find timestamp...")
    a_offset = find_offset(a_chunk, mix_y, sr)
    a_start_in_mix = a_offset - (30 * sr)
    print(f"Track A starts at {a_start_in_mix/sr:.2f}s in the mix")
    
    print("Correlating Track B to find timestamp...")
    b_offset = find_offset(b_chunk, mix_y, sr)
    b_start_in_mix = b_offset - (30 * sr)
    print(f"Track B starts at {b_start_in_mix/sr:.2f}s in the mix")
    
    # Calculate Phase Alignment
    # How many Track B beats pass between Track A's start and Track B's start?
    diff_samples = b_start_in_mix - a_start_in_mix
    print(f"Transition Offset: {diff_samples/sr:.2f} seconds")
    
    # EQ Extraction (Bass Washout Curve)
    # We analyze the window from (b_start - 30s) to (b_start + 30s)
    window_start = b_start_in_mix - (15 * sr)
    window_end = b_start_in_mix + (15 * sr)
    
    mix_window = mix_y[window_start:window_end]
    a_bass_window = a["bass"][window_start - a_start_in_mix : window_end - a_start_in_mix]
    b_bass_window = b["bass"][window_start - b_start_in_mix : window_end - b_start_in_mix]
    
    # Calculate RMS energy of mix vs original stems
    mix_rms = librosa.feature.rms(y=mix_window, frame_length=4096, hop_length=1024)[0]
    a_bass_rms = librosa.feature.rms(y=a_bass_window, frame_length=4096, hop_length=1024)[0]
    b_bass_rms = librosa.feature.rms(y=b_bass_window, frame_length=4096, hop_length=1024)[0]
    
    # If the derivative of energy is huge, it's a CUT
    mix_diff = np.diff(mix_rms)
    max_jump = np.max(np.abs(mix_diff))
    
    if max_jump > 0.5:
        print("DETECTED A HARD CUT! Ignoring transition technique...")
        return
        
    print("DETECTED A BLEND TRANSITION! Mathematical extraction successful.")
    # Smooth the curve for the AI to use
    envelope = mix_rms / (a_bass_rms + b_bass_rms + 1e-6)
    smoothed = signal.savgol_filter(envelope, window_length=11, polyorder=2)
    
    # Save the technique
    tech_dir = "C:/Projects/Cloudy DJ 2.0/techniques"
    os.makedirs(tech_dir, exist_ok=True)
    
    technique = {
        "source_mix": "YouTube DJ Mix",
        "track_out": "James Hype - Wild",
        "track_in": "Sprinter (KARMA Remix)",
        "duration": 30.0,
        "curve_type": "bass_washout",
        "envelope_data": smoothed.tolist()[::10] # downsample for storage
    }
    
    out_file = os.path.join(tech_dir, "extracted_technique_1.json")
    with open(out_file, "w") as f:
        json.dump(technique, f, indent=4)
        
    print(f"Extracted a {len(mix_rms)*1024/sr:.1f} second transition envelope.")
    print(f"Saved technique to {out_file}!")

if __name__ == "__main__":
    extract_technique()
