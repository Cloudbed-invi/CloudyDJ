import os
import json
import glob
import subprocess
import librosa
import numpy as np

RAW_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "library", "raw_audio")
STEMS_DIR = os.path.join(os.path.dirname(__file__), "library", "stems")
CRATE_FILE = os.path.join(os.path.dirname(__file__), "library", "crate.json")

# Major and Minor Profiles for Key Detection
maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def detect_key_and_bpm(y, sr):
    print("Detecting key and BPM with librosa...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    
    maj_corrs = [np.corrcoef(chroma_vals, np.roll(maj_profile, i))[0, 1] for i in range(12)]
    min_corrs = [np.corrcoef(chroma_vals, np.roll(min_profile, i))[0, 1] for i in range(12)]
    
    max_maj = max(maj_corrs)
    max_min = max(min_corrs)
    
    if max_maj > max_min:
        key_idx = maj_corrs.index(max_maj)
        mode = "Major"
    else:
        key_idx = min_corrs.index(max_min)
        mode = "Minor"
        
    key_str = f"{PITCH_CLASSES[key_idx]} {mode}"
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    return key_str, round(bpm_val, 2)

import torchaudio
import soundfile as sf
import torch
import demucs.separate

def custom_torchaudio_load(filepath):
    data, samplerate = sf.read(str(filepath), dtype='float32')
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    data = data.T
    return torch.from_numpy(data), samplerate

def custom_torchaudio_save(filepath, tensor, sample_rate, **kwargs):
    data = tensor.T.numpy()
    sf.write(str(filepath), data, sample_rate)

torchaudio.load = custom_torchaudio_load
torchaudio.save = custom_torchaudio_save

def run_demucs(file_path):
    safe_path = file_path.encode('ascii', 'ignore').decode('ascii')
    print(f"Running Demucs stem separation on {safe_path}...")
    demucs.separate.main([
        "-n", "htdemucs",
        "-o", STEMS_DIR,
        file_path
    ])
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    stems_folder = os.path.join(STEMS_DIR, "htdemucs", base_name)
    return {
        "vocals": os.path.join(stems_folder, "vocals.wav"),
        "drums": os.path.join(stems_folder, "drums.wav"),
        "bass": os.path.join(stems_folder, "bass.wav"),
        "other": os.path.join(stems_folder, "other.wav")
    }

def analyze_all():
    crate = {}
    if os.path.exists(CRATE_FILE):
        with open(CRATE_FILE, 'r') as f:
            crate = json.load(f)
            
    wav_files = glob.glob(os.path.join(RAW_AUDIO_DIR, "*.wav"))
    
    # Process only the first 2 tracks to save CPU time during development
    processed_count = 0
    
    for wav_file in wav_files:
        track_name = os.path.basename(wav_file)
        safe_name = track_name.encode('ascii', 'ignore').decode('ascii')
        if track_name in crate:
            print(f"Skipping {safe_name}, already in crate.")
            continue
            
        print(f"\n--- Analyzing {safe_name} ---")
        
        y, sr = librosa.load(wav_file, sr=22050, duration=120) 
        key, bpm = detect_key_and_bpm(y, sr)
        
        print(f"Detected: {bpm} BPM | {key}")
        
        stem_paths = run_demucs(wav_file)
        
        # Try to extract genre from NCS standard title format: "Artist - Title [Genre] NCS" or "Artist - Title Genre NCS"
        base_name = os.path.splitext(track_name)[0]
        genre = "Unknown EDM"
        if "NCS" in base_name:
            parts = base_name.split("  ")
            if len(parts) >= 3:
                genre = parts[1].strip()
                
        crate[track_name] = {
            "path": wav_file,
            "key": key,
            "bpm": bpm,
            "genre": genre,
            "stems": stem_paths
        }
        
        with open(CRATE_FILE, 'w') as f:
            json.dump(crate, f, indent=4)
            
        processed_count += 1
        if processed_count >= 2:
            print("\nProcessed 2 tracks. Stopping early for demo.")
            break
            
    print("\nAnalysis Complete! Crate saved.")

if __name__ == "__main__":
    analyze_all()
