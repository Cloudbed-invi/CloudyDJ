import os
import json
import subprocess
import librosa
import soundfile as sf
import numpy as np
import sys

# Exact filenames downloaded
TRACKS = [
    "FISHER & Aatig - Take It Off (Extended Mix)",
    "MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)",
    "Chris Lake - In The Yuma (Extended Mix) [feat. Aatig]",
    "John Summit - Deep End (Extended Mix)",
    "Dom Dolla - San Frandisco (Extended Mix)",
    "Knock2 - dashstar (Extended Mix)"
]

LIBRARY_DIR = "library"
RAW_DIR = os.path.join(LIBRARY_DIR, "raw_audio")
DEMUCS_OUT = os.path.join(LIBRARY_DIR, "demucs_output")
CRATE_FILE = os.path.join(LIBRARY_DIR, "crate.json")

sys.path.insert(0, os.path.abspath("."))
from src.engine.generate_all_transitions import detect_first_drop
from analyzer import detect_key_and_bpm

def process():
    if os.path.exists(CRATE_FILE):
        with open(CRATE_FILE, 'r') as f:
            crate = json.load(f)
    else:
        crate = {}

    for title in TRACKS:
        if f"{title}.wav" in crate:
            print(f"Skipping {title} (already in crate)")
            continue
            
        audio_path = os.path.join(RAW_DIR, f"{title}.wav")
        print(f"--- Processing {title} ---")
        
        stems_dir = os.path.join(DEMUCS_OUT, "htdemucs", title)
        if not os.path.exists(stems_dir):
            print("Running Demucs...")
            subprocess.run([
                "conda", "run", "-n", "cloudy_dj", "demucs",
                "-n", "htdemucs",
                "-o", DEMUCS_OUT,
                audio_path
            ], check=True)
        else:
            print("Stems already exist, skipping Demucs.")
        
        print("Extracting metadata...")
        stems = {
            "vocals": os.path.join(stems_dir, "vocals.wav").replace("\\", "/"),
            "drums": os.path.join(stems_dir, "drums.wav").replace("\\", "/"),
            "bass": os.path.join(stems_dir, "bass.wav").replace("\\", "/"),
            "other": os.path.join(stems_dir, "other.wav").replace("\\", "/")
        }
        
        y, sr = librosa.load(audio_path, sr=22050, duration=120)
        key, bpm = detect_key_and_bpm(y, sr)
        
        y_drums, sr_d = sf.read(stems["drums"])
        if len(y_drums.shape) > 1: y_drums = y_drums.mean(axis=1)
        tempo, beats = librosa.beat.beat_track(y=y_drums, sr=sr_d, units='samples')
        
        y_bass, _ = sf.read(stems["bass"])
        if len(y_bass.shape) > 1: y_bass = y_bass.mean(axis=1)
        drop_idx, _ = detect_first_drop(y_bass, beats, sr_d, bpm)
        if drop_idx <= 0 and len(beats) > 16:
            drop_idx = 16
            
        crate[f"{title}.wav"] = {
            "path": audio_path.replace("\\", "/"),
            "key": key,
            "bpm": round(float(bpm), 2),
            "genre": "EDM",
            "stems": stems,
            "drop_idx": int(drop_idx)
        }
        
        with open(CRATE_FILE, 'w') as f:
            json.dump(crate, f, indent=4)
        print(f"Added {title} -> BPM: {bpm:.1f}, Key: {key}, Drop: {drop_idx}")

if __name__ == "__main__":
    process()
