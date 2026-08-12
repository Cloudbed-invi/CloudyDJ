import os
import json
import librosa
import soundfile as sf
import numpy as np

# Adjust sys.path to be able to import from src.engine
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.engine.generate_all_transitions import detect_first_drop

LIBRARY_DIR = "C:/Users/sriha/Documents/Cloudy DJ 2.0/library"
CRATE_FILE = os.path.join(LIBRARY_DIR, "crate.json")

def rebuild_crate():
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    print("Rebuilding crate.json... Detecting drops for all tracks.")
    print("-" * 60)
    print(f"{'Track Name':<30} | {'Drop Beat':<10} | {'Time (s)':<10}")
    print("-" * 60)

    for title, data in crate.items():
        stems = data.get("stems", {})
        bass_path = stems.get("bass")
        
        if not bass_path or not os.path.exists(bass_path):
            print(f"Skipping {title} - no bass stem found.")
            continue
            
        try:
            # Load bass stem
            y_bass, sr = librosa.load(bass_path, sr=44100, mono=True)
            
            # Recompute beats
            bpm = data.get("bpm", 128.0)
            drums_path = stems.get("drums")
            y_drums, _ = librosa.load(drums_path, sr=44100, mono=True) if drums_path and os.path.exists(drums_path) else (None, sr)
            
            if y_drums is not None:
                tempo, beats = librosa.beat.beat_track(y=y_drums, sr=sr, units='samples')
            else:
                tempo, beats = librosa.beat.beat_track(y=y_bass, sr=sr, units='samples')

            # Detect drop
            drop_beat_idx, drop_confident = detect_first_drop(y_bass, beats, sr, bpm)
            
            # If the drop is somehow index 0, make it a bit safer
            if drop_beat_idx <= 0 and len(beats) > 16:
                drop_beat_idx = 16
                
            drop_sample = beats[drop_beat_idx]
            drop_time = drop_sample / sr
            
            print(f"{title:<30} | {drop_beat_idx:<10} | {drop_time:.2f}")
            
            # Update crate (remove manual override, set auto-detected)
            data["drop_idx"] = int(drop_beat_idx)
            
        except Exception as e:
            print(f"Error processing {title}: {e}")

    # Write back to crate.json
    with open(CRATE_FILE, 'w') as f:
        json.dump(crate, f, indent=4)
        
    print("-" * 60)
    print("Done! Updated crate.json with automatic drop_idx values.")

if __name__ == "__main__":
    rebuild_crate()
