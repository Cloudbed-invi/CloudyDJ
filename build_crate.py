import os
import json
import librosa
import numpy as np
import sys
sys.path.insert(0, os.path.abspath("."))
from src.engine.structure_analyzer import analyze_track_structure
from analyzer import detect_key_and_bpm

crate_file = "library/crate.json"
base_dir = os.path.abspath(".")

tracks = [
    ("Tiesto_Secrets", "library/Tiesto_Secrets.wav", "library/demucs_output/htdemucs/Tiesto_Secrets"),
    ("James_Hype_Wild", "library/James_Hype_Wild.wav", "library/demucs_output/htdemucs/James_Hype_Wild"),
    ("Elvis_JailhouseRock", "library/Elvis_JailhouseRock.wav", "library/demucs_output/htdemucs/Elvis_JailhouseRock"),
    ("Enya_OrinocoFlow", "library/Enya_OrinocoFlow.wav", "library/demucs_output/htdemucs/Enya_OrinocoFlow")
]

crate = {}
for name, audio_path, stems_dir in tracks:
    print(f"Processing {name}...")
    y, sr = librosa.load(audio_path, sr=22050, duration=120)
    key, bpm = detect_key_and_bpm(y, sr)
    
    # We will run structure_analyzer which takes the full audio path (but we can pass stems_dir/other.wav or audio_path)
    # The structure analyzer script we saw earlier had: y, sr = librosa.load(audio_path, sr=sr, mono=True)
    segments = analyze_track_structure(audio_path, sr=22050)
    
    crate[name + ".wav"] = {
        "path": audio_path,
        "key": key,
        "bpm": bpm,
        "genre": "EDM",
        "stems": {
            "vocals": os.path.join(stems_dir, "vocals.wav").replace("\\", "/"),
            "drums": os.path.join(stems_dir, "drums.wav").replace("\\", "/"),
            "bass": os.path.join(stems_dir, "bass.wav").replace("\\", "/"),
            "other": os.path.join(stems_dir, "other.wav").replace("\\", "/")
        },
        "segments": segments
    }

with open(crate_file, "w") as f:
    json.dump(crate, f, indent=4)
print("crate.json built successfully!")
