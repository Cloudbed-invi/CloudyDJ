import os
import json
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

def analyze_track(track_name):
    print(f"Analyzing {track_name}...")
    stem_dir = f"library/demucs_output/htdemucs/{track_name}"
    
    # Load all stems
    stems = {}
    for name in ["bass", "drums", "other", "vocals"]:
        path = os.path.join(stem_dir, f"{name}.wav")
        if not os.path.exists(path):
            print(f"Missing {name} stem for {track_name}")
            return
        audio, sr = librosa.load(path, sr=44100, mono=True)
        stems[name] = audio

    # Compute RMS energy over time (e.g., 0.5s windows)
    hop_length = 512
    frame_time = librosa.frames_to_time(1, sr=sr, hop_length=hop_length)
    
    energies = {}
    for name, audio in stems.items():
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        # Smooth the RMS a bit
        rms_smoothed = np.convolve(rms, np.ones(10)/10, mode='same')
        energies[name] = rms_smoothed

    times = librosa.times_like(energies["bass"], sr=sr, hop_length=hop_length)

    plt.figure(figsize=(20, 10))
    for i, name in enumerate(["bass", "drums", "other", "vocals"]):
        plt.subplot(4, 1, i+1)
        plt.plot(times, energies[name], label=f"{name.capitalize()} Energy", color=f"C{i}")
        plt.title(f"{track_name} - {name.capitalize()}")
        plt.ylabel("RMS")
        plt.grid(True)
        # Mark beat 125 for James Hype and 224 for Tiesto
        if track_name == "James_Hype_Wild":
            drop_time = (125 / 126.0) * 60
            plt.axvline(drop_time, color='r', linestyle='--', label="Drop (Beat 125)")
        elif track_name == "Tiesto_Secrets":
            drop_time = (224 / 128.0) * 60
            plt.axvline(drop_time, color='r', linestyle='--', label="Drop (Beat 224)")
        plt.legend()
        plt.xlim(0, max(times))

    plt.xlabel("Time (s)")
    plt.tight_layout()
    out_path = f"C:/Users/sriha/.gemini/antigravity/brain/c59eb038-aea4-4134-804c-7714de9f666f/scratch/{track_name}_full_analysis.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

analyze_track("James_Hype_Wild")
analyze_track("Tiesto_Secrets")
