import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_track_4():
    print("Analyzing Track 4 Cut Timing...")
    mix_path = "output/camelot/4_Expert_Energy_cut_Dom Dolla - San Frandisco (Extended Mix)_to_Knock2 - dashstar (Extended Mix).wav"
    
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    
    drop_sec = 10.0
    drop_sample = int(drop_sec * sr)
    
    # 0.5 seconds before and 0.5 seconds after
    start_sec = 9.5
    end_sec = 10.5
    
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    
    times = np.linspace(start_sec, end_sec, end_sample - start_sample)
    y_zoom = y_mix[start_sample:end_sample]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, y_zoom, alpha=0.8, color='blue', label='Mix Waveform')
    plt.axvline(x=drop_sec, color='red', linestyle='--', linewidth=2, label='Swap Point (10.0s)')
    
    # Let's also compute RMS in tiny windows to see the energy
    rms = librosa.feature.rms(y=y_zoom, frame_length=512, hop_length=128)[0]
    rms_times = np.linspace(start_sec, end_sec, len(rms))
    
    plt.plot(rms_times, rms, color='orange', linewidth=2, label='RMS Energy')
    
    plt.title("Track 4 (CUT) - Timing Analysis around 10s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    out_path = "C:/Users/sriha/.gemini/antigravity/brain/c59eb038-aea4-4134-804c-7714de9f666f/scratch/4_cut_timing.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def analyze_track_1():
    print("Analyzing Track 1 Cut Timing...")
    mix_path = "output/camelot/1_Easy_SameKey_cut_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav"
    
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    drop_sec = 10.0
    start_sample = int(9.5 * sr)
    end_sample = int(10.5 * sr)
    times = np.linspace(9.5, 10.5, end_sample - start_sample)
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, y_mix[start_sample:end_sample], alpha=0.8, color='green', label='Mix Waveform')
    plt.axvline(x=drop_sec, color='red', linestyle='--', linewidth=2, label='Swap Point (10.0s)')
    
    rms = librosa.feature.rms(y=y_mix[start_sample:end_sample], frame_length=512, hop_length=128)[0]
    rms_times = np.linspace(9.5, 10.5, len(rms))
    plt.plot(rms_times, rms, color='orange', linewidth=2, label='RMS Energy')
    
    plt.title("Track 1 (CUT) - Timing Analysis around 10s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    out_path = "C:/Users/sriha/.gemini/antigravity/brain/c59eb038-aea4-4134-804c-7714de9f666f/scratch/1_cut_timing.png"
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    analyze_track_4()
    analyze_track_1()
