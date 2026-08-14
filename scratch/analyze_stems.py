import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_track_4():
    print("Stem analysis for Track 4...")
    # Load A's stems and B's stems
    # Dom Dolla is A. Knock2 is B.
    
    # We will load the output mix first to see the "one beat" issue.
    mix_path = "output/camelot/4_Expert_Energy_cut_Dom Dolla - San Frandisco (Extended Mix)_to_Knock2 - dashstar (Extended Mix).wav"
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    
    drop_sample = int(10.0 * sr)
    
    # Look at the first beat: 126 BPM = 2.1 beats per second = ~0.47 seconds per beat.
    # 1 beat is 0.47s. So from 10.0s to 10.5s.
    start = int(9.8 * sr)
    end = int(10.6 * sr)
    
    times = np.linspace(9.8, 10.6, end - start)
    y_zoom = y_mix[start:end]
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, y_zoom, alpha=0.8, color='blue', label='Mix Waveform')
    plt.axvline(x=10.0, color='red', linestyle='--', linewidth=2, label='Swap Point (10.0s)')
    
    rms = librosa.feature.rms(y=y_zoom, frame_length=512, hop_length=128)[0]
    rms_times = np.linspace(9.8, 10.6, len(rms))
    
    plt.plot(rms_times, rms, color='orange', linewidth=2, label='RMS Energy')
    
    plt.title("Track 4 (CUT) - First Beat Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scratch/4_first_beat.png")
    print("Saved 4_first_beat.png")

def analyze_track_2():
    print("Stem analysis for Track 2...")
    mix_path = "output/camelot/2_Medium_Neighbor_echo_out_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav"
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    
    # Look at the transition around 10.0s
    start = int(8.0 * sr)
    end = int(12.0 * sr)
    times = np.linspace(8.0, 12.0, end - start)
    
    plt.figure(figsize=(12, 6))
    plt.plot(times, y_mix[start:end], alpha=0.8, color='green', label='Mix Waveform')
    plt.axvline(x=10.0, color='red', linestyle='--', linewidth=2, label='Swap Point (10.0s)')
    
    rms = librosa.feature.rms(y=y_mix[start:end], frame_length=2048, hop_length=512)[0]
    rms_times = np.linspace(8.0, 12.0, len(rms))
    plt.plot(rms_times, rms, color='orange', linewidth=2, label='RMS Energy')
    
    plt.title("Track 2 (ECHO OUT) - Volume Ramp Analysis")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("scratch/2_volume_ramp.png")
    print("Saved 2_volume_ramp.png")

if __name__ == "__main__":
    analyze_track_4()
    analyze_track_2()
