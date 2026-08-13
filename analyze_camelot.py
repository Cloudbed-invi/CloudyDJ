import os
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "output/camelot"
ARTIFACTS_DIR = "C:/Users/sriha/.gemini/antigravity/brain/c59eb038-aea4-4134-804c-7714de9f666f/scratch"

files = [
    "1_Easy_SameKey_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav",
    "2_Medium_Neighbor_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav",
    "3_Hard_Relative_Chris Lake - In The Yuma (Extended Mix) [feat. Aatig]_to_John Summit - Deep End (Extended Mix).wav",
    "4_Expert_Energy_Dom Dolla - San Frandisco (Extended Mix)_to_Knock2 - dashstar (Extended Mix).wav"
]

def analyze_transition(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {filename}, not found.")
        return
        
    y, sr = sf.read(path, always_2d=True)
    mono = y.mean(axis=1)
    
    # 1. Look for silence exactly around the drop point (10.0 seconds)
    # 10s is 220500 samples if sr is 22050.
    drop_sample = int(10.0 * sr)
    window_start = drop_sample - int(2.0 * sr)
    window_end = drop_sample + int(2.0 * sr)
    
    zoom = mono[window_start:window_end]
    times = np.linspace(8.0, 12.0, len(zoom))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"Analysis: {filename[:20]}...", fontsize=14)
    
    # Waveform zoom around drop
    axes[0].plot(times, zoom, color='gray', alpha=0.8, linewidth=0.5)
    axes[0].axvline(10.0, color='red', linestyle='--', label='Swap Point (10.0s)')
    axes[0].set_title("Waveform (8s to 12s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    
    # Spectrogram to see beat alignment and frequency cutoffs
    D = np.abs(librosa.stft(zoom, n_fft=2048, hop_length=512))
    times_frames = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=512) + 8.0
    img = librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
                                   y_axis='log', x_axis='time', sr=sr, hop_length=512,
                                   ax=axes[1], x_coords=times_frames)
    axes[1].axvline(10.0, color='red', linestyle='--')
    axes[1].set_title("Spectrogram (8s to 12s)")
    
    out_img = os.path.join(ARTIFACTS_DIR, f"analyze_{filename[:7].strip('_')}.png")
    plt.tight_layout()
    plt.savefig(out_img)
    plt.close()
    
    # Check for silent gap (amplitude < 0.001) near 10s
    near_drop = mono[drop_sample - int(0.1*sr): drop_sample + int(0.1*sr)]
    rms_near = librosa.feature.rms(y=near_drop, frame_length=512, hop_length=128)[0]
    min_rms = np.min(rms_near)
    
    print(f"\n--- {filename[:20]}... ---")
    print(f"Minimum RMS within 100ms of drop (Silence threshold = 0.01): {min_rms:.5f}")
    if min_rms < 0.01:
        print("!! WARNING: Micro-silence gap detected at the transition boundary !!")

for f in files:
    analyze_transition(f)
