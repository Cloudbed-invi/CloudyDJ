import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def analyze_audio(wav_path, out_png, pre_len_sec=10.0, blend_len_sec=7.62):
    print(f"Loading {wav_path}...")
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    
    # Calculate times
    total_duration = len(y) / sr
    times = np.linspace(0, total_duration, len(y))
    
    # 1. Waveform
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    axes[0].plot(times, y, color='b', alpha=0.7)
    axes[0].set_title(f"Waveform - {wav_path}")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)
    
    # 2. Spectrogram (Mel)
    print("Computing Mel Spectrogram...")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=16000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=16000, ax=axes[1])
    axes[1].set_title("Mel Spectrogram (Frequency Clash / Bass Map)")
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # 3. RMS Energy
    print("Computing RMS...")
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    frames = range(len(rms))
    t_rms = librosa.frames_to_time(frames, sr=sr, hop_length=512)
    
    axes[2].plot(t_rms, rms, color='r', label='RMS Energy')
    axes[2].set_title("RMS Energy Curve (Tension & Drops)")
    axes[2].set_ylabel("RMS")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True)
    
    # Add vertical lines for transition zones
    drop_time = pre_len_sec + blend_len_sec
    for ax in axes:
        # Build-up Start
        ax.axvline(x=pre_len_sec, color='yellow', linestyle='--', linewidth=2, label='Build-up Start (10s)')
        # Drop / Swap
        ax.axvline(x=drop_time, color='red', linestyle='-', linewidth=3, label='Drop / Swap')
        
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved visual analysis to {out_png}")

if __name__ == "__main__":
    wav_file = sys.argv[1]
    out_file = sys.argv[2]
    # We will assume pre_len is 10s and blend_len is roughly 16 beats at 126/128 BPM (~7.5s - 7.6s)
    analyze_audio(wav_file, out_file, pre_len_sec=10.0, blend_len_sec=7.62)
