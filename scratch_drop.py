import json
import librosa
import numpy as np

def _to_mono(audio):
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)

import soundfile as sf
def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    if y.ndim == 1: y = np.stack([y, y], axis=1)
    elif y.shape[1] > 2: y = y[:, :2]
    return y, sr

from scipy.ndimage import uniform_filter1d

def detect_first_drop(bass_mono, beats, sr, bpm):
    hop = 1024
    rms = librosa.feature.rms(y=bass_mono, frame_length=2048, hop_length=hop)[0]
    
    smooth_len = int((60.0 / bpm) * sr / hop)
    if smooth_len > 0:
        rms = uniform_filter1d(rms, size=smooth_len)

    mean_bass = np.mean(rms) + 1e-9
    frames_per_sec = sr / hop
    start_frame = int(10 * frames_per_sec)  # Changed to 10 seconds!
    end_frame   = int(120 * frames_per_sec)

    best_frame, max_contrast = 0, 0
    window = smooth_len * 32
    
    contrasts = []
    for frame_idx in range(max(window, start_frame), min(end_frame, len(rms) - window)):
        pre_mean  = np.mean(rms[max(0, frame_idx - window):frame_idx])
        post_mean = np.mean(rms[frame_idx:frame_idx + window])
        contrast  = post_mean - pre_mean
        
        # Only consider it a drop if post_mean is loud enough
        if contrast > max_contrast and post_mean > 0.02:
            max_contrast = contrast
            best_frame   = frame_idx
            contrasts.append((frame_idx / frames_per_sec, contrast, pre_mean, post_mean))

    print(f"Top 5 candidates by contrast:")
    for c in sorted(contrasts, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {c[0]:.2f}s: Contrast {c[1]:.4f} (Pre: {c[2]:.4f}, Post: {c[3]:.4f})")

    if best_frame == 0:
        return 0, False

    drop_sample = librosa.frames_to_samples(best_frame, hop_length=hop)
    return np.argmin(np.abs(beats - drop_sample)), True

with open("library/crate.json", "r") as f:
    crate = json.load(f)

for k, data in crate.items():
    if "James_Hype_Wild" in k:
        print("Analyzing James Hype Wild...")
        bpm = data["bpm"]
        bass, sr = load_audio(data["stems"]["bass"])
        mono = _to_mono(bass)
        _, beats = librosa.beat.beat_track(y=mono, sr=sr, bpm=bpm, units='samples')
        drop_idx, _ = detect_first_drop(mono, beats, sr, bpm)
        
        print(f"BPM: {bpm}")
        print(f"Detected Drop Beat: {drop_idx}")
        if drop_idx < len(beats):
            drop_s = beats[drop_idx] / sr
            print(f"Drop Time: {drop_s:.2f}s")
        break
