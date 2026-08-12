import librosa
import numpy as np

for stem in ["bass"]:
    path = f"library/demucs_output/htdemucs/James_Hype_Wild/{stem}.wav"
    y, sr = librosa.load(path, sr=44100)
    
    print(f"\n--- James Hype Wild: {stem} ---")
    for beat in range(0, 300, 10):
        beat_time = (beat / 126.0) * 60
        start = int(beat_time * sr)
        end = int((beat_time + 4) * sr) # 4 sec
        if start < len(y):
            rms = librosa.feature.rms(y=y[start:end])[0]
            if np.mean(rms) > 0.05:
                print(f"Beat {beat}: {np.mean(rms):.4f}")
