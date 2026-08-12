import librosa
import numpy as np

path = "library/demucs_output/htdemucs/James_Hype_Wild/bass.wav"
y, sr = librosa.load(path, sr=44100)

beat_time = (125 / 126.0) * 60
start = int(beat_time * sr)
end = int((beat_time + 4) * sr) # Next 4 seconds

rms = librosa.feature.rms(y=y[start:end])[0]
print(f"James Hype Wild Bass RMS at beat 125 (drop): {np.mean(rms):.4f}")

path_tiesto = "library/demucs_output/htdemucs/Tiesto_Secrets/bass.wav"
y, sr = librosa.load(path_tiesto, sr=44100)
beat_time = (224 / 128.0) * 60
start = int(beat_time * sr)
end = int((beat_time + 4) * sr) # Next 4 seconds

rms = librosa.feature.rms(y=y[start:end])[0]
print(f"Tiesto Secrets Bass RMS at beat 224 (drop): {np.mean(rms):.4f}")
