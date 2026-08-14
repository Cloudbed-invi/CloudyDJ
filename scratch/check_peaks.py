import librosa
import numpy as np

def analyze_peaks():
    tracks = {
        "Track 4 (Dashstar)": "output/camelot/4_Expert_Energy_cut_Dom Dolla - San Frandisco (Extended Mix)_to_Knock2 - dashstar (Extended Mix).wav",
        "Track 1 (Rhyme Dust)": "output/camelot/1_Easy_SameKey_cut_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav"
    }
    
    for name, path in tracks.items():
        print(f"--- {name} ---")
        y, sr = librosa.load(path, sr=None, mono=True)
        
        # Look around 9.9 to 10.2 seconds
        start_sec = 9.9
        end_sec = 10.2
        start_samp = int(start_sec * sr)
        end_samp = int(end_sec * sr)
        
        y_zoom = y[start_samp:end_samp]
        
        # Find local peaks of the envelope
        rms = librosa.feature.rms(y=y_zoom, frame_length=256, hop_length=64)[0]
        
        # Time of max energy
        max_idx = np.argmax(rms)
        max_time_sec = start_sec + (max_idx * 64 / sr)
        
        print(f"Max energy peak occurs at: {max_time_sec:.4f} seconds (Swap is 10.0000s)")
        print(f"Offset from swap: {max_time_sec - 10.0:.4f} seconds")

analyze_peaks()
