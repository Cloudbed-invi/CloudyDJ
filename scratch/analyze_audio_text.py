import librosa
import numpy as np

def analyze_track_4_beat():
    print("--- Track 4 (San Frandisco -> Dashstar) ---")
    mix_path = "output/camelot/4_Expert_Energy_cut_Dom Dolla - San Frandisco (Extended Mix)_to_Knock2 - dashstar (Extended Mix).wav"
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    
    # Analyze exactly at 10.000s
    drop_sample = int(10.0 * sr)
    beat_samples = int(0.47 * sr) # ~1 beat at 126 BPM
    
    # Check max peak in first 50ms (is the kick happening here?)
    first_50ms = y_mix[drop_sample : drop_sample + int(0.05 * sr)]
    print(f"Max peak in first 50ms of B: {np.max(np.abs(first_50ms)):.4f}")
    
    # Check max peak in the second beat (10.47s to 10.52s)
    second_beat = y_mix[drop_sample + beat_samples : drop_sample + beat_samples + int(0.05 * sr)]
    print(f"Max peak in second beat of B: {np.max(np.abs(second_beat)):.4f}")

def analyze_track_2_volume():
    print("--- Track 2 (Rhyme Dust -> Where You Are) ---")
    mix_path = "output/camelot/2_Medium_Neighbor_echo_out_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav"
    y_mix, sr = librosa.load(mix_path, sr=None, mono=True)
    
    drop = int(10.0 * sr)
    # Print RMS over 1-second chunks around the drop
    for sec in range(7, 13):
        start = int(sec * sr)
        end = int((sec + 1) * sr)
        chunk = y_mix[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        print(f"RMS [{sec}s - {sec+1}s]: {rms:.4f}")

if __name__ == "__main__":
    analyze_track_4_beat()
    analyze_track_2_volume()
