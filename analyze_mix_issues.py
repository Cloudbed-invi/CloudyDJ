import librosa
import numpy as np
import os
import json

def analyze_track_1():
    print("--- Analyzing Track 1 Gap ---")
    path = "output/camelot/1_Easy_SameKey_cut_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav"
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # Track 1 has a 10s pre-drop, so the drop is around 10s.
    # The gap should be around 10s. Let's find any window of low energy near the drop.
    drop_sample = int(10 * sr)
    search_start = max(0, drop_sample - sr*5)
    search_end = min(len(y), drop_sample + sr*5)
    
    rms = librosa.feature.rms(y=y[search_start:search_end], frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512) + (search_start / sr)
    
    threshold = 0.02
    low_energy_frames = np.where(rms < threshold)[0]
    
    if len(low_energy_frames) > 0:
        start_low = times[low_energy_frames[0]]
        end_low = times[low_energy_frames[-1]]
        print(f"Low energy gap detected from {start_low:.2f}s to {end_low:.2f}s. Duration: {end_low - start_low:.2f}s")
    else:
        print("No significant low energy gap detected near the drop.")
        print(f"Min RMS near drop: {np.min(rms):.4f}")

def analyze_track_2():
    print("\n--- Analyzing Track 2 Volume Increase ---")
    path = "output/camelot/2_Medium_Neighbor_echo_out_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav"
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # 10s pre-drop, 32-beat blend, 10s post
    drop_sample = int(10 * sr)
    
    # Measure LUFS/RMS before and after
    pre_rms = np.sqrt(np.mean(y[drop_sample - 5*sr:drop_sample]**2))
    post_rms = np.sqrt(np.mean(y[drop_sample:drop_sample + 5*sr]**2))
    
    print(f"RMS 5s before drop: {pre_rms:.4f}")
    print(f"RMS 5s after drop:  {post_rms:.4f}")
    
    if post_rms > pre_rms * 1.5:
        print(f"Volume surge detected! (+{(post_rms/pre_rms - 1)*100:.1f}%)")
        
def analyze_track_4():
    print("\n--- Analyzing Track 4 Clashing ---")
    # Actually, we can check the keys.
    # Dom Dolla is C Minor (5A), Knock2 is E Minor (9A).
    # 5A and 9A are completely clashing. 
    # But wait, generate_camelot_suite uses strategy based on these keys. 
    print("Track 4 Keys: Dom Dolla (C Minor) to Knock2 (E Minor)")
    print("These keys definitely clash (5A to 9A). The strategy 'CUT' was used.")
    print("However, if they are clashing despite the CUT, it might be due to vocals or melodies overlapping.")

analyze_track_1()
analyze_track_2()
analyze_track_4()
