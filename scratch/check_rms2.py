import librosa
import numpy as np

def analyze():
    path = "output/camelot/2_Medium_Neighbor_echo_out_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav"
    y, sr = librosa.load(path, sr=None, mono=True)
    
    # 10s pre-drop is the transition point
    drop_sample = int(10 * sr)
    
    # Let's measure RMS for the first 5 seconds of Track A (which is A's drop)
    a_drop_rms = np.sqrt(np.mean(y[0 : 5*sr]**2))
    
    # Let's measure RMS for the first 5 seconds of Track B's drop (post_drop)
    b_drop_rms = np.sqrt(np.mean(y[drop_sample : drop_sample + 5*sr]**2))
    
    print(f"A's Drop RMS (0-5s): {a_drop_rms:.4f}")
    print(f"B's Drop RMS (10-15s): {b_drop_rms:.4f}")
    print(f"Difference: +{((b_drop_rms/a_drop_rms)-1)*100:.1f}%")

analyze()
