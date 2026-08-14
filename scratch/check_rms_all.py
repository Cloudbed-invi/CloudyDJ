import librosa
import numpy as np

def analyze():
    print("--- Track 1: CUT ---")
    path = "output/camelot/1_Easy_SameKey_cut_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav"
    y, sr = librosa.load(path, sr=None, mono=True)
    drop = int(10 * sr)
    pre = np.sqrt(np.mean(y[drop - 5*sr : drop]**2))
    post = np.sqrt(np.mean(y[drop : drop + 5*sr]**2))
    print(f"Pre-drop RMS: {pre:.4f}")
    print(f"Post-drop RMS: {post:.4f}")
    
    print("\n--- Track 2: ECHO OUT ---")
    path2 = "output/camelot/2_Medium_Neighbor_echo_out_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)_to_JohnSummit_WhereYouAre.wav"
    y2, sr = librosa.load(path2, sr=None, mono=True)
    pre2 = np.sqrt(np.mean(y2[drop - 5*sr : drop]**2))
    post2 = np.sqrt(np.mean(y2[drop : drop + 5*sr]**2))
    print(f"Pre-drop RMS: {pre2:.4f}")
    print(f"Post-drop RMS: {post2:.4f}")
    
    # Let's check Track 1 gap again
    rms_frames = librosa.feature.rms(y=y[drop - 2000 : drop + 20000], frame_length=2048, hop_length=512)[0]
    print(f"Track 1 Min RMS near drop: {np.min(rms_frames):.4f}")

analyze()
