import numpy as np
import librosa
from src.engine import dsp_utils

def test_lufs():
    print("Testing LUFS measurement and gain matching...")
    pathA = "output/camelot/1_Easy_SameKey_cut_FISHER & Aatig - Take It Off (Extended Mix)_to_MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav"
    # Actually, I'll just load the stems for Rhyme Dust and Where You Are.
    path_A_stems = "library/demucs_output/htdemucs/MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)"
    path_B_stems = "library/demucs_output/htdemucs/JohnSummit_WhereYouAre"
    
    # We don't have the exact warped stems here easily, let's just load the raw mix of A and B
    yA, sr = librosa.load("library/raw_audio/MK, Dom Dolla - Rhyme Dust (Extended - Official Audio).wav", sr=None, mono=True)
    yB, _ = librosa.load("library/raw_audio/JohnSummit_WhereYouAre.wav", sr=sr, mono=True)
    
    # Drops
    drop_A = 291 # beats... we need sample indices. Just use the first 5 seconds.
    # It's fine, let's just use check_rms2.py logic:
    # A's Drop RMS is 0.1400.
    
    lufs_A = dsp_utils.measure_lufs(yA[int(45*sr):int(50*sr)], sr)
    lufs_B = dsp_utils.measure_lufs(yB[int(30*sr):int(35*sr)], sr)
    
    print(f"LUFS A (Rhyme Dust): {lufs_A}")
    print(f"LUFS B (Where You Are): {lufs_B}")
    
    diff_db = lufs_A - lufs_B
    gain = 10.0 ** (diff_db / 20.0)
    print(f"Gain calculated: {gain}")
    
test_lufs()
