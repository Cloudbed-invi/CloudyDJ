import sys
import os

# Ensure the src module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def main():
    # Bug 6 Stereo Regression test
    print("Running Bug 6 Regression Test: Checking for Stereo Collapse...")
    from src.engine.generate_all_transitions import apply_warping
    # Create fake 4-beat stereo 120BPM stem
    import numpy as np
    fake_stem = np.random.randn(44100 * 2, 2).astype(np.float32)
    fake_stem[:, 0] *= 0.5 # L and R are different
    fake_track = {
        "name": "fake", "bpm": 120, "sr": 44100, "beats": np.array([0, 44100]), "key": 0, "drop_idx": 0,
        "vocals": fake_stem, "drums": fake_stem, "bass": fake_stem, "other": fake_stem
    }
    warped = apply_warping(fake_track, 130)
    is_mono = np.allclose(warped["vocals"][:, 0], warped["vocals"][:, 1])
    if is_mono:
        print("❌ FAILED: Warping collapsed stereo to mono!")
    else:
        print("✅ PASSED: Stereo field preserved after warping.\n")

from src.engine.generate_all_transitions import generate_bass_swap_transition
combinations = [
    ("Tiesto_Secrets", "James_Hype_Wild"),
    ("James_Hype_Wild", "Tiesto_Secrets")
]

modes = ["no_vocals_no_fx", "with_vocals_no_fx", "with_vocals_with_fx"]

out_dir = "c:/Users/sriha/Documents/Cloudy DJ 2.0/output"

for mode in modes:
    print(f"\n========================================")
    print(f"Generating Mode: {mode}")
    print(f"========================================")
    for i, (a_name, b_name) in enumerate(combinations):
        out_file = os.path.join(out_dir, f"{i+1}_{a_name}_to_{b_name}_{mode}.wav")
        print(f"\n[Bass Swap] {a_name} → {b_name} ({mode})")
        
        try:
            generate_bass_swap_transition(a_name, b_name, out_file, mode=mode)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
