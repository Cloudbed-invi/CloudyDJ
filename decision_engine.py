import os
import json

CRATE_FILE = os.path.join(os.path.dirname(__file__), "library", "crate.json")

def analyze_transition_strategy(track_a_name, track_b_name):
    """
    Analyzes the BPM and Key of Track A and Track B to determine
    the mathematically optimal transition strategy.
    """
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
        
    bpm_a = crate[track_a_name]["bpm"]
    bpm_b = crate[track_b_name]["bpm"]
    key_a = crate[track_a_name]["key"]
    key_b = crate[track_b_name]["key"]
    
    bpm_difference = abs(bpm_a - bpm_b)
    
    print(f"\n--- DJ Bot Brain: Analyzing Next Transition ---")
    print(f"Track A: {track_a_name} | {bpm_a} BPM | {key_a}")
    print(f"Track B: {track_b_name} | {bpm_b} BPM | {key_b}")
    print(f"BPM Variance: {bpm_difference:.2f} BPM")
    
    # 1. HUGE BPM DIFFERENCE -> Echo Out / Hard Drop
    if bpm_difference > 8:
        print("Strategy Selected: HARD CUT ECHO-OUT")
        print("Reason: BPM gap is too large to time-stretch without heavy distortion.")
        return "echo_out"
        
    # 2. ACAPELLA MASHUP -> Identical Keys
    if key_a == key_b:
        print("Strategy Selected: ACAPELLA MASHUP")
        print("Reason: Tracks are perfectly harmonic. Layering vocals over the new instrumental.")
        return "mashup"
        
    # 3. STANDARD SMOOTH EQ BLEND -> Similar BPMs
    print("Strategy Selected: SMOOTH EQ BLEND")
    print("Reason: Tempos are close. We can time-stretch safely and crossfade the basslines.")
    return "eq_blend"

if __name__ == "__main__":
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    tracks = list(crate.keys())
    if len(tracks) >= 2:
        strategy = analyze_transition_strategy(tracks[0], tracks[1])
        print(f"\nReady to execute {strategy} in mixer.py!")
