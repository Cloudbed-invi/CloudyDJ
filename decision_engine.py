import os
import json
import numpy as np

CRATE_FILE = os.path.join(os.path.dirname(__file__), "library", "crate.json")

CAMELOT_MAP = {
    "C": (8, "B"), "Am": (8, "A"), "A": (11, "B"), "F#m": (11, "A"),
    "G": (9, "B"), "Em": (9, "A"), "E": (12, "B"), "C#m": (12, "A"),
    "D": (10, "B"), "Bm": (10, "A"), "B": (1, "B"), "G#m": (1, "A"),
    "F#": (2, "B"), "D#m": (2, "A"), "Db": (3, "B"), "Bbm": (3, "A"),
    "Ab": (4, "B"), "Fm": (4, "A"), "Eb": (5, "B"), "Cm": (5, "A"),
    "Bb": (6, "B"), "Gm": (6, "A"), "F": (7, "B"), "Dm": (7, "A"),
    
    # Enharmonics
    "D#": (2, "A"), "A#": (3, "A"), "G#": (1, "A"),
    "C#": (12, "A"), "E": (9, "A")
}

def _camelot_distance(key_a, key_b):
    # Normalize strings like "C Minor" -> "Cm", "G Major" -> "G"
    ka = key_a.replace(" Minor", "m").replace(" Major", "")
    kb = key_b.replace(" Minor", "m").replace(" Major", "")
    
    ca = CAMELOT_MAP.get(ka)
    cb = CAMELOT_MAP.get(kb)
    if not ca or not cb: return "clash"
    
    num_a, let_a = ca
    num_b, let_b = cb
    num_diff = min(abs(num_a - num_b), 12 - abs(num_a - num_b))
    
    if num_diff == 0 and let_a == let_b: return "same"
    if num_diff == 0 and let_a != let_b: return "relative"
    if num_diff == 1 and let_a == let_b: return "neighbor"
    if num_diff == 1 and let_a != let_b: return "diagonal"
    return "clash"

def _get_drop_energy(crate_data, track_name):
    # Try to find the segment containing the drop_idx
    track = crate_data[track_name]
    if "segments" in track:
        for seg in track["segments"]:
            if seg["label"] == "drop":
                return seg["energy"]
        # Fallback to the highest energy segment
        return max([s["energy"] for s in track["segments"]]) if track["segments"] else 0.5
    return 0.5

def analyze_transition_strategy(track_a_name, track_b_name):
    """
    Analyzes the BPM, Key, and Energy of Track A and Track B to determine
    the optimal Mixgraph transition strategy.
    """
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    if not track_a_name.endswith('.wav'): track_a_name += '.wav'
    if not track_b_name.endswith('.wav'): track_b_name += '.wav'
    
    bpm_a = crate[track_a_name]["bpm"]
    bpm_b = crate[track_b_name]["bpm"]
    key_a = crate[track_a_name]["key"]
    key_b = crate[track_b_name]["key"]
    
    energy_a = _get_drop_energy(crate, track_a_name)
    energy_b = _get_drop_energy(crate, track_b_name)
    
    bpm_diff = abs(bpm_a - bpm_b)
    key_dist = _camelot_distance(key_a, key_b)
    energy_shift = abs(energy_a - energy_b) / max(energy_a, 0.001) > 0.15 # >15% difference
    
    print(f"\n--- Mixgraph Engine: Analyzing Transition ---")
    print(f"Track A: {track_a_name} | {bpm_a} BPM | {key_a} | Energy: {energy_a:.3f}")
    print(f"Track B: {track_b_name} | {bpm_b} BPM | {key_b} | Energy: {energy_b:.3f}")
    
    print(f"Rhythmic Compatibility: {'LOW' if bpm_diff > 3 else 'HIGH'} ({bpm_diff:.1f} BPM diff)")
    print(f"Harmonic Compatibility: {key_dist.upper()}")
    print(f"Energy Shift: {'HIGH' if energy_shift else 'SIMILAR'}")
    
    # Mixgraph Decision Matrix
    if bpm_diff > 4:
        print("Strategy Selected: THE ECHO OUT")
        print("Reason: Rhythmic compatibility is too low. Echo out creates a tempo reset.")
        return "echo_out"
        
    if energy_shift:
        print("Strategy Selected: THE FILTER SWEEP")
        print("Reason: Significant energy gap between drops. Filters will bridge the contrast.")
        return "filter_sweep"
        
    if key_dist in ["same", "relative", "neighbor"]:
        print("Strategy Selected: THE LONG BLEND")
        print("Reason: High harmonic compatibility and similar energy. Seamless 32-bar mix.")
        return "long_blend"
        
    print("Strategy Selected: THE CUT")
    print("Reason: Keys clash but BPM and energy are stable. Instant swap avoids dissonance.")
    return "cut"

if __name__ == "__main__":
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
    tracks = list(crate.keys())
    if len(tracks) >= 2:
        strategy = analyze_transition_strategy(tracks[0], tracks[1])
        print(f"\nReady to execute {strategy}!")
