import sys, os
sys.path.insert(0, '.')
from src.engine.generate_all_transitions import generate_bass_swap_transition

out_dir = "output/camelot"
os.makedirs(out_dir, exist_ok=True)

# Define the 4 levels based on the user's request
# Since tracks have ' (Extended Mix)' and other tags in their filenames, we match substrings to IDs.
levels = [
    {
        "level": "1_Easy_SameKey",
        "a": "Fisher and Aatig - Take It Off (Extended Mix)",
        "b": "Dom Dolla - Rhyme Dust (Extended Mix)"
    },
    {
        "level": "2_Medium_Neighbor",
        "a": "Dom Dolla - Rhyme Dust (Extended Mix)",
        "b": "John Summit and Hayla - Where You Are (Extended Mix)"
    },
    {
        "level": "3_Hard_Relative",
        "a": "Chris Lake - In The Yuma (Extended Mix) [feat. Aatig]",
        "b": "John Summit - Deep End (Extended Mix)"
    },
    {
        "level": "4_Expert_Energy",
        "a": "Dom Dolla - San Frandisco (Extended Mix)",
        "b": "Knock2 - dashstar (Extended Mix)"
    }
]

import json
crate = json.load(open("library/crate.json"))

def find_track_key(substring):
    for k in crate:
        if substring in k or k.startswith(substring):
            return k
    return None

for l in levels:
    print(f"\n--- Generating {l['level']} ---")
    track_a = find_track_key(l['a'].split(" (")[0])
    track_b = find_track_key(l['b'].split(" (")[0])
    
    if track_a and track_b:
        print(f"Match A: {track_a} ({crate[track_a]['key']}, {crate[track_a]['bpm']} BPM)")
        print(f"Match B: {track_b} ({crate[track_b]['key']}, {crate[track_b]['bpm']} BPM)")
        
        # We need the base ID (without .wav) for the generator
        id_a = track_a.replace(".wav", "")
        id_b = track_b.replace(".wav", "")
        
        out_name = f"{out_dir}/{l['level']}_{id_a}_to_{id_b}.wav"
        
        try:
            generate_bass_swap_transition(id_a, id_b, out_name)
        except Exception as e:
            print(f"Failed to generate {l['level']}: {e}")
    else:
        print(f"Could not find tracks in crate: A={track_a}, B={track_b}")
