import sys, os
sys.path.insert(0, '.')
from src.engine.generate_all_transitions import generate_bass_swap_transition

out_dir = "output/camelot"
os.makedirs(out_dir, exist_ok=True)

levels = [
    {
        "level": "1_Easy_SameKey",
        "a": "FISHER & Aatig - Take It Off (Extended Mix)",
        "b": "MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)"
    },
    {
        "level": "2_Medium_Neighbor",
        "a": "MK, Dom Dolla - Rhyme Dust (Extended - Official Audio)",
        "b": "JohnSummit_WhereYouAre"
    }
]

import json
crate = json.load(open("library/crate.json"))

for l in levels:
    print(f"\n--- Generating {l['level']} ---")
    track_a = l['a'] + ".wav"
    track_b = l['b'] + ".wav"
    
    if track_a in crate and track_b in crate:
        print(f"Match A: {track_a} ({crate[track_a]['key']}, {crate[track_a]['bpm']} BPM)")
        print(f"Match B: {track_b} ({crate[track_b]['key']}, {crate[track_b]['bpm']} BPM)")
        
        # We need the base ID (without .wav) for the generator
        id_a = l['a']
        id_b = l['b']
        
        out_name = f"{out_dir}/{l['level']}_{id_a}_to_{id_b}.wav"
        
        try:
            generate_bass_swap_transition(id_a, id_b, out_name)
        except Exception as e:
            print(f"Failed to generate {l['level']}: {e}")
    else:
        print(f"Could not find tracks in crate: A={track_a}, B={track_b}")
