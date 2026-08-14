import os
from src.engine.generate_all_transitions import generate_transition

out_dir = "output/all_strategies"
os.makedirs(out_dir, exist_ok=True)

track_a = "Tiesto_Secrets"
track_b = "Knock2 - dashstar (Extended Mix)"
# Removing the .wav if present, generate_transition accepts the base ID.
# Actually, Knock2 - dashstar is usually 'Knock2 - dashstar (Extended Mix)' in crate.

strategies = [
    "cut",
    "long_blend",
    "filter_sweep",
    "echo_out",
    "loop_and_build"
]

for strategy in strategies:
    out_name = f"{out_dir}/{track_a}_to_{track_b}_{strategy}.wav"
    print(f"\n========================================")
    print(f"Generating Strategy: {strategy.upper()}")
    print(f"========================================")
    try:
        generate_transition(track_a, track_b, out_name, strategy=strategy)
        print(f"-> SUCCESS: Saved {out_name}")
    except Exception as e:
        print(f"-> ERROR: Failed to generate {strategy}: {e}")
