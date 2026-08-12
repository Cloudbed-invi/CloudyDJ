import os
from src.engine.generate_all_transitions import generate_bass_swap_transition

out_dir = "c:/Users/sriha/Documents/Cloudy DJ 2.0/output"

# Generate 1 (Sparse)
print("Generating Tiesto -> James Hype")
generate_bass_swap_transition(
    "Tiesto_Secrets", 
    "James_Hype_Wild", 
    os.path.join(out_dir, "1_Tiesto_Secrets_to_James_Hype_Wild.wav")
)

# Generate 2 (Dense)
print("Generating James Hype -> Tiesto")
generate_bass_swap_transition(
    "James_Hype_Wild", 
    "Tiesto_Secrets", 
    os.path.join(out_dir, "2_James_Hype_Wild_to_Tiesto_Secrets.wav")
)
