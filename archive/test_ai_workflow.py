import json
import soundfile as sf
import os
from src.engine.song_card import generate_song_card
from src.engine.ai_prompt_generator import generate_transition_prompt
from src.engine.automation_executor import execute_automation_script

def test_pipeline():
    crate_path = "C:/Projects/Cloudy DJ 2.0/library/crate.json"
    with open(crate_path, "r") as f:
        crate = json.load(f)
        
    track_a = "Alan Walker - Faded"
    track_b = "Alan Walker - Spectre"
    
    # 1. Generate Song Cards
    print("Generating Song Cards...")
    card_a = generate_song_card(track_a, crate[track_a])
    card_b = generate_song_card(track_b, crate[track_b])
    
    # 2. Generate Prompt
    prompt_path = generate_transition_prompt(card_a, card_b)
    print(f"Prompt generated: {prompt_path}")
    
    # Use the drops from the song cards for the transition point
    track_a_start = max(0, card_a.get("drop_beat", 128) - 32)
    track_b_start = max(0, card_b.get("drop_beat", 64) - 64)
    
    # 3. MOCK GEMINI RESPONSE (User would normally copy-paste this)
    # We will pretend the user pasted this valid JSON script
    mock_script = {
        "track_a_start_beat": track_a_start,
        "track_a_end_beat": track_a_start + 64,
        "track_b_start_beat": track_b_start,
        "track_b_end_beat": track_b_start + 64,
        "actions": [
            {
                "track": "B",
                "stem": "bass",
                "beat_start": 0,
                "beat_end": 32,
                "effect": "hpf_sweep_down" # Sweeps from no bass down to full bass at beat 32
            },
            {
                "track": "A",
                "stem": "bass",
                "beat_start": 32,
                "beat_end": 64,
                "effect": "hpf_sweep_up" # Sweeps bass away starting at beat 32
            },
            {
                "track": "B",
                "stem": "drums",
                "beat_start": 16,
                "beat_end": 32,
                "effect": "fade_in"
            },
            {
                "track": "B",
                "stem": "other",
                "beat_start": 0,
                "beat_end": 32,
                "effect": "fade_in"
            },
            {
                "track": "B",
                "stem": "vocals",
                "beat_start": 0,
                "beat_end": 32,
                "effect": "fade_in"
            },
            {
                "track": "A",
                "stem": "other",
                "beat_start": 32,
                "beat_end": 64,
                "effect": "echo_out"
            },
            {
                "track": "A",
                "stem": "drums",
                "beat_start": 48,
                "beat_end": 64,
                "effect": "fade_out"
            },
            {
                "track": "A",
                "stem": "vocals",
                "beat_start": 32,
                "beat_end": 48,
                "effect": "reverb_throw"
            }
        ]
    }
    
    script_path = "C:/Projects/Cloudy DJ 2.0/prompts/automation_script.json"
    with open(script_path, "w") as f:
        json.dump(mock_script, f, indent=2)
        
    # 4. Execute
    with open(script_path, "r") as f:
        script = json.load(f)
        
    output_audio = execute_automation_script(script, card_a, card_b, crate)
    
    # 5. Save
    os.makedirs("C:/Projects/Cloudy DJ 2.0/output", exist_ok=True)
    out_file = "C:/Projects/Cloudy DJ 2.0/output/ai_transition_test.wav"
    sf.write(out_file, output_audio, 44100)
    print(f"Transition rendered and saved to {out_file}")

if __name__ == "__main__":
    test_pipeline()
