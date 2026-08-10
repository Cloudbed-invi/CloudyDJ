import json
import os
import numpy as np

ACTION_VOCABULARY = """
## Actions Vocabulary (Effects)
You can apply the following effects to any track or stem during the transition.
Note: You must output ONLY valid JSON.

- `volume_fade`: Linearly fades volume up or down.
- `hpf_sweep_up`: Sweeps a high-pass filter from 20Hz to 2000Hz (washes out the low end).
- `hpf_sweep_down`: Sweeps a high-pass filter from 2000Hz down to 20Hz (brings back the low end).
- `lpf_sweep_down`: Sweeps a low-pass filter from 20000Hz to 200Hz (muffles the sound).
- `lpf_sweep_up`: Sweeps a low-pass filter from 200Hz to 20000Hz (un-muffles the sound).
- `echo_out`: Applies a 3/4 beat delay and fades the original signal to 0. Great for ending a vocal phrase.
- `reverb_throw`: Applies a massive reverb tail and fades the original signal to 0. Great for drops.
"""

TRANSITION_RULES = """
## Rules
1. **Phrase Alignment**: All `beat_start` and `beat_end` values MUST align with the 16-beat phrase boundaries provided in the Song Cards.
2. **Stem-Level Control**: You can apply actions to specific stems: `drums`, `bass`, `vocals`, `other`. Or you can use `master` to apply to the whole mix.
3. **Mute by Default**: Track B will start playing at 100% volume at beat 0 UNLESS you explicitly apply a `fade_in` or `hpf_sweep_down` effect to its stems. Always use `fade_in` for Track B stems that shouldn't play immediately!
4. **No Clashing Frequencies**: If both songs have heavy bass or drums, you must apply a filter (`hpf_sweep_up` or `volume_fade`) to one of them to prevent mud.
5. **Vocals First**: If Song A has a vocal tail, use `echo_out` on the vocals before Song B's vocal begins.
6. **JSON Only**: You must respond ONLY with a raw JSON block. No markdown formatting, no explanations, no ` ```json ` blocks. Just the raw JSON object.

## JSON Output Schema
{
  "track_a_start_beat": int (e.g. 128),
  "track_a_end_beat": int (e.g. 192, where the track A stops playing),
  "track_b_start_beat": int (e.g. 0),
  "track_b_end_beat": int (e.g. 64, this is how much of track B to render for the transition),
  "actions": [
    {
      "track": "A" or "B",
      "stem": "drums", "bass", "vocals", "other", or "master",
      "beat_start": int (relative to the transition start, e.g. 0),
      "beat_end": int,
      "effect": "hpf_sweep_up" (from vocabulary)
    }
  ]
}
"""

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_transition_prompt(card_a, card_b, context="Mid-set, maintain energy"):
    """
    Generates a prompt text file for the user to paste into Gemini.
    """
    print(f"Generating prompt for transition: {card_a['title']} -> {card_b['title']}")
    
    prompt = f"You are an expert AI DJ. You need to design a stem-level transition between two songs.\n\n"
    prompt += f"Context for this mix: {context}\n\n"
    
    prompt += "### Track A (Playing)\n"
    prompt += json.dumps(card_a, indent=2, cls=NumpyEncoder) + "\n\n"
    
    prompt += "### Track B (Incoming)\n"
    prompt += json.dumps(card_b, indent=2, cls=NumpyEncoder) + "\n\n"
    
    prompt += ACTION_VOCABULARY + "\n"
    prompt += TRANSITION_RULES + "\n"
    
    os.makedirs("C:/Projects/Cloudy DJ 2.0/prompts", exist_ok=True)
    out_path = "C:/Projects/Cloudy DJ 2.0/prompts/transition_prompt.txt"
    with open(out_path, "w") as f:
        f.write(prompt)
        
    print(f"Prompt saved to {out_path}")
    return out_path
