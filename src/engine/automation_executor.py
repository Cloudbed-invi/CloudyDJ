import numpy as np
import librosa
from .effects import apply_effect
from .song_card import load_audio

def execute_automation_script(script, track_a_card, track_b_card, crate):
    """
    Renders audio based on an AI-generated JSON automation script.
    """
    print("Executing automation script...")
    
    # 1. Load Audio Stems
    sr = 44100
    stems_a = {}
    stems_b = {}
    
    for stem_name, path in crate[track_a_card["title"]]["stems"].items():
        stems_a[stem_name], _ = load_audio(path)
        
    for stem_name, path in crate[track_b_card["title"]]["stems"].items():
        stems_b[stem_name], _ = load_audio(path)
        
    # Full mix arrays
    full_mix_a = stems_a["drums"] + stems_a["bass"] + stems_a["other"] + stems_a["vocals"]
    full_mix_b = stems_b["drums"] + stems_b["bass"] + stems_b["other"] + stems_b["vocals"]
    
    # 2. Extract Timing Information
    # Using Track B's BPM as the master BPM for the transition
    master_bpm = track_b_card["bpm"]
    
    # Calculate samples per beat
    samples_per_beat = int((60.0 / master_bpm) * sr)
    
    # Length of the transition in beats
    trans_beats = script["track_a_end_beat"] - script["track_a_start_beat"]
    trans_samples = trans_beats * samples_per_beat
    
    # 3. Create Output Buffers for the transition section
    # We create a buffer that holds the transition, plus a 4-bar tail for lingering effects
    tail_beats = 16
    tail_samples = tail_beats * samples_per_beat
    
    out_buffer = np.zeros(trans_samples + tail_samples, dtype=np.float32)
    
    # Track A base audio (time stretched to master BPM if needed)
    start_a_sec = script["track_a_start_beat"] * (60.0 / track_a_card["bpm"])
    end_a_sec = script["track_a_end_beat"] * (60.0 / track_a_card["bpm"])
    
    start_a_samp = int(start_a_sec * sr)
    end_a_samp = int(end_a_sec * sr)
    
    if track_a_card["bpm"] != master_bpm:
        rate = master_bpm / track_a_card["bpm"]
        # Time stretch A's stems
        for stem in stems_a:
            chunk = stems_a[stem][start_a_samp:end_a_samp]
            if len(chunk) > 0:
                stems_a[stem] = librosa.effects.time_stretch(chunk, rate=rate)
            else:
                stems_a[stem] = np.zeros(trans_samples, dtype=np.float32)
    else:
        for stem in stems_a:
            stems_a[stem] = stems_a[stem][start_a_samp:end_a_samp]
            
    # Track B base audio
    start_b_sec = script["track_b_start_beat"] * (60.0 / track_b_card["bpm"])
    end_b_sec = script["track_b_end_beat"] * (60.0 / track_b_card["bpm"])
    
    start_b_samp = int(start_b_sec * sr)
    end_b_samp = int(end_b_sec * sr)
    
    for stem in stems_b:
        # Include tail for B
        tail_b_samp = min(end_b_samp + tail_samples, len(stems_b[stem]))
        chunk = stems_b[stem][start_b_samp:tail_b_samp]
        # Pad if short
        if len(chunk) < trans_samples + tail_samples:
            chunk = np.pad(chunk, (0, (trans_samples + tail_samples) - len(chunk)))
        stems_b[stem] = chunk
        
    # 4. Apply Actions
    for action in script.get("actions", []):
        track = action["track"] # "A" or "B"
        stem = action["stem"] # "drums", "bass", "vocals", "other", or "master"
        effect = action["effect"]
        b_start = action["beat_start"] # Relative to start of transition (0 = start of transition)
        b_end = action["beat_end"]
        
        samp_start = b_start * samples_per_beat
        samp_end = b_end * samples_per_beat
        duration_samples = samp_end - samp_start
        
        if duration_samples <= 0:
            continue
            
        if track == "A":
            target_dict = stems_a
        else:
            target_dict = stems_b
            
        if stem == "master":
            # Apply to all stems
            for s in target_dict:
                chunk = target_dict[s][samp_start:samp_end]
                processed = apply_effect(effect, chunk, duration_samples, sr)
                target_dict[s][samp_start:samp_end] = processed
                
                # State persistence
                if effect in ["fade_out", "echo_out", "reverb_throw", "hpf_sweep_up", "lpf_sweep_down"]:
                    if samp_end < len(target_dict[s]):
                        target_dict[s][samp_end:] = 0.0
                elif effect in ["fade_in"]:
                    if samp_start > 0:
                        target_dict[s][:samp_start] = 0.0
        else:
            chunk = target_dict[stem][samp_start:samp_end]
            processed = apply_effect(effect, chunk, duration_samples, sr)
            target_dict[stem][samp_start:samp_end] = processed
            
            # State persistence
            if effect in ["fade_out", "echo_out", "reverb_throw", "hpf_sweep_up", "lpf_sweep_down"]:
                if samp_end < len(target_dict[stem]):
                    target_dict[stem][samp_end:] = 0.0
            elif effect in ["fade_in"]:
                if samp_start > 0:
                    target_dict[stem][:samp_start] = 0.0
            
    # 5. Mix Down
    mix_a = np.zeros_like(stems_a["drums"])
    if len(mix_a) > 0:
        mix_a = stems_a["drums"] + stems_a["bass"] + stems_a["other"] + stems_a["vocals"]
        
    mix_b = stems_b["drums"] + stems_b["bass"] + stems_b["other"] + stems_b["vocals"]
    
    # Pad A with zeros for the tail
    if len(mix_a) < len(mix_b):
        mix_a = np.pad(mix_a, (0, len(mix_b) - len(mix_a)))
    elif len(mix_b) < len(mix_a):
        mix_b = np.pad(mix_b, (0, len(mix_a) - len(mix_b)))
        
    out_buffer = mix_a + mix_b
    
    # Normalize
    max_val = np.max(np.abs(out_buffer))
    if max_val > 0.95:
        out_buffer = (out_buffer / max_val) * 0.95
        
    return out_buffer
