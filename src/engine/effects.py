import numpy as np
import pedalboard

def volume_fade(audio_data, start_ratio, end_ratio, duration_samples):
    """Applies a linear volume fade."""
    if duration_samples <= 0 or len(audio_data) == 0:
        return audio_data
        
    duration_samples = min(duration_samples, len(audio_data))
    
    fade_curve = np.linspace(start_ratio, end_ratio, duration_samples, dtype=np.float32)
    
    out = audio_data.copy()
    
    if len(out.shape) > 1:
        out[:duration_samples, 0] *= fade_curve
        out[:duration_samples, 1] *= fade_curve
    else:
        out[:duration_samples] *= fade_curve
        
    return out

def hpf_sweep(audio_data, start_freq, end_freq, duration_samples, sample_rate=44100):
    """High-pass filter sweep (washing out the low end)."""
    if duration_samples <= 0 or len(audio_data) == 0:
        return audio_data
        
    duration_samples = min(duration_samples, len(audio_data))
    out = audio_data.copy()
    
    # We apply the filter in chunks to simulate a sweep
    chunk_size = 2048
    num_chunks = duration_samples // chunk_size
    
    freqs = np.linspace(start_freq, end_freq, num_chunks)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        board = pedalboard.Pedalboard([
            pedalboard.HighpassFilter(cutoff_frequency_hz=freqs[i])
        ])
        
        chunk = out[start_idx:end_idx]
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)
            processed = board(chunk, sample_rate)
            out[start_idx:end_idx] = processed.flatten()
        else:
            processed = board(chunk.T, sample_rate)
            out[start_idx:end_idx] = processed.T
            
    return out

def lpf_sweep(audio_data, start_freq, end_freq, duration_samples, sample_rate=44100):
    """Low-pass filter sweep (muffling the sound)."""
    if duration_samples <= 0 or len(audio_data) == 0:
        return audio_data
        
    duration_samples = min(duration_samples, len(audio_data))
    out = audio_data.copy()
    
    chunk_size = 2048
    num_chunks = duration_samples // chunk_size
    
    freqs = np.linspace(start_freq, end_freq, num_chunks)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        
        board = pedalboard.Pedalboard([
            pedalboard.LowpassFilter(cutoff_frequency_hz=freqs[i])
        ])
        
        chunk = out[start_idx:end_idx]
        if len(chunk.shape) == 1:
            chunk = chunk.reshape(1, -1)
            processed = board(chunk, sample_rate)
            out[start_idx:end_idx] = processed.flatten()
        else:
            processed = board(chunk.T, sample_rate)
            out[start_idx:end_idx] = processed.T
            
    return out

def echo_out(audio_data, duration_samples, sample_rate=44100):
    """Applies a delay/echo effect that decays."""
    if duration_samples <= 0 or len(audio_data) == 0:
        return audio_data
        
    duration_samples = min(duration_samples, len(audio_data))
    out = audio_data.copy()
    
    # 3/4 beat delay at 128bpm is ~350ms
    board = pedalboard.Pedalboard([
        pedalboard.Delay(delay_seconds=0.35, feedback=0.6, mix=0.5)
    ])
    
    chunk = out[:duration_samples]
    if len(chunk.shape) == 1:
        chunk = chunk.reshape(1, -1)
        processed = board(chunk, sample_rate)
        out[:duration_samples] = processed.flatten()
    else:
        processed = board(chunk.T, sample_rate)
        out[:duration_samples] = processed.T
        
    # Fade out the original signal to just leave the echo
    out = volume_fade(out, 1.0, 0.0, duration_samples)
    
    return out

def reverb_throw(audio_data, duration_samples, sample_rate=44100):
    """Massive reverb tail."""
    if duration_samples <= 0 or len(audio_data) == 0:
        return audio_data
        
    duration_samples = min(duration_samples, len(audio_data))
    out = audio_data.copy()
    
    board = pedalboard.Pedalboard([
        pedalboard.Reverb(room_size=0.9, damping=0.1, width=1.0, wet_level=0.8, dry_level=0.2)
    ])
    
    chunk = out[:duration_samples]
    if len(chunk.shape) == 1:
        chunk = chunk.reshape(1, -1)
        processed = board(chunk, sample_rate)
        out[:duration_samples] = processed.flatten()
    else:
        processed = board(chunk.T, sample_rate)
        out[:duration_samples] = processed.T
        
    out = volume_fade(out, 1.0, 0.0, duration_samples)
        
    return out

def apply_effect(effect_name, audio_data, duration_samples, sample_rate=44100, **kwargs):
    """Router for all effects."""
    if effect_name == "fade_out":
        return volume_fade(audio_data, 1.0, 0.0, duration_samples)
    elif effect_name == "fade_in":
        return volume_fade(audio_data, 0.0, 1.0, duration_samples)
    elif effect_name == "hpf_sweep_up":
        return hpf_sweep(audio_data, 20, 2000, duration_samples, sample_rate)
    elif effect_name == "hpf_sweep_down":
        return hpf_sweep(audio_data, 2000, 20, duration_samples, sample_rate)
    elif effect_name == "lpf_sweep_down":
        return lpf_sweep(audio_data, 20000, 200, duration_samples, sample_rate)
    elif effect_name == "lpf_sweep_up":
        return lpf_sweep(audio_data, 200, 20000, duration_samples, sample_rate)
    elif effect_name == "echo_out":
        return echo_out(audio_data, duration_samples, sample_rate)
    elif effect_name == "reverb_throw":
        return reverb_throw(audio_data, duration_samples, sample_rate)
    else:
        print(f"Warning: Unknown effect {effect_name}")
        return audio_data
