import numpy as np
from scipy.signal import butter, lfilter

def generate_impact_fx(sr, duration_sec, b_first_beat=None):
    """
    Generates a classic EDM 'Downlifter' FX (Noise Sweep).
    Analyzes the High-Frequency content of B's first beat. If B is already bright,
    it reduces the volume of the noise sweep to avoid harshness/clashing.
    - Stereo white noise with exponential decay.
    - High-pass filter to prevent low-end mud.
    - No sub-drop (prevents phase cancellation 'trash can' effect).
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec))
    
    # Analyze B's HF content (above 5kHz)
    hf_rms = 0.0
    if b_first_beat is not None and len(b_first_beat) > 0:
        # Simple HPF for analysis
        b, a = butter(4, 5000 / (sr / 2), btype='high')
        b_hf = lfilter(b, a, b_first_beat[:, 0] if b_first_beat.ndim == 2 else b_first_beat)
        hf_rms = np.sqrt(np.mean(b_hf**2))
    
    # If HF RMS is high (e.g., > 0.05), B already has a crash cymbal. We scale the FX down.
    # If HF RMS is low (e.g., < 0.01), B is dull, so we play FX full volume.
    vol_scale = 1.0
    if hf_rms > 0.05:
        vol_scale = 0.0   # No FX needed, B is bright enough
    elif hf_rms > 0.01:
        # Scale between 0.0 and 1.0 based on how far it is from 0.01 to 0.05
        vol_scale = 1.0 - ((hf_rms - 0.01) / 0.04)
        
    if vol_scale <= 0.0:
        return None
        
    # White Noise
    noise = np.random.normal(0, 1, len(t)).astype(np.float32)
    
    # Exponential decay envelope
    env = np.exp(-t * 2.0)  # slightly faster decay
    noise *= env
    
    # Bandpass filter (HPF at 1000Hz, LPF at 8000Hz) to sit nicely in the mix
    # We remove lows so it doesn't clash with the kick
    b, a = butter(2, [1000 / (sr/2), 8000 / (sr/2)], btype='bandpass')
    filtered_noise = lfilter(b, a, noise)
    
    impact = (filtered_noise * 0.4 * vol_scale).astype(np.float32)
    
    # Stereo
    impact_stereo = np.column_stack((impact, impact))
    
    return impact_stereo
