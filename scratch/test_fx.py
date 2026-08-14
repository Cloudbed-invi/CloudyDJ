import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

def synthesize_downlifter(sr=44100, duration=4.0):
    # 1. White Noise
    t = np.linspace(0, duration, int(sr * duration))
    noise = np.random.normal(0, 1, len(t)).astype(np.float32)
    
    # 2. Envelope (exponential decay)
    env = np.exp(-t * 2)  # fast decay
    noise *= env
    
    # 3. Static LPF as a simple test (a swept LPF is better, but harder to do instantly without a custom filter loop)
    # We can just apply a gentle EQ or let the noise be broadband for the impact
    b, a = butter(2, 5000 / (sr/2), btype='low')
    filtered_noise = lfilter(b, a, noise)
    
    # 4. Sub drop (sine wave sliding from 80Hz to 30Hz)
    freqs = np.linspace(80, 20, len(t))
    phase = np.cumsum(freqs * 2 * np.pi / sr)
    sub = np.sin(phase) * env * 0.5
    
    # Mix
    impact = (filtered_noise * 0.5 + sub).astype(np.float32)
    
    # Stereo
    impact_stereo = np.column_stack((impact, impact))
    
    sf.write("scratch/test_impact.wav", impact_stereo, sr)
    print("Synthesized scratch/test_impact.wav")

if __name__ == "__main__":
    synthesize_downlifter()
