import os
import soundfile as sf
import sounddevice as sd
import time
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
MIX_FILE = os.path.join(OUTPUT_DIR, "mashup_mix.wav")

def play_live():
    if not os.path.exists(MIX_FILE):
        print(f"Error: Could not find {MIX_FILE}. Run mixer.py first!")
        return
        
    print("Initializing Cloudy DJ Live Output Engine...")
    
    # We use soundfile to read the numpy array from the rendered mix
    data, sr = sf.read(MIX_FILE, dtype='float32')
    
    # The mix file is now already trimmed by mixer.py!
    print("DJ Bot taking over! Playing the AI-mixed HIGH ENERGY ACAPELLA MASHUP Live!")
    print("Listen closely as Track B's Vocals are layered perfectly over Track A's Instrumental Drop!")
    
    # Play the audio stream synchronously
    sd.play(data, sr)
    
    # Wait until file is done playing
    duration = 45 # 15s build + 15s blend + 15s of Track B alone
    print(f"Playing the transition zone for {duration} seconds...")
    sd.wait(duration)
    
    # Stop playback
    sd.stop()
    print("Playback complete!")

if __name__ == "__main__":
    play_live()
