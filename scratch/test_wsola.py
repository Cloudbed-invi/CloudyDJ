import librosa
import numpy as np
import audiotsm
import audiotsm.io.array

def test_8_channel_wsola():
    # Create fake 8-channel audio
    sr = 22050
    t = np.linspace(0, 1.0, sr)
    fake_audio = np.random.randn(sr, 8).astype(np.float32)
    
    rate = 1.05
    target_len = int(len(fake_audio) / rate)
    
    print("Running WSOLA on 8 channels...")
    y_in = fake_audio.T
    reader = audiotsm.io.array.ArrayReader(y_in)
    tsm = audiotsm.wsola(channels=8, speed=rate)
    writer = audiotsm.io.array.ArrayWriter(channels=8)
    tsm.run(reader, writer)
    
    out = writer.data.T
    print(f"Original shape: {fake_audio.shape}")
    print(f"Stretched shape: {out.shape}")

test_8_channel_wsola()
