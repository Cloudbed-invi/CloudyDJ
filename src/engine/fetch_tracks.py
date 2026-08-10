import os
import json
import subprocess
import librosa
import soundfile as sf
import yt_dlp
import numpy as np

LIBRARY_DIR = "C:/Users/sriha/Documents/Cloudy DJ 2.0/library"
CRATE_FILE = os.path.join(LIBRARY_DIR, "crate.json")

def download_audio(url, title):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(LIBRARY_DIR, f'{title}.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ffmpeg_location': 'C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin',
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return os.path.join(LIBRARY_DIR, f"{title}.wav")

def separate_stems(file_path, output_dir):
    print(f"Separating stems for {file_path}...")
    subprocess.run([
        "C:/Projects/Cloudy DJ 2.0/env/Scripts/demucs.exe",
        "-n", "htdemucs",
        "--two-stems", "vocals", # Just get vocals vs instrumental for now to save time, actually wait, we need bass and drums for beat/drop detection!
        "-o", output_dir,
        file_path
    ], check=True)

def separate_all_stems(file_path, output_dir):
    print(f"Separating ALL stems for {file_path}...")
    subprocess.run([
        "conda", "run", "-n", "cloudy_dj",
        "demucs",
        "-n", "htdemucs",
        "-o", output_dir,
        file_path
    ], check=True)

def update_crate(title, track_dir, bpm_hint=None):
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)
        
    stems = {
        "vocals": os.path.join(track_dir, "vocals.wav"),
        "drums": os.path.join(track_dir, "drums.wav"),
        "bass": os.path.join(track_dir, "bass.wav"),
        "other": os.path.join(track_dir, "other.wav")
    }
    
    # Calculate BPM from drums
    y, sr = sf.read(stems["drums"])
    if len(y.shape) > 1: y = y.mean(axis=1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if isinstance(tempo, (list, tuple, np.ndarray)) else float(tempo)
    
    if bpm_hint:
        # snap to hint if close
        if abs(bpm - bpm_hint) > 10:
            bpm = bpm_hint
            
    crate[title] = {
        "bpm": round(bpm),
        "stems": stems
    }
    
    with open(CRATE_FILE, 'w') as f:
        json.dump(crate, f, indent=4)
        
    print(f"Added {title} to crate with BPM {round(bpm)}")

if __name__ == "__main__":
    import numpy as np
    
    # 3. Elvis - Jailhouse Rock
    elvis_url = "ytsearch1:Elvis Presley - Jailhouse Rock (Official Audio)"
    elvis_file = download_audio(elvis_url, "Elvis_JailhouseRock")
    
    # 4. Enya - Orinoco Flow
    enya_url = "ytsearch1:Enya - Orinoco Flow (Official 4k Music Video)"
    enya_file = download_audio(enya_url, "Enya_OrinocoFlow")
    
    # Run Demucs
    demucs_out = os.path.join(LIBRARY_DIR, "demucs_output")
    separate_all_stems(elvis_file, demucs_out)
    separate_all_stems(enya_file, demucs_out)
    
    print("Fetch and separation complete! Now run build_crate.py to analyze structure and update crate.json.")
