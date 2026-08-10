import os
import yt_dlp

# New 128 BPM House Tracks for testing Smooth EQ Blend
TRACKS = [
    "https://www.youtube.com/watch?v=3nQNiWdeH2Q", # Janji - Heroes Tonight (128 BPM)
]

def download_tracks():
    out_dir = os.path.join(os.path.dirname(__file__), "library", "raw_audio")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Downloading 5 tracks to {out_dir}...")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
        'ffmpeg_location': r'C:\Users\sriha\Documents\Cloudy DJ 2.0\ffmpeg-master-latest-win64-gpl\bin',
        'ignoreerrors': True,
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(TRACKS)

if __name__ == "__main__":
    download_tracks()
    print("Download complete.")
