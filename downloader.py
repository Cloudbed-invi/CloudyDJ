import os
import yt_dlp

TRACKS = [
    "ytsearch1:Fisher and Aatig Take It Off Extended Mix",
    "ytsearch1:Dom Dolla Rhyme Dust Extended Mix",
    "ytsearch1:Chris Lake In The Yuma Extended Mix",
    "ytsearch1:John Summit Deep End Extended Mix",
    "ytsearch1:Dom Dolla San Frandisco Extended Mix",
    "ytsearch1:Knock2 dashstar Extended Mix",
]

def download_tracks():
    out_dir = os.path.join(os.path.dirname(__file__), "library", "raw_audio")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Downloading {len(TRACKS)} tracks to {out_dir}...")
    
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
