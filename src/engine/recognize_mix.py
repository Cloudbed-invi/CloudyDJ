import os
import asyncio
from shazamio import Shazam
from pydub import AudioSegment
import yt_dlp

os.environ["PATH"] += os.pathsep + "C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin"
os.environ["FFMPEG_BINARY"] = "C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = "C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"

OUTPUT_DIR = "C:/Projects/Cloudy DJ 2.0/output"
MIX_AUDIO = os.path.join(OUTPUT_DIR, "mix_audio.wav")

def download_mix(url):
    print("Downloading DJ Mix...")
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(OUTPUT_DIR, 'mix_audio.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'ffmpeg_location': 'C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin',
        'playlist_items': '1',
        'quiet': False
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

async def recognize_tracks():
    print("Slicing audio and recognizing tracks...")
    shazam = Shazam()
    audio = AudioSegment.from_wav(MIX_AUDIO)
    
    # Slice the mix into 30 second chunks
    chunk_length_ms = 30000
    chunks = len(audio) // chunk_length_ms
    
    identified_tracks = []
    
    for i in range(chunks):
        start_ms = i * chunk_length_ms
        end_ms = start_ms + chunk_length_ms
        chunk = audio[start_ms:end_ms]
        
        chunk_file = os.path.join(OUTPUT_DIR, f"chunk_{i}.wav")
        chunk.export(chunk_file, format="wav")
        
        # Recognize
        out = await shazam.recognize_song(chunk_file)
        if 'track' in out:
            title = out['track']['title']
            artist = out['track']['subtitle']
            print(f"[{i*30}s - {(i+1)*30}s] Detected: {artist} - {title}")
            
            track_name = f"{artist} - {title}"
            if len(identified_tracks) == 0 or identified_tracks[-1] != track_name:
                identified_tracks.append(track_name)
        else:
            print(f"[{i*30}s - {(i+1)*30}s] No track detected")
            
        os.remove(chunk_file)
        
    print("\n--- Identified Setlist ---")
    for idx, t in enumerate(identified_tracks):
        print(f"{idx+1}. {t}")

if __name__ == "__main__":
    import sys
    url = "https://www.youtube.com/playlist?list=PLazRGdDsvL2Cg5lLeTiKrRhAWxXqoN_xB"
    
    # Download
    download_mix(url)
    
    # Recognize
    asyncio.run(recognize_tracks())
