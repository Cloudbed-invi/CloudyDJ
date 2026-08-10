import librosa
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf
import os
import json
import warnings

def analyze_track_structure(audio_path, sr=22050):
    """
    Analyzes an audio file to determine structural boundaries (Intro, Verse/Chorus, Drop, Outro).
    Returns a list of segment dictionaries: [{'label': 'intro', 'start_sample': 0, 'end_sample': X}, ...]
    """
    warnings.filterwarnings("ignore")
    
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # Extract chroma and beat-sync it
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    if len(beats) < 10:
        return [{"label": "full", "start_sample": 0, "end_sample": len(y)}]
        
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    
    # Calculate Self-Similarity Matrix
    rec = librosa.segment.recurrence_matrix(chroma_sync, mode='affinity', metric='cosine')
    
    # Enhance the diagonals
    rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)
    
    # Build sequence using Agglomerative Clustering
    # We expect roughly 4-6 distinct section types in a standard EDM track
    n_clusters = 5 
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    
    # Compute affinity matrix for clustering
    affinity = np.max(rec_smooth) - rec_smooth
    np.fill_diagonal(affinity, 0)
    
    # It might happen that the matrix is not perfectly symmetric due to path_enhance
    affinity = (affinity + affinity.T) / 2
    
    try:
        labels = clustering.fit_predict(affinity)
    except:
        return [{"label": "full", "start_sample": 0, "end_sample": len(y)}]
        
    # Find segment boundaries where the cluster label changes
    boundaries_beats = np.where(np.diff(labels) != 0)[0]
    
    segments = []
    start_beat_idx = 0
    
    for bound_idx in boundaries_beats:
        end_beat_idx = bound_idx
        
        # Ensure segments are at least 16 beats long
        if end_beat_idx - start_beat_idx >= 16:
            start_sample = librosa.frames_to_samples(beats[start_beat_idx])
            end_sample = librosa.frames_to_samples(beats[end_beat_idx])
            
            segments.append({
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "label_id": int(labels[start_beat_idx])
            })
            start_beat_idx = end_beat_idx + 1
            
    # Add final segment
    start_sample = librosa.frames_to_samples(beats[start_beat_idx]) if start_beat_idx < len(beats) else segments[-1]["end_sample"]
    segments.append({
        "start_sample": int(start_sample),
        "end_sample": len(y),
        "label_id": int(labels[start_beat_idx]) if start_beat_idx < len(labels) else -1
    })
    
    # Assign semantic labels (Intro, Outro, Main) based on position and energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    rms_sync = librosa.util.sync(rms, beats, aggregate=np.mean)[0] # Extract the 1D array
    
    for i, seg in enumerate(segments):
        # Calculate energy for this segment
        start_b = np.argmin(np.abs(librosa.frames_to_samples(beats) - seg["start_sample"]))
        end_b = np.argmin(np.abs(librosa.frames_to_samples(beats) - seg["end_sample"]))
        if start_b < end_b and start_b < len(rms_sync):
            energy = np.mean(rms_sync[start_b:end_b])
        else:
            energy = 0
        seg["energy"] = float(energy)
        
        if i == 0:
            seg["label"] = "intro"
        elif i == len(segments) - 1:
            seg["label"] = "outro"
        else:
            # Drop is usually the highest energy segment
            seg["label"] = "main"

    # Identify the highest energy main segment as the "drop"
    main_segments = [s for s in segments if s["label"] == "main"]
    if main_segments:
        drop_seg = max(main_segments, key=lambda s: s["energy"])
        drop_seg["label"] = "drop"
        
        # Sections before drop are buildup/verse
        found_drop = False
        for s in main_segments:
            if s == drop_seg:
                found_drop = True
            elif not found_drop:
                s["label"] = "buildup"
            else:
                s["label"] = "verse"

    return segments

if __name__ == "__main__":
    crate_file = "C:/Projects/Cloudy DJ 2.0/library/crate.json"
    with open(crate_file, "r") as f:
        crate = json.load(f)
        
    for name, data in crate.items():
        if "segments" not in data:
            print(f"Analyzing structure for {name}...")
            # We use the 'other' stem + 'bass' to avoid vocal-only or drum-only confusion
            # Or just use the original audio path if we had it. We'll use stems.
            audio_path = data["stems"]["other"]
            segments = analyze_track_structure(audio_path)
            data["segments"] = segments
            print(f"Found {len(segments)} segments.")
            
    with open(crate_file, "w") as f:
        json.dump(crate, f, indent=4)
        
    print("Structure analysis complete and saved to crate.json!")
