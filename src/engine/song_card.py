import os
import json
import librosa
import numpy as np
import soundfile as sf
import whisper_timestamped as whisper
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add ffmpeg to PATH for whisper
FFMPEG_PATH = "C:/Users/sriha/Documents/Cloudy DJ 2.0/ffmpeg-master-latest-win64-gpl/bin"
if FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

CAMELOT_WHEEL = {
    'C major': '8B', 'C minor': '5A',
    'C# major': '3B', 'C# minor': '12A',
    'D major': '10B', 'D minor': '7A',
    'D# major': '5B', 'D# minor': '2A',
    'E major': '12B', 'E minor': '9A',
    'F major': '7B', 'F minor': '4A',
    'F# major': '2B', 'F# minor': '11A',
    'G major': '9B', 'G minor': '6A',
    'G# major': '4B', 'G# minor': '3A',
    'A major': '11B', 'A minor': '8A',
    'A# major': '6B', 'A# minor': '5A',
    'B major': '1B', 'B minor': '10A'
}

def load_audio(path):
    y, sr = sf.read(path, dtype='float32')
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)
    return y, sr

def calculate_energy_curves(stems, beat_grid, sr=44100):
    """Calculate RMS energy for full mix, vocals, and bass grouped by phrase."""
    # We will assume 16-beat phrases (4 bars)
    phrase_len = 16
    curves = {
        "energy_curve_per_phrase": [],
        "vocal_density_per_phrase": [],
        "bass_density_per_phrase": []
    }
    
    # Calculate full mix by summing stems
    full_mix = stems["drums"] + stems["bass"] + stems["other"] + stems["vocals"]
    
    num_phrases = len(beat_grid) // phrase_len
    
    for i in range(num_phrases):
        start_beat = i * phrase_len
        end_beat = min((i + 1) * phrase_len, len(beat_grid) - 1)
        
        start_sample = beat_grid[start_beat]
        end_sample = beat_grid[end_beat]
        
        # Ensure we have audio
        if start_sample >= end_sample:
            continue
            
        # Full mix energy (RMS)
        chunk_full = full_mix[start_sample:end_sample]
        rms_full = librosa.feature.rms(y=chunk_full, frame_length=2048, hop_length=512)[0]
        curves["energy_curve_per_phrase"].append(round(float(np.mean(rms_full) * 10), 2))
        
        # Vocal density
        chunk_vocal = stems["vocals"][start_sample:end_sample]
        rms_vocal = librosa.feature.rms(y=chunk_vocal, frame_length=2048, hop_length=512)[0]
        curves["vocal_density_per_phrase"].append(round(float(np.mean(rms_vocal) * 10), 2))
        
        # Bass density
        chunk_bass = stems["bass"][start_sample:end_sample]
        rms_bass = librosa.feature.rms(y=chunk_bass, frame_length=2048, hop_length=512)[0]
        curves["bass_density_per_phrase"].append(round(float(np.mean(rms_bass) * 10), 2))
        
    return curves

def detect_key_librosa(y, sr):
    y_harmonic, _ = librosa.effects.hpss(y)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_key = None
    best_corr = -2
    
    from scipy.stats import pearsonr
    for i in range(12):
        maj_corr = pearsonr(np.roll(mean_chroma, -i), major_profile)[0]
        min_corr = pearsonr(np.roll(mean_chroma, -i), minor_profile)[0]
        
        if maj_corr > best_corr:
            best_corr = maj_corr
            best_key = f"{keys[i]} major"
        if min_corr > best_corr:
            best_corr = min_corr
            best_key = f"{keys[i]} minor"
            
    return best_key

def get_whisper_transcription(vocal_path, beat_grid, sr_stem=44100):
    model = whisper.load_model("base", device="cpu")
    print("Running Whisper on vocals...")
    result = whisper.transcribe(model, vocal_path, language="en")
    
    words_data = []
    best_loop_words = []
    
    if "segments" not in result:
        return words_data, best_loop_words
        
    for segment in result["segments"]:
        for word in segment.get("words", []):
            text = word["text"].strip().lower()
            start_sec = word["start"]
            end_sec = word["end"]
            
            start_sample = int(start_sec * sr_stem)
            end_sample = int(end_sec * sr_stem)
            
            # Find closest beat
            beat_start_idx = np.argmin(np.abs(beat_grid - start_sample))
            beat_end_idx = np.argmin(np.abs(beat_grid - end_sample))
            
            duration_beats = beat_end_idx - beat_start_idx
            if duration_beats == 0: duration_beats = 0.5
            
            words_data.append({
                "word": text,
                "beat": int(beat_start_idx),
                "duration_beats": duration_beats
            })
            
            if duration_beats >= 0.5 and len(text) > 2:
                best_loop_words.append(text)
                
    return words_data, best_loop_words

from sklearn.cluster import AgglomerativeClustering

def analyze_track_structure(y, sr, beat_grid, sr_stem=44100):
    print("Running librosa structure analysis...")
    
    # Resample to 22050 for analysis speed if needed
    if sr != 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050
        
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    _, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    if len(beats) < 10:
        return [{"label": "full", "beat_start": 0, "beat_end": len(beat_grid)}], 0
        
    chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.median)
    rec = librosa.segment.recurrence_matrix(chroma_sync, mode='affinity', metric='cosine')
    rec_smooth = librosa.segment.path_enhance(rec, 51, window='hann', n_filters=7)
    
    n_clusters = 5 
    clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    
    affinity = np.max(rec_smooth) - rec_smooth
    np.fill_diagonal(affinity, 0)
    affinity = (affinity + affinity.T) / 2
    
    try:
        labels = clustering.fit_predict(affinity)
    except:
        return [{"label": "full", "beat_start": 0, "beat_end": len(beat_grid)}], 0
        
    boundaries_beats = np.where(np.diff(labels) != 0)[0]
    
    segments = []
    start_beat_idx = 0
    
    for bound_idx in boundaries_beats:
        end_beat_idx = bound_idx
        if end_beat_idx - start_beat_idx >= 16:
            start_sample = librosa.frames_to_samples(beats[start_beat_idx])
            end_sample = librosa.frames_to_samples(beats[end_beat_idx])
            
            # Map back to 44100 beat grid
            b_start = np.argmin(np.abs(beat_grid - int(start_sample * (sr_stem / sr))))
            b_end = np.argmin(np.abs(beat_grid - int(end_sample * (sr_stem / sr))))
            
            segments.append({
                "beat_start": int(b_start),
                "beat_end": int(b_end),
                "label_id": int(labels[start_beat_idx])
            })
            start_beat_idx = end_beat_idx + 1
            
    # Add final
    start_sample = librosa.frames_to_samples(beats[start_beat_idx]) if start_beat_idx < len(beats) else segments[-1]["beat_end"]
    b_start = np.argmin(np.abs(beat_grid - int(start_sample * (sr_stem / sr))))
    segments.append({
        "beat_start": int(b_start),
        "beat_end": len(beat_grid),
        "label_id": int(labels[start_beat_idx]) if start_beat_idx < len(labels) else -1
    })
    
    # Assign labels based on energy
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)
    rms_sync = librosa.util.sync(rms, beats, aggregate=np.mean)[0]
    
    drop_beat = 0
    max_energy = -1
    
    for i, seg in enumerate(segments):
        start_b = np.argmin(np.abs(librosa.frames_to_samples(beats) - beat_grid[seg["beat_start"]]/(sr_stem/sr)))
        end_b = np.argmin(np.abs(librosa.frames_to_samples(beats) - beat_grid[min(seg["beat_end"], len(beat_grid)-1)]/(sr_stem/sr)))
        
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
            seg["label"] = "main"
            
        if energy > max_energy:
            max_energy = energy
            drop_beat = seg["beat_start"]
            
    # Refine main labels
    found_drop = False
    for s in segments:
        if s["label"] == "main":
            if s["beat_start"] == drop_beat:
                s["label"] = "drop"
                found_drop = True
            elif not found_drop:
                s["label"] = "buildup"
            else:
                s["label"] = "verse"
                
    return segments, drop_beat

def generate_song_card(track_name, crate_data):
    print(f"Generating Song Card for {track_name}...")
    
    card = {
        "title": track_name,
    }
    
    # 1. Load Stems
    stems = {}
    for stem_name, path in crate_data["stems"].items():
        stems[stem_name], sr = load_audio(path)
        
    full_mix = stems["drums"] + stems["bass"] + stems["other"] + stems["vocals"]
    
    # 2. BPM and Beat Grid
    print("Detecting BPM...")
    tempo, beats = librosa.beat.beat_track(y=full_mix, sr=sr)
    card["bpm"] = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    beat_grid = librosa.frames_to_samples(beats)
    card["duration_beats"] = len(beat_grid)
    card["phrase_boundaries"] = [int(i) for i in range(0, len(beat_grid), 16)]
    
    # 3. Key Detection
    print("Detecting Key...")
    key = detect_key_librosa(full_mix, sr)
    card["key"] = key
    card["camelot"] = CAMELOT_WHEEL.get(key, "Unknown")
    
    # 4. Energy Curves
    print("Calculating Energy Curves...")
    curves = calculate_energy_curves(stems, beat_grid, sr)
    card.update(curves)
    
    card["energy_score"] = round(np.mean(curves["energy_curve_per_phrase"]) + (np.mean(curves["bass_density_per_phrase"]) * 0.5), 1)
    
    # 5. Structure Detection (librosa recurrence matrix)
    try:
        segments, drop_beat = analyze_track_structure(full_mix, sr, beat_grid, sr)
        card["segments"] = segments
        card["drop_beat"] = int(drop_beat)
    except Exception as e:
        print(f"Structure analysis failed: {e}. Falling back to empty segments.")
        card["segments"] = []
        card["drop_beat"] = 64
        
    # 6. Whisper Transcription
    try:
        words, best_words = get_whisper_transcription(crate_data["stems"]["vocals"], beat_grid, sr)
        card["vocal_transcript"] = words
        card["best_loop_words"] = best_words
    except Exception as e:
        print(f"Whisper failed: {e}")
        card["vocal_transcript"] = []
        card["best_loop_words"] = []
        
    # 7. Mixable Zones
    mix_in = []
    mix_out = []
    
    for seg in card.get("segments", []):
        if seg["label"] in ["intro", "build"]:
            mix_in.append({"beat_start": seg["beat_start"], "beat_end": seg["beat_end"], "reason": seg["label"]})
        if seg["label"] in ["outro", "tail"]:
            mix_out.append({"beat_start": seg["beat_start"], "beat_end": seg["beat_end"], "reason": seg["label"]})
            
    card["mixable_zones"] = {"mix_in": mix_in, "mix_out": mix_out}
    
    return card

if __name__ == "__main__":
    crate_path = "C:/Projects/Cloudy DJ 2.0/library/crate.json"
    with open(crate_path, "r") as f:
        crate = json.load(f)
        
    # Test on one track
    track = "Syn Cole - Feel Good"
    card = generate_song_card(track, crate[track])
    
    os.makedirs("C:/Projects/Cloudy DJ 2.0/library/song_cards", exist_ok=True)
    out_path = f"C:/Projects/Cloudy DJ 2.0/library/song_cards/{track}.json"
    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)
        
    print(f"Saved Song Card to {out_path}")
