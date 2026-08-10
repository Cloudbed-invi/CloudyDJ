import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy import signal
import whisper_timestamped as whisper


def generate_echo_tail(audio_chunk, sr, bpm, beats=4):
    delay_samples = int((60.0 / bpm) * sr)
    tail_length = delay_samples * beats
    out = np.zeros(len(audio_chunk) + tail_length, dtype=np.float32)
    out[:len(audio_chunk)] = audio_chunk
    decay = 0.6
    for b in range(1, beats + 1):
        start = b * delay_samples
        end = start + len(audio_chunk)
        if end <= len(out):
            out[start:end] += audio_chunk * (decay ** b)
    return out[len(audio_chunk):]


def generate_white_noise_sweep(length_samples, sr):
    noise = np.random.normal(0, 1, length_samples).astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, length_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, length_samples, dtype=np.float32)
    envelope = fade_in * fade_out * 4.0
    nyq = 0.5 * sr
    out_noise = np.zeros_like(noise)
    chunk_size = 2048
    for i in range(0, len(noise), chunk_size):
        end = min(i + chunk_size, len(noise))
        progress = i / max(1, len(noise))
        cutoff = min(400 + progress * 8000, nyq * 0.99)
        b, a = signal.butter(2, cutoff / nyq, btype='low')
        if end - i > 12:
            out_noise[i:end] = signal.filtfilt(b, a, noise[i:end])
    return (out_noise * envelope * 0.12).astype(np.float32)


def apply_bandpass(audio, sr, low=300, high=4000):
    nyq = 0.5 * sr
    b, a = signal.butter(2, [low / nyq, high / nyq], btype='band')
    if len(audio) > 12:
        return signal.filtfilt(b, a, audio).astype(np.float32)
    return audio


def apply_sidechain_duck(instrumental, vocal, sr, hop=512):
    vocal_rms = librosa.feature.rms(y=vocal, frame_length=2048, hop_length=hop)[0]
    if np.max(vocal_rms) > 0:
        vocal_rms = vocal_rms / np.max(vocal_rms)
    duck_gain = np.ones(len(instrumental), dtype=np.float32)
    for i, r in enumerate(vocal_rms):
        start = i * hop
        end = min(start + hop, len(instrumental))
        if r > 0.05:
            duck_gain[start:end] = 0.3
    smooth_samples = int(0.01 * sr)
    duck_gain = uniform_filter1d(duck_gain, size=smooth_samples)
    return instrumental * duck_gain


def apply_hpf_sweep(audio, sr, start_freq=2000, end_freq=80, num_chunks=32):
    """
    Apply a high-pass filter that gradually sweeps from start_freq down to end_freq.
    Useful for 'revealing' a track (like vocals) smoothly.
    """
    if len(audio) == 0:
        return audio
        
    out = np.zeros_like(audio)
    chunk_size = len(audio) // num_chunks
    
    # Logarithmic sweep sounds more natural for frequencies
    freqs = np.logspace(np.log10(start_freq), np.log10(end_freq), num_chunks)
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(audio)
        
        chunk = audio[start_idx:end_idx]
        freq = freqs[i]
        
        # Apply butterworth HPF
        nyq = 0.5 * sr
        normal_cutoff = freq / nyq
        
        if normal_cutoff < 1.0 and normal_cutoff > 0:
            b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)
            filtered_chunk = signal.filtfilt(b, a, chunk)
            out[start_idx:end_idx] = filtered_chunk
        else:
            out[start_idx:end_idx] = chunk
            
    return out

def find_best_loop_source(track_data, drop_sample, bpm, sr):
    """
    Use Whisper to find the last cleanly-bounded WORD before the drop.
    Reject any word that's longer than 1 beat (Whisper hallucination).
    Fall back to onset detection if no clean word is found.
    """
    vocal_stem = track_data["vocals"]
    samples_per_beat = int((60.0 / bpm) * sr)
    max_word_duration = 60.0 / bpm  # reject words longer than 1 beat
    
    # 1. Focus search ONLY on the 16 beats before the drop to find a massive word
    search_start = max(0, drop_sample - (16 * samples_per_beat))
    search_end = drop_sample
    
    search_region_audio = vocal_stem[search_start:search_end]

    try:
        whisper_sr = 16000
        search_region_16k = librosa.resample(search_region_audio, orig_sr=sr, target_sr=whisper_sr)
        import whisper
        model = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, search_region_16k, language="en", word_timestamps=True)

        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append(word)

        # Filter: accept words
        valid_words = []
        for w in words:
            start_s = w["start"]
            end_s = w["end"]
            w_start_sample = search_start + int(start_s * sr)
            w_end_sample = search_start + int(end_s * sr)
            duration_s = end_s - start_s
            
            # Reject Whisper hallucination (word > 1 beat) or micro-clicks (< 0.1s)
            if duration_s >= 0.1 and duration_s <= max_word_duration and w_end_sample <= drop_sample:
                valid_words.append((w, w_start_sample, w_end_sample, duration_s))

        if valid_words:
            # Sort valid words by duration descending, pick the longest one to give us a good vowel tail to loop
            valid_words.sort(key=lambda x: x[3], reverse=True)
            best_word_data = valid_words[0]
            best_word, w_start_sample, w_end_sample, duration_s = best_word_data
            print(f"Whisper picked word: '{best_word['word'].strip()}' at {best_word['start']:.2f}s - {best_word['end']:.2f}s (duration: {duration_s:.2f}s)")
            return w_start_sample, w_end_sample
        else:
            print(f"Whisper found {len(words)} words but none met criteria in 16-beat region.")
    except Exception as e:
        print(f"Whisper failed: {e}")

    # Fallback: onset detection if Whisper fails or finds nothing
    print("Falling back to onset detection...")
    onsets = librosa.onset.onset_detect(
        y=search_region_audio, sr=sr, units='samples', backtrack=True
    )
    if len(onsets) == 0:
        return search_start, search_start + samples_per_beat

    # Use the LAST onset as the loop start, duration = 1 beat (max length to prevent clicking)
    onset_sample = search_start + onsets[-1]
    return onset_sample, onset_sample + samples_per_beat


def find_instrumental_intro(track_data, sr):
    """
    Find the start of B's intro that has enough musical energy (so we don't blend into silence).
    Prefer a segment labeled 'intro' that has SOME energy but no heavy vocals.
    Falls back to 32 beats before the drop.
    """
    drop_idx = track_data.get("drop_idx", 0)
    beats = track_data.get("beats", np.array([]))
    bpm = track_data.get("bpm", 120)
    spb = int((60.0 / bpm) * sr)

    # First try labeled intro segments with minimal vocals but some instrumental presence
    for s in track_data.get("segments", []):
        if "intro" in s["label"].lower():
            start = s["start_sample"]
            end = min(s["end_sample"], start + int(20 * sr))
            if end > start + sr:  # must be at least 1s
                vocal_chunk = track_data["vocals"][start:end]
                inst_chunk = track_data["drums"][start:end] + track_data["other"][start:end]
                vocal_rms = np.mean(librosa.feature.rms(y=vocal_chunk)) if len(vocal_chunk) > 0 else 0
                inst_rms = np.mean(librosa.feature.rms(y=inst_chunk)) if len(inst_chunk) > 0 else 0
                # Good intro: has some instrumental energy and not too many vocals
                if inst_rms > 0.005 and vocal_rms < 0.02:
                    print(f"Found instrumental intro at sample {start} (inst_rms={inst_rms:.4f}, vocal_rms={vocal_rms:.4f})")
                    return start

    # Fallback: 32 beats before the drop (always has musical energy in EDM)
    if len(beats) > 0 and drop_idx > 0:
        fallback_idx = max(0, drop_idx - 32)
        fallback_start = int(beats[fallback_idx])
        print(f"No clean intro found. Falling back to 32 beats before drop: sample {fallback_start}")
        return fallback_start

    return 0


def find_vocal_cutoff_in_buildup(track_data, build_start, drop_sample, sr):
    """
    1. Scan the last 8 beats of the buildup using Whisper.
    2. Find the last complete word.
    3. Snap the cutoff to the end of that word (snapped to the nearest beat).
    """
    if build_start >= drop_sample:
        return drop_sample
        
    beats = track_data["beats"]
    spb = int((60.0 / track_data["bpm"]) * sr)

    # Search region: 8 beats before the drop
    search_start = max(build_start, drop_sample - 8 * spb)
    search_end = drop_sample
    
    search_region_audio = track_data["vocals"][search_start:search_end]
    
    try:
        whisper_sr = 16000
        search_region_16k = librosa.resample(search_region_audio, orig_sr=sr, target_sr=whisper_sr)
        import whisper
        model = whisper.load_model("tiny", device="cpu")
        result = whisper.transcribe(model, search_region_16k, language="en", word_timestamps=True)

        words = []
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                words.append(word)

        # We want the last word that ends before the drop
        valid_words = []
        for w in words:
            start_s = w["start"]
            end_s = w["end"]
            w_start_sample = search_start + int(start_s * sr)
            w_end_sample = search_start + int(end_s * sr)
            
            # Allow some margin (e.g. 1 beat before drop) so we don't cut too close
            if w_end_sample <= drop_sample - spb:
                valid_words.append((w, w_start_sample, w_end_sample))
                
        if valid_words:
            # Pick the last valid word
            last_word, w_start, w_end = valid_words[-1]
            print(f"Vocal cutoff: Whisper found word '{last_word['word'].strip()}' ending at {w_end}")
            
            # Snap to nearest beat
            if len(beats) > 0:
                nearest_beat_idx = np.argmin(np.abs(beats - w_end))
                snapped = int(beats[nearest_beat_idx])
                snapped = min(snapped, drop_sample - spb)
                snapped = max(snapped, build_start)
                print(f"Vocal cutoff snapped to beat at sample {snapped} (raw word end {w_end})")
                return snapped
            return w_end
            
    except Exception as e:
        print(f"Whisper vocal cutoff failed: {e}")

    # Fallback: if Whisper fails, just cut 2 beats before the drop
    fallback = max(build_start, drop_sample - 2 * spb)
    print(f"Vocal cutoff fallback: cutting 2 beats before drop at sample {fallback}")
    return fallback
