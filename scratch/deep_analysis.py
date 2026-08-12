"""
deep_analysis.py - Complete analysis of both tracks' stem structure,
drop detection accuracy, and vocal entry points.
Generates multiple annotated images.
"""
import librosa
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import uniform_filter1d

OUTPUT_DIR = "C:/Users/sriha/.gemini/antigravity/brain/c59eb038-aea4-4134-804c-7714de9f666f/scratch"

with open('library/crate.json') as f:
    crate = json.load(f)

def analyze_track_full(track_name):
    full_key = next(k for k in crate if track_name in k)
    data = crate[full_key]
    bpm = data['bpm']
    spb_s = 60.0 / bpm
    sr = 44100
    spb = int(spb_s * sr)
    drop_override = data.get('drop_idx')

    print(f"\n{'='*60}")
    print(f"TRACK: {track_name} | BPM: {bpm} | drop_idx override: {drop_override}")
    print(f"{'='*60}")

    stems = {}
    for name in ['bass', 'drums', 'other', 'vocals']:
        stems[name], _ = librosa.load(data['stems'][name], sr=sr)

    n_beats = int(len(stems['bass']) / spb)

    # Compute per-beat RMS for all stems
    beat_rms = {name: [] for name in stems}
    for beat in range(n_beats):
        s = beat * spb
        e = s + spb
        for name in stems:
            chunk = stems[name][s:e]
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
            beat_rms[name].append(rms)

    beats_arr = np.arange(n_beats)
    times_arr = beats_arr * spb_s

    # -----------------------------------------------------------------------
    # Figure 1: All 4 stems full track energy with annotations
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(24, 14), sharex=True)
    colors = {'bass': '#FF6B6B', 'drums': '#4ECDC4', 'other': '#45B7D1', 'vocals': '#96CEB4'}
    
    for i, name in enumerate(['bass', 'drums', 'other', 'vocals']):
        ax = axes[i]
        rms = np.array(beat_rms[name])
        ax.fill_between(times_arr, 0, rms, color=colors[name], alpha=0.7)
        ax.plot(times_arr, rms, color=colors[name], linewidth=0.8)
        ax.set_ylabel(f'{name.upper()}\nRMS', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(rms) * 1.2 + 1e-6)
        
        # Mark the crate override drop
        if drop_override is not None:
            drop_time = drop_override * spb_s
            ax.axvline(drop_time, color='red', linewidth=2, linestyle='--', label=f'drop_idx={drop_override}')
        
        # Detect and annotate real bass drops (for bass stem)
        if name == 'bass':
            for beat in range(1, len(rms)):
                if rms[beat] - rms[beat-1] > 0.05:
                    ax.axvline(times_arr[beat], color='orange', linewidth=1.5, linestyle=':', alpha=0.8)
                    ax.annotate(f'B{beat}', xy=(times_arr[beat], rms[beat]),
                                fontsize=7, color='orange', rotation=90)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    fig.suptitle(f'{track_name} — Full Track Stem Energy Analysis\n(BPM: {bpm}, drop_idx override: {drop_override})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{track_name}_deep_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Zoomed view around the drop +/- 30 beats
    # -----------------------------------------------------------------------
    if drop_override is not None:
        zoom_start = max(0, drop_override - 30)
        zoom_end = min(n_beats, drop_override + 50)
        zoom_beats = beats_arr[zoom_start:zoom_end]
        zoom_times = zoom_beats * spb_s

        fig, axes = plt.subplots(4, 1, figsize=(20, 12), sharex=True)
        for i, name in enumerate(['bass', 'drums', 'other', 'vocals']):
            ax = axes[i]
            rms = np.array(beat_rms[name][zoom_start:zoom_end])
            ax.fill_between(zoom_times, 0, rms, color=colors[name], alpha=0.7)
            ax.plot(zoom_times, rms, color=colors[name], linewidth=1.2, marker='o', markersize=3)
            
            # Annotate each beat number
            for j, (t, r) in enumerate(zip(zoom_times, rms)):
                if r > 0.01:
                    ax.annotate(str(zoom_start + j), xy=(t, r), fontsize=6, ha='center', va='bottom')
            
            ax.set_ylabel(f'{name.upper()}\nRMS', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            drop_time = drop_override * spb_s
            ax.axvline(drop_time, color='red', linewidth=2.5, linestyle='--')
            ax.axvspan(drop_time, drop_time + 32 * spb_s, alpha=0.1, color='red', label='32-beat blend zone')
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
        fig.suptitle(f'{track_name} — Zoom: Drop ± 30 beats (drop_idx={drop_override})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = f"{OUTPUT_DIR}/{track_name}_drop_zoom.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Figure 3: Drop detector simulation — what does detect_first_drop SEE?
    # -----------------------------------------------------------------------
    hop = 1024
    rms_bass = librosa.feature.rms(y=stems['bass'], frame_length=2048, hop_length=hop)[0]
    smooth_len = int(spb_s * sr / hop)
    if smooth_len > 0:
        rms_smooth = uniform_filter1d(rms_bass, size=smooth_len)
    else:
        rms_smooth = rms_bass
    
    frames_per_sec = sr / hop
    frame_times = librosa.frames_to_time(np.arange(len(rms_smooth)), sr=sr, hop_length=hop)
    
    # Simulate the contrast scan
    window = smooth_len * 32
    start_frame = int(30 * frames_per_sec)
    end_frame = int(120 * frames_per_sec)
    contrast_arr = np.zeros(len(rms_smooth))
    mean_bass = np.mean(rms_smooth) + 1e-9
    
    best_frame, max_contrast = 0, 0
    for frame_idx in range(max(window, start_frame), min(end_frame, len(rms_smooth) - window)):
        pre_mean = np.mean(rms_smooth[max(0, frame_idx - window):frame_idx])
        post_mean = np.mean(rms_smooth[frame_idx:frame_idx + window])
        contrast = post_mean - pre_mean
        contrast_arr[frame_idx] = contrast
        if contrast > max_contrast and post_mean > 0.03:
            max_contrast = contrast
            best_frame = frame_idx
    
    detected_time = librosa.frames_to_time(best_frame, sr=sr, hop_length=hop)
    confidence = max_contrast / mean_bass
    
    print(f"  detect_first_drop → time={detected_time:.1f}s, confidence={confidence:.2f}")
    if drop_override is not None:
        expected_time = drop_override * spb_s
        print(f"  Expected (override): {expected_time:.1f}s | Error: {abs(detected_time - expected_time):.1f}s")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 10), sharex=False)
    
    # Top: raw and smoothed bass RMS
    ax1.plot(frame_times, rms_bass, alpha=0.4, color='#FF6B6B', linewidth=0.5, label='Raw bass RMS')
    ax1.plot(frame_times, rms_smooth, color='#FF6B6B', linewidth=2, label='Smoothed bass RMS')
    ax1.axvline(detected_time, color='blue', linewidth=2, linestyle='--', label=f'Detected drop: {detected_time:.1f}s')
    if drop_override is not None:
        ax1.axvline(drop_override * spb_s, color='green', linewidth=2, linestyle='--',
                    label=f'Override drop: {drop_override * spb_s:.1f}s (beat {drop_override})')
    ax1.axvspan(30, 120, alpha=0.1, color='yellow', label='Search window (30–120s)')
    ax1.set_ylabel('Bass RMS', fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(frame_times))
    
    # Bottom: contrast signal
    ax2.plot(frame_times[:len(contrast_arr)], contrast_arr, color='purple', linewidth=1.5, label='Contrast (post_mean - pre_mean)')
    ax2.axvline(detected_time, color='blue', linewidth=2, linestyle='--')
    if drop_override is not None:
        ax2.axvline(drop_override * spb_s, color='green', linewidth=2, linestyle='--')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_ylabel('Contrast Score', fontsize=10)
    ax2.set_xlabel('Time (seconds)', fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(frame_times))

    fig.suptitle(f'{track_name} — Drop Detector Internals\n'
                 f'Detected={detected_time:.1f}s (confidence={confidence:.2f}) | '
                 f'Override={drop_override if drop_override else "None"}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = f"{OUTPUT_DIR}/{track_name}_drop_detector.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    bass_arr = np.array(beat_rms['bass'])
    drums_arr = np.array(beat_rms['drums'])
    print(f"\n  BASS: max={bass_arr.max():.4f} at beat {bass_arr.argmax()}, "
          f"active beats (>0.05): {(bass_arr>0.05).sum()}")
    print(f"  DRUMS: max={drums_arr.max():.4f} at beat {drums_arr.argmax()}")
    
    # Find where bass actually enters strongly
    bass_entries = []
    prev = 0
    for i in range(1, len(bass_arr)):
        if bass_arr[i] - prev > 0.05 and bass_arr[i] > 0.05:
            bass_entries.append((i, bass_arr[i]))
        prev = bass_arr[i]
    
    print(f"  Real bass entry points (jump >0.05): {[(b, f'{r:.3f}') for b,r in bass_entries[:10]]}")
    
    return {
        'track_name': track_name,
        'bpm': bpm,
        'drop_override': drop_override,
        'detected_drop_time': detected_time,
        'detected_confidence': confidence,
        'bass_entries': bass_entries,
        'beat_rms': beat_rms,
    }

results = {}
for name in ['James_Hype_Wild', 'Tiesto_Secrets']:
    results[name] = analyze_track_full(name)

# -----------------------------------------------------------------------
# Figure 4: Side-by-side comparison at transition point
# -----------------------------------------------------------------------
print("\n=== TRANSITION ANALYSIS ===")
jh = results['James_Hype_Wild']
ts = results['Tiesto_Secrets']

# Transition 1: Tiesto A → James Hype B at James Hype beat 125
jh_drop = 125
ts_drop = 224
jh_bpm = jh['bpm']  # 126
ts_bpm = ts['bpm']  # 128
jh_spb = 60.0 / jh_bpm
ts_spb = 60.0 / ts_bpm

# What is James Hype's bass doing at beat 125?
jh_bass = np.array(jh['beat_rms']['bass'])
ts_bass = np.array(ts['beat_rms']['bass'])

print(f"\nTransition 1 (Tiesto → James Hype) swap at JH beat 125:")
print(f"  JH bass at beats 120-140:")
for b in range(120, min(141, len(jh_bass))):
    flag = ' <<< DROP CONTEXT' if b == 125 else ''
    print(f"    Beat {b}: bass={jh_bass[b]:.4f}{flag}")

print(f"\nTransition 2 (James Hype → Tiesto) swap at Tiesto beat 224:")
print(f"  Tiesto bass at beats 220-240:")
for b in range(220, min(241, len(ts_bass))):
    flag = ' <<< DROP CONTEXT' if b == 224 else ''
    print(f"    Beat {b}: bass={ts_bass[b]:.4f}{flag}")

print("\nDone! Check scratch/ for analysis images.")
