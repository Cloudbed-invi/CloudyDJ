import re
import os

filepath = r"c:/Users/sriha/Documents/Cloudy DJ 2.0/src/engine/generate_all_transitions.py"
with open(filepath, 'r') as f:
    content = f.read()

new_func = r'''def generate_bass_swap_transition(track_a_str, track_b_str, out_name):
    """
    Drop-Point Stem Swap:
      • A plays at 100% energy up to the EXACT drop.
      • B is 100% muted before the drop (no mud).
      • At the exact drop: A's bass stops, B's bass comes in FULL.
      • A's other stems (drums, synths) overlap and fade out slowly over 32 beats.
      • B's other stems (drums, synths) fade in slowly over 32 beats.
      • A's vocals use a "Baton Pass" - they are cut right before B's first vocal entry.
    """
    print(f"\n[Bass Swap] {track_a_str} -> {track_b_str}")
    with open(CRATE_FILE, 'r') as f:
        crate = json.load(f)

    a_orig = get_track_data(crate, track_a_str)
    b      = get_track_data(crate, track_b_str)
    a      = apply_warping(a_orig, b["bpm"])
    sr     = a["sr"]
    spb    = int((60.0 / b["bpm"]) * sr)

    # EXACT DROP POINTS
    a_drop = _safe_beat(a["beats"], a["drop_idx"])
    b_drop = _safe_beat(b["beats"], b["drop_idx"])
    
    a_swap = a_drop
    b_swap = b_drop

    # ------------------------------------------------------------------
    # Whisper vocal boundaries - BATON PASS
    # ------------------------------------------------------------------
    # B: find first vocal onset after B's swap point
    b_vocal_entry = dsp_utils.find_vocal_entry(b["vocals"], b_swap, sr)
    b_entry_offset = b_vocal_entry - b_swap # How far after the drop does B sing?
    
    # A: cut vocals right before B sings (minus 1 beat for breathing room)
    target_cut_point = a_swap + b_entry_offset - (1 * spb)
    a_vocal_cut = dsp_utils.find_vocal_cutoff_in_buildup(a, a_swap, max(a_swap + 8 * spb, target_cut_point), sr)
    # Ensure we don't cut later than B's actual entry!
    if (a_vocal_cut - a_swap) > b_entry_offset:
        a_vocal_cut = a_swap + max(0, b_entry_offset - int(0.5 * spb)) # hard cap it

    print(f"  B vocal entry: {b_entry_offset/sr:.2f}s after drop")
    print(f"  A vocal cut: {(a_vocal_cut - a_swap)/sr:.2f}s after drop")

    pre_len   = int(10 * sr) # 10s of build-up
    blend_len = 32 * spb     # 32 beats of secret crossfade post-drop
    post_len  = int(10 * sr) # 10s of B solo
    total_len = pre_len + blend_len + post_len

    out = np.zeros((total_len, 2), dtype=np.float32)

    # ------------------------------------------------------------------
    # Pre-Drop: A at full energy (B is 100% muted)
    # ------------------------------------------------------------------
    a_pre_start = max(0, a_swap - pre_len)
    actual_pre  = a_swap - a_pre_start
    out_pre_off = pre_len - actual_pre
    
    if actual_pre > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            wl = _write_len(out, out_pre_off, a[stem], a_pre_start, a_swap)
            out[out_pre_off:out_pre_off + wl] += a[stem][a_pre_start:a_pre_start + wl]

    # White noise impact at the exact drop to mask the bass switch
    noise_len = int(2.0 * spb)
    noise_mono = dsp_utils.generate_white_noise_sweep(noise_len, sr)
    # We want an impact, so reverse the sweep to make it a decay instead of a rise
    noise_mono = noise_mono[::-1] * 0.4
    noise_stereo = np.stack([noise_mono, noise_mono], axis=1)
    
    noise_end = min(pre_len + noise_len, total_len)
    nw = noise_end - pre_len
    if nw > 0:
        out[pre_len:noise_end] += noise_stereo[:nw]

    # ------------------------------------------------------------------
    # Post-Drop Blend Zone: 32 beats of silent morphing
    # ------------------------------------------------------------------
    n = blend_len
    t = np.linspace(0.0, np.pi / 2, n, dtype=np.float32)
    fade_out = np.cos(t)[:, None]  
    fade_in  = np.sin(t)[:, None]  

    # LUFS gain match for B
    a_measure = a["bass"][max(0, a_swap - 4 * spb):a_swap] + \
                a["drums"][max(0, a_swap - 4 * spb):a_swap]
    b_measure = b["bass"][b_swap:min(b_swap + 4 * spb, len(b["bass"]))] + \
                b["drums"][b_swap:min(b_swap + 4 * spb, len(b["drums"]))]
    gain = _lufs_gain(a_measure, b_measure, sr)

    # 1. B Bass: FULL VOLUME IMMEDIATELY at the drop
    b_bass_end = min(b_swap + blend_len, len(b["bass"]))
    bbl = b_bass_end - b_swap
    if bbl > 0:
        out[pre_len:pre_len + bbl] += b["bass"][b_swap:b_bass_end] * gain

    # 2. A Mids/Highs: Fade out silently over 32 beats (Bass is 0)
    for stem in ["drums", "other"]:
        src_end = min(a_swap + blend_len, len(a[stem]))
        wl = src_end - a_swap
        if wl > 0:
            out[pre_len:pre_len + wl] += a[stem][a_swap:src_end] * fade_out[:wl]

    # 3. B Mids/Highs: Fade in silently over 32 beats
    for stem in ["drums", "other"]:
        src_end = min(b_swap + blend_len, len(b[stem]))
        wl = src_end - b_swap
        if wl > 0:
            out[pre_len:pre_len + wl] += b[stem][b_swap:src_end] * fade_in[:wl] * gain

    # 4. A Vocals: Play at full volume until a_vocal_cut, then echo out
    if a_vocal_cut > a_swap:
        wl = a_vocal_cut - a_swap
        # They play full volume
        out[pre_len:pre_len + wl] += a["vocals"][a_swap:a_vocal_cut]
        
        # Add echo tail right after the cut
        vocal_snip = a["vocals"][a_vocal_cut - spb:a_vocal_cut]
        echo_tail = dsp_utils.generate_echo_tail(vocal_snip, sr, b["bpm"], beats=4)
        if echo_tail.ndim == 1:
            echo_stereo = np.stack([echo_tail, echo_tail], axis=1)
        else:
            echo_stereo = echo_tail
            
        out_vocal_cut = pre_len + wl
        tail_len = min(len(echo_stereo), total_len - out_vocal_cut)
        if tail_len > 0 and out_vocal_cut >= 0:
            out[out_vocal_cut:out_vocal_cut + tail_len] += echo_stereo[:tail_len] * 0.7

    # 5. B Vocals: Enter when they enter, at full volume
    b_vocal_delay = b_vocal_entry - b_swap
    if b_vocal_delay < blend_len:
        src_end = min(b_swap + blend_len, len(b["vocals"]))
        wl = src_end - b_vocal_entry
        if wl > 0:
            out_start = pre_len + b_vocal_delay
            # We fade them in quickly so they don't click, then full volume
            v_fade_len = min(spb, wl)
            v_fade = np.concatenate([
                np.linspace(0.0, 1.0, v_fade_len, dtype=np.float32),
                np.ones(wl - v_fade_len, dtype=np.float32)
            ])[:, None]
            out[out_start:out_start + wl] += b["vocals"][b_vocal_entry:src_end] * v_fade * gain

    # ------------------------------------------------------------------
    # Post-Blend: B Solo (everything at 1.0 gain)
    # ------------------------------------------------------------------
    b_post_start = b_swap + blend_len
    b_post_end   = min(len(b["drums"]), b_post_start + post_len)
    wl_post      = b_post_end - b_post_start
    out_post_off = pre_len + blend_len
    if wl_post > 0:
        for stem in ["drums", "bass", "other", "vocals"]:
            # Protect against differing stem lengths
            stem_len = len(b[stem])
            end = min(stem_len, b_post_start + wl_post)
            w = end - b_post_start
            if w > 0:
                out[out_post_off:out_post_off + w] += b[stem][b_post_start:end] * gain

    out = dsp_utils.apply_limiter(out, threshold=0.9)
    print(f"  Saved -> {out_name}")
    sf.write(out_name, out, sr)
'''

pattern = re.compile(r'def generate_bass_swap_transition.*?def generate_treble_swap_transition', re.DOTALL)
content = pattern.sub(new_func + '\n\ndef generate_treble_swap_transition', content)

with open(filepath, 'w') as f:
    f.write(content)
