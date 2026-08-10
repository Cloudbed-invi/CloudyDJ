import numpy as np
import librosa
from scipy import signal


def generate_loop_roll(audio_stem, bpm, sr, drop_sample_idx, num_beats=8,
                       exact_word_start=None, exact_word_end=None):
    from src.engine.dsp_utils import find_best_loop_source

    samples_per_beat = int((60.0 / bpm) * sr)

    # Get the source word/sound to loop
    if exact_word_start is not None and exact_word_end is not None:
        source = audio_stem[exact_word_start:exact_word_end].copy()
    else:
        loop_start, loop_end = find_best_loop_source(audio_stem, drop_sample_idx, bpm, sr)
        source = audio_stem[loop_start:loop_end].copy()

    # Safety: if source is silent, grab a beat before the drop
    if len(source) == 0 or np.max(np.abs(source)) < 0.001:
        fallback = max(0, drop_sample_idx - samples_per_beat)
        source = audio_stem[fallback:drop_sample_idx].copy()

    # Crossfade loop boundaries to eliminate clicks
    def make_loop_chunk(src, target_len):
        """Tile the source to fill target_len, using zero-crossing snap."""
        if len(src) == 0:
            return np.zeros(target_len, dtype=np.float32)
        
        # Find nearest zero-crossing at the start to prevent clicks without adding silence
        search_range = min(512, len(src) // 4)
        zero_crossings = np.where(np.diff(np.signbit(src[:search_range])))[0]
        if len(zero_crossings) > 0:
            src = src[zero_crossings[0]:]
            
        # Trim to target length if longer
        src = src[:target_len].copy()
        
        # Pad by tiling if shorter
        if len(src) < target_len:
            repeats = (target_len // len(src)) + 2
            src = np.tile(src, repeats)[:target_len]
        
        # Small 2ms crossfade just at the very end to prevent hard cut click when chunk repeats
        cf_len = min(int(0.002 * sr), len(src) // 4)
        if cf_len > 0:
            fade_out = np.linspace(1.0, 0.0, cf_len, dtype=np.float32)
            src[-cf_len:] *= fade_out
            
        return src.astype(np.float32)

    sections = []

    word_len = len(source)
    min_loop_samples = max(int(0.05 * sr), samples_per_beat // 4)  # min 50ms or 1/16th note
    
    # Phase 1: 2 beats — loop the full word once per beat
    for _ in range(2):
        sections.append(make_loop_chunk(source, samples_per_beat))

    # Phase 2: 2 beats — loop back half of word (vowel tail), 2x per beat
    half_src = source[word_len // 2:]
    for _ in range(2):
        sections.append(make_loop_chunk(half_src, samples_per_beat))

    # Phase 3: 2 beats — loop quarter of word (vowel tail), 4x per beat
    qtr_len = max(min_loop_samples, word_len // 4)
    qtr_src = source[-qtr_len:]
    for _ in range(2):
        sections.append(make_loop_chunk(qtr_src, samples_per_beat))

    # Phase 4: 2 beats — loop eighth of word (vowel tail), 8x per beat
    eighth_len = max(min_loop_samples, word_len // 8)
    eighth_src = source[-eighth_len:]
    for _ in range(2):
        sections.append(make_loop_chunk(eighth_src, samples_per_beat))

    full_buildup = np.concatenate(sections).astype(np.float32)

    # Trim/pad to exactly num_beats
    target_len = num_beats * samples_per_beat
    if len(full_buildup) > target_len:
        full_buildup = full_buildup[:target_len]
    elif len(full_buildup) < target_len:
        full_buildup = np.pad(full_buildup, (0, target_len - len(full_buildup)))

    # Apply crescendo (volume automation) to build tension: 0.7x to 1.3x
    crescendo = np.linspace(0.7, 1.3, target_len, dtype=np.float32)
    full_buildup *= crescendo

    # NOTE: We do NOT apply a HPF to the voice here.
    # The HPF only applies to bass/drums in the transition context.
    return full_buildup
