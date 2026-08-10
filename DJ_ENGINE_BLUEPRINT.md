# Cloudy DJ 2.0 — Complete Engine Blueprint

> Everything we have, everything that's broken, every technique a real DJ uses, what Tomorrowland-level requires beyond basic, and the exact build order to get there.

---

## Part 1: Current Engine Status

### ✅ What Works

| Component | How It Works | File |
|---|---|---|
| Track Downloading | yt-dlp search & download | [fetch_tracks.py](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/fetch_tracks.py) |
| Stem Separation | Demucs htdemucs → vocals, drums, bass, other | [fetch_tracks.py](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/fetch_tracks.py) |
| BPM Detection | librosa beat_track | [generate_batch.py](file:///C:/Projects/Cloudy%20DJ%202.0/generate_batch.py) |
| Key Detection | Chroma CQT → pitch class | [generate_batch.py](file:///C:/Projects/Cloudy%20DJ%202.0/generate_batch.py) |
| Beat Grid | librosa beat_track in samples | [generate_batch.py](file:///C:/Projects/Cloudy%20DJ%202.0/generate_batch.py) |
| Drop Detection | Bass RMS contrast (30s–120s window) | [generate_batch.py](file:///C:/Projects/Cloudy%20DJ%202.0/generate_batch.py) |
| Time Stretching | librosa time_stretch per-stem | [generate_all_transitions.py](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/generate_all_transitions.py) |
| 3-Band EQ Filtering | scipy butter (low/mid/high split) | [apply_technique.py](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/apply_technique.py) |
| DJ Mix Reverse-Engineering | Cross-correlation to extract EQ envelopes from real mixes | [extract_all_techniques.py](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/extract_all_techniques.py) |
| Echo Out Delay Line | BPM-synced feedback delay (0.65 decay) | [mixer.py L24-39](file:///C:/Projects/Cloudy%20DJ%202.0/mixer.py#L24-L39) |
| Crate Management | JSON metadata store | `library/crate.json` |

### ❌ What's Broken (and the exact line causing it)

| Technique | Symptom | Root Cause | Broken Code |
|---|---|---|---|
| **1. Outro→Intro Blend** | Silent gap, ghost audio | No song structure — uses file length as "outro" | [L160-161](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/generate_all_transitions.py#L160-L161): `target_a_end = len(a["drums"]) - blend_length` |
| **3. Drop Swap** | Robotic vocal chop | Hard RMS cut, no echo/reverb tail | [L97-105](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/generate_all_transitions.py#L97-L105): `find_vocal_cut_point()` walks backward to first silence dip |
| **4. Vocal Blend** | Vocals shadowed after drop | No sidechain ducking — vocal simply added to loud instrumental | [L274](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/generate_all_transitions.py#L274): `out_audio += b["drums"] + b["bass"] + b["other"]` |
| **5. Loop Roll** | Loops garbage, leak before drop | Blind offset grab (no onset detection), no Track B mask | [loop_roll.py L50-54](file:///C:/Projects/Cloudy%20DJ%202.0/src/engine/loop_roll.py#L50-L54): `source_beat = audio_stem[start_sample : start_sample + samples_per_beat]` |

### ⚠️ Hidden Win Already Built
`generate_echo_tail()` in [mixer.py L24-39](file:///C:/Projects/Cloudy%20DJ%202.0/mixer.py#L24-L39) is a proper BPM-synced delay line with 0.65 feedback. It was never ported to the new engine. Copy-pasting it fixes techniques 3 and 5 instantly.

---

## Part 2: The Complete DJ Technique Library

### Are 5 techniques enough for basic?

**Yes — IF they work properly.** The 5 core techniques below cover every scenario a basic DJ faces. The DJ you visited at the club was using exactly these 5, just rotating between them every minute. The problem isn't that we need more techniques — it's that 3 out of 5 are currently broken.

### The 5 Core Techniques (Basic DJ Level)

#### T1. Bass Swap (The "Every Minute" Transition)
**When:** Anytime. Mid-song, mid-drop, anywhere.
**Duration:** 4-16 beats (instant feel).
**How a real DJ does it:**
1. Track B is already playing silently underneath with bass EQ at zero
2. On the downbeat of a phrase (beat 1 of a 16-beat block), the DJ simultaneously:
   - Kills Track A's bass knob → 0
   - Brings Track B's bass knob → full
3. Over the next 8-16 beats, crossfades mids and highs
4. Track A's volume fader goes to zero. Done.

**Why it's the #1 technique:** It works on ANY two tracks at ANY point in the song. It never sounds bad because only one bass plays at a time. This is what your DJ was doing every minute.

**AI Implementation:**
```
Split both tracks into Low / Mid / High bands
At phrase boundary:
  A_low:  1.0 → 0.0 in 2 beats (exponential)
  B_low:  0.0 → 1.0 in 2 beats (exponential)
  A_mid:  1.0 → 0.0 in 16 beats (linear)
  B_mid:  0.0 → 1.0 in 16 beats (linear)
  A_high: 1.0 → 0.0 in 16 beats (linear)
  B_high: 0.0 → 1.0 in 16 beats (linear)
  A_vocal: fade out over 8 beats (or echo out)
  B_vocal: fade in after A_vocal ends
```

---

#### T2. Long EQ Blend (Outro → Intro)
**When:** Track A is winding down, Track B is just starting.
**Duration:** 32-64 beats (long and smooth).
**How a real DJ does it:**
1. Identifies Track A's **Outro** (stripped-down drums, no vocals)
2. Identifies Track B's **Intro** (also stripped-down drums, no vocals)
3. Starts Track B's intro playing underneath Track A's outro
4. Slowly crossfades bass, then mids, then highs over 32 beats
5. Key rule: **Never overlap two vocals.** Outros and intros are chosen specifically because they have no singing.

**AI requirement:** Song Structure Segmentation to find real Outro/Intro boundaries.

---

#### T3. Drop Swap + Echo Out
**When:** Maximum energy moment. Track A is building up to a drop.
**Duration:** Instant (the swap itself is 0 beats).
**How a real DJ does it:**
1. Track A is in its buildup (drums getting faster, bass swept out)
2. Right at the drop, DJ hard-cuts Track A
3. Simultaneously fires Track B from its drop
4. The last vocal word from Track A echoes out over Track B (delay FX, 4-8 beats, decaying)
5. Result: The crowd thinks they're getting Track A's drop but gets SURPRISED by Track B's drop

**AI requirement:** Echo Out delay line + proper drop detection.

---

#### T4. Loop Roll → Drop
**When:** Building tension before a drop.
**Duration:** 4-8 beats of looping.
**How a real DJ does it:**
1. Finds a punchy vocal word ("Wait!", "Go!", "Fire!") — not a random breath
2. Hits the loop button on that word
3. Progressively halves the loop: 1 beat → 1/2 → 1/4 → 1/8
4. This creates a "machine gun" stutter that builds insane tension
5. Releases the loop EXACTLY on the drop of Track B

**AI requirement:** Onset detection to find the right word, not a random audio chunk.

---

#### T5. Mashup (A Vocals + B Beat)
**When:** Two tracks share similar energy/key and you want to combine the best of both.
**Duration:** 16-64 beats.
**How a real DJ does it:**
1. Plays Track A's vocals over Track B's drums+bass
2. Mutes Track A's drums+bass and Track B's vocals (to prevent clashing)
3. Uses sidechain ducking so the vocal sits ON TOP of the beat cleanly
4. After the mashup section, fully transitions into Track B

**AI requirement:** Sidechain ducking + harmonic key compatibility check.

---

### 5 Advanced Techniques (Tomorrowland Level)

> [!IMPORTANT]
> These are what separate a bedroom DJ from a festival headliner. They're not about mixing — they're about **performance, energy sculpting, and creating moments.**

#### T6. White Noise Riser / FX Sweep
**What it is:** A synthesized "whoooosh" sound that rises in pitch and intensity before a drop. Every single Tomorrowland set uses these.
**How to build it:**
```python
def generate_white_noise_riser(sr, duration_beats, bpm):
    duration_sec = (duration_beats / bpm) * 60
    num_samples = int(duration_sec * sr)
    
    # Generate white noise
    noise = np.random.uniform(-1, 1, num_samples).astype(np.float32)
    
    # Sweep a high-pass filter from 200Hz to 8000Hz (exponential)
    # + Volume envelope from 0.0 to 0.8
    volume_env = np.linspace(0.0, 0.8, num_samples) ** 2
    
    # Apply HPF sweep in chunks (like loop_roll.py's apply_highpass_sweep)
    filtered = apply_highpass_sweep(noise, sr, start_freq=200, end_freq=8000)
    
    return filtered * volume_env
```
**When to use:** Layer it underneath the last 8-16 beats before ANY drop swap. It subconsciously tells the crowd "something big is coming."

---

#### T7. Backspin / Spinback
**What it is:** Track A's audio rapidly plays in reverse (sounds like a record spinning backward), then Track B drops.
**How to build it:**
```python
def generate_spinback(audio_chunk, sr, duration_sec=1.0):
    num_samples = int(duration_sec * sr)
    chunk = audio_chunk[-num_samples:]  # Last N samples
    reversed_chunk = chunk[::-1]  # Reverse!
    
    # Apply pitch-down effect (decelerating playback speed)
    decel_curve = np.linspace(1.0, 0.0, num_samples) ** 0.5
    # Fade volume simultaneously
    reversed_chunk *= decel_curve
    
    return reversed_chunk
```
**When to use:** Instead of echo-out. More dramatic, more "DJ performance" feel. Used when you want to surprise the crowd with a genre switch.

---

#### T8. Double Drop
**What it is:** TWO tracks drop at the exact same time. Their drums layer, their bass combines, and the result is TWICE as heavy as either track alone.
**How to build it:**
- Play Track A's buildup normally
- At the drop point, play BOTH Track A's drop AND Track B's drop simultaneously
- Use sidechain to prevent muddiness
- After 16-32 beats, fade Track A out, leaving Track B

**When to use:** Peak energy moments. The crowd loses their mind because they recognize BOTH songs hitting at once.

---

#### T9. Energy Arc Management
**What it is:** Not a technique — it's a **strategy**. The AI must plan the entire set as a journey, not just individual transitions.

A Tomorrowland set follows this emotional arc:
```
Energy
  ▲
  │           ████                    ████████
  │        ███    ███              ███        ███
  │     ███          ███        ███              ███
  │  ███                ██████ ██                  █████
  │ ██                                                 ██
  │█                                                     █
  └──────────────────────────────────────────────────────────▶ Time
    Warmup    Build    Peak    Breather    FINAL PEAK    Cooldown
    (5 min)  (10 min) (15 min) (5 min)    (20 min)      (5 min)
```

**What the AI needs:**
- Each track gets an "energy score" (0-10) based on BPM, bass density, vocal presence
- The set planner arranges tracks to follow the arc
- Transitions are chosen based on the energy delta between tracks:
  - Small delta (+1 or -1): Bass Swap
  - Medium delta (+3): Drop Swap
  - Large delta (+5): Loop Roll + White Noise Riser → Drop Swap
  - Negative delta (-3): Echo Out → Long Blend into quieter track

---

#### T10. Harmonic Mixing (Camelot Wheel)
**What it is:** Tracks that are in compatible musical keys blend smoothly. Tracks in clashing keys sound terrible together.

The Camelot Wheel maps all 24 keys to a number+letter system:
```
Compatible moves:
  Same number (8A → 8A)     = Perfect match
  +1 number  (8A → 9A)      = Smooth energy lift
  -1 number  (8A → 7A)      = Smooth energy drop
  A ↔ B      (8A → 8B)      = Major/Minor swap (emotional shift)
```

**What we already have:** Key detection via chroma CQT.
**What we need:** A compatibility checker that scores track pairs and warns when keys clash.

```python
CAMELOT = {
    'C': '8B', 'C#': '3B', 'D': '10B', 'D#': '5B', 'E': '12B',
    'F': '7B', 'F#': '2B', 'G': '9B', 'G#': '4B', 'A': '11B',
    'A#': '6B', 'B': '1B'
}

def key_compatible(key_a, key_b):
    ca, cb = CAMELOT[key_a], CAMELOT[key_b]
    num_a, num_b = int(ca[:-1]), int(cb[:-1])
    letter_a, letter_b = ca[-1], cb[-1]
    
    if ca == cb: return 1.0  # Perfect
    if abs(num_a - num_b) <= 1 and letter_a == letter_b: return 0.9  # Great
    if num_a == num_b and letter_a != letter_b: return 0.8  # Mood shift
    if abs(num_a - num_b) == 2 and letter_a == letter_b: return 0.6  # Risky
    return 0.3  # Clash — avoid overlapping melodies
```

---

## Part 3: The "Feel" Layer — What Makes It Alive

> [!CAUTION]
> This is what separates a working DJ program from one that sounds like a Spotify crossfade. Without the feel layer, even perfect transitions sound robotic.

### 1. Never Cut Vocals Dead
Every vocal exit must use one of:
- **Echo Out** (delay tail, 4-8 beats, 0.5-0.65 feedback)
- **Reverb Wash** (long reverb tail, 2-4 seconds)
- **Natural Phrase End** (let the singer finish their word/sentence)

Rule: If the vocal RMS is above 0.05 at the cut point, apply Echo Out. Period.

### 2. Never Have Two Basses
At any given sample, only ONE track's bass should be above 0.1 amplitude. If both are playing, the mix sounds muddy and clips. The bass swap must be near-instant (2-4 beats max).

### 3. Always Transition on Phrase Boundaries
Music is built in blocks of 4, 8, 16, or 32 beats. Starting a transition on beat 7 of a 16-beat phrase sounds "off" even if the audio blends perfectly. The AI must snap all transition start points to the nearest downbeat of a 16-beat phrase.

### 4. Energy Must Flow, Not Jump
Don't go from a mellow verse directly into a screaming drop. The AI needs at least 8 beats of "preparation" — either a filter sweep, a drum fill, a loop roll, or a white noise riser — before any energy jump of +3 or more.

### 5. Silence Is a Tool
A half-second of silence before a massive drop makes the crowd hold their breath. The AI should insert a 0.5-beat silence gap (with a reverb tail filling it) before the highest-energy drops. This is called a "breakdown fake-out."

---

## Part 4: Implementation Roadmap

### Phase 1: Fix the Foundation
> [!IMPORTANT]
> Song Structure Segmentation fixes 3 broken techniques and enables 2 new ones. It's the highest-impact single change.

| Task | Details | Unblocks |
|---|---|---|
| Install `allin1` | `pip install allin1` — DL model for EDM structure detection | Everything |
| Analyze all tracks | Run `allin1` on every track, store segments in `crate.json` | T1, T2, T3, T5, T6 |
| Refactor transitions | Replace all file-length math with segment lookups | T1, T2 |

### Phase 2: Port & Fix (Free Wins)
| Task | Details | Fixes |
|---|---|---|
| Port `generate_echo_tail()` | Copy from [mixer.py L24-39](file:///C:/Projects/Cloudy%20DJ%202.0/mixer.py#L24-L39) to new engine | T3, T5 |
| Add Track B zero-mask | `out_audio[:transition_point] *= 0` for Track B | T5 leak |
| Fix Outro→Intro with segments | Use `outro` / `intro` labels instead of file-length | T1 |

### Phase 3: Core DSP
| Task | Details | Fixes |
|---|---|---|
| Build `apply_sidechain_duck()` | RMS-triggered gain ducking (-6dB) | T4, T5, T8 |
| Build `find_best_loop_source()` | `librosa.onset.onset_detect` → find punchiest word | T5 |
| Build `bass_swap_transition()` | 3-band split, instant bass swap, gradual mid/high fade | T1 (new) |

### Phase 4: Tomorrowland FX
| Task | Details | Enables |
|---|---|---|
| Build `generate_white_noise_riser()` | Synthesized HPF-swept noise riser | T6 |
| Build `generate_spinback()` | Reversed audio with deceleration | T7 |
| Build `key_compatible()` | Camelot wheel compatibility scoring | T10 |

### Phase 5: Intelligence Layer
| Task | Details | Enables |
|---|---|---|
| Energy scoring per track | BPM + bass RMS + vocal density → 0-10 score | T9 |
| Set planner | Arrange tracks following the energy arc curve | T9 |
| Technique selector | Auto-pick transition type based on energy delta + key compat | All |

### Phase 6: Validate & Ship
| Task | Details |
|---|---|
| Re-generate all transitions | Run all 5 core + 5 advanced on Secrets → Wild |
| A/B test against real DJ mixes | Compare extracted EQ envelopes with our generated ones |
| Full 10-track set | Generate a complete 10-minute set with automatic technique selection |

---

## Open Questions

> [!IMPORTANT]
> 1. Should we try `allin1` first, or build our own structure detection with librosa's recurrence matrix? `allin1` is faster to integrate but may not work on all EDM sub-genres.
> 2. For the energy arc — do you want to define the arc manually per set, or should the AI auto-generate it based on the track list?
> 3. Should we implement all 10 techniques before testing, or fix the 5 core first, validate they sound good, then add the Tomorrowland layer?
