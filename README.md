# Cloudy DJ 2.0 - AI DJ Engine

Cloudy DJ 2.0 is an advanced, automated AI DJ engine that can download tracks, extract stems, analyze BPM/Key, beat-grid, and perform complex DJ transitions (Bass Swaps, Drop Swaps, Loop Rolls, and more). It mathematically reverse-engineers real DJ mixes to generate professional, Tomorrowland-level transitions.

## Features
- **Track Downloading & Stem Separation**: Automatically fetches from YouTube and splits into vocals, drums, bass, and other using Demucs.
- **BPM & Key Detection**: Uses librosa to beat-track and chroma CQT to find the exact pitch class.
- **Drop Detection & Time Stretching**: Mathematically detects drops using bass RMS contrast and perfectly time-stretches stems.
- **Advanced Transitions**: Capable of Drop Swaps, Bass Swaps, Loop Rolls, Echo Outs, and Multi-band EQ blends.

## Getting Started

### Prerequisites
1. Python 3.8+
2. FFmpeg installed and added to your PATH (or placed in the project directory).
3. Required Python packages:
   ```bash
   pip install numpy scipy librosa soundfile pydub yt-dlp shazamio
   ```

### Running the Engine

The project consists of multiple modules depending on your goal.

1. **Batch Generation / Analysis**:
   ```bash
   python generate_batch.py
   ```
   This will process the library, extract stems, detect BPM/Key, and prepare the `crate.json`.

2. **Transition Extraction**:
   To reverse-engineer a real DJ mix and extract EQ automation curves:
   ```bash
   python src/engine/extract_all_techniques.py
   ```

3. **Live Engine / Mixer**:
   Run the mixer to apply transitions to tracks:
   ```bash
   python mixer.py
   ```

## Licensing & Commercial Use

This software is dual-licensed to maximize open-source collaboration while protecting commercial interests.

**Open Source License:** 
This project is licensed under the **GNU Affero General Public License v3.0 (AGPLv3)**. This means if you use this code in a project or run it as a service over a network (e.g., a SaaS AI DJ app), you **must** open-source your entire project's code under the same license. 

**Commercial License:**
If you wish to use Cloudy DJ 2.0 in a closed-source commercial product or service without distributing your own source code, you must purchase a commercial license. Please contact the repository owner to arrange a commercial usage agreement.
