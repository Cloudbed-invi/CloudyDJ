import soundfile as sf
import json

with open('library/crate.json') as f:
    crate = json.load(f)

for name in crate:
    if 'Secrets' in name or 'Wild' in name:
        stems = crate[name]['stems']
        info = sf.info(stems['drums'])
        print(f'{name[:30]}: sr={info.samplerate}, frames={info.frames}, duration={info.frames/info.samplerate:.1f}s')
